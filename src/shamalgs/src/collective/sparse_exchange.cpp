// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sparse_exchange.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/collective/sparse_exchange.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/narrowing.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/RequestList.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shambackends/USMPtrHolder.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/worldInfo.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
namespace shamalgs::collective {

    namespace {

        /// gathered message data of every rank
        struct GlobalMessageData {
            std::vector<u64_2> global_data; ///< packed {pack32(sender, receiver), size}
            std::vector<int> displs;        ///< start of the block of each rank in global_data
        };

        /// allgather the packed (sender, receiver, size) triples of every rank
        GlobalMessageData fetch_global_message_data(
            const std::vector<CommMessageInfo> &messages_send) {
            __shamrock_stack_entry();

            std::vector<u64_2> local_data = std::vector<u64_2>(messages_send.size());

            for (size_t i = 0; i < messages_send.size(); i++) {
                u32 sender          = static_cast<u32>(messages_send[i].rank_sender);
                u32 receiver        = static_cast<u32>(messages_send[i].rank_receiver);
                size_t message_size = messages_send[i].message_size;

                if (sender != shamcomm::world_rank()) {
                    throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                        "You are trying to send a message from a rank that does not posses it\n"
                        "    sender = {}, receiver = {}, world_rank = {}",
                        sender,
                        receiver,
                        shamcomm::world_rank()));
                }

                local_data[i] = u64_2{sham::pack32(sender, receiver), message_size};
            }

            GlobalMessageData ret{};
            ret.displs = vector_allgatherv(local_data, ret.global_data, MPI_COMM_WORLD);

            return ret;
        }

        /// place a message of size message_size at the end of the current buffer, or in a new
        /// buffer if it does not fit
        CommMessageBufOffset place_in_buffers(
            std::vector<size_t> &buf_sizes,
            size_t &current_offset,
            size_t message_size,
            size_t max_alloc_size) {

            if (message_size > max_alloc_size) {
                throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                    "Message size is greater than the max alloc size\n"
                    "    message_size = {}, max_alloc_size = {}",
                    message_size,
                    max_alloc_size));
            }

            if (buf_sizes.size() == 0) {
                buf_sizes.push_back(0);
            }

            if (current_offset + message_size >= max_alloc_size) {
                current_offset = 0;
                buf_sizes.push_back(0);
            }

            CommMessageBufOffset ret{.buf_id = buf_sizes.size() - 1, .data_offset = current_offset};
            current_offset += message_size;
            buf_sizes.back() += message_size;

            return ret;
        }

    } // namespace

    CommTable details::build_sparse_exchange_table_from_global(
        const std::vector<u64_2> &global_data,
        const std::vector<int> &displs,
        i32 world_rank,
        size_t max_alloc_size) {
        __shamrock_stack_entry();

        CommTable ret{};

        const u64 rank = static_cast<u32>(world_rank);

        size_t send_offset = 0;
        size_t recv_offset = 0;

        for (size_t i = 0; i < global_data.size(); i++) {
            const u64 comm_vec = global_data[i].x();
            const u64 sender   = comm_vec >> 32U;
            const u64 receiver = comm_vec & 0xFFFFFFFFU;

            if (sender != rank && receiver != rank) {
                continue;
            }

            size_t message_size = global_data[i].y();

            if (message_size == 0) {
                throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                    "Message size is 0 for rank {}, sender = {}, receiver = {}",
                    world_rank,
                    sender,
                    receiver));
            }

            // vector_allgatherv concatenates the messages of each rank in rank order, hence the
            // running per-sender message index is the offset within the sender's block
            i32 tag = shambase::narrow_or_throw<i32>(i - static_cast<size_t>(displs.at(sender)));

            CommMessageInfo message_info{
                .message_size                = message_size,
                .rank_sender                 = static_cast<i32>(sender),
                .rank_receiver               = static_cast<i32>(receiver),
                .message_tag                 = tag,
                .message_bytebuf_offset_send = std::nullopt,
                .message_bytebuf_offset_recv = std::nullopt};

            if (sender == rank) {
                message_info.message_bytebuf_offset_send = place_in_buffers(
                    ret.send_total_sizes, send_offset, message_size, max_alloc_size);
            }

            if (receiver == rank) {
                message_info.message_bytebuf_offset_recv = place_in_buffers(
                    ret.recv_total_sizes, recv_offset, message_size, max_alloc_size);
            }

            if (sender == rank) {
                ret.messages_send.push_back(message_info);
                ret.send_message_global_ids.push_back(i);
            }

            if (receiver == rank) {
                ret.messages_recv.push_back(message_info);
                ret.recv_message_global_ids.push_back(i);
            }
        }

        return ret;
    }

    CommTable build_sparse_exchange_table(
        const std::vector<CommMessageInfo> &messages_send, size_t max_alloc_size) {
        __shamrock_stack_entry();

        GlobalMessageData gathered = fetch_global_message_data(messages_send);

        return details::build_sparse_exchange_table_from_global(
            gathered.global_data, gathered.displs, shamcomm::world_rank(), max_alloc_size);
    }

    void sparse_exchange(
        const std::shared_ptr<sham::DeviceScheduler> &dev_sched,
        const std::vector<const u8 *> &bytebuffer_send,
        const std::vector<u8 *> &bytebuffer_recv,
        const CommTable &comm_table) {

        __shamrock_stack_entry();

        u32 SHAM_SPARSE_COMM_INFLIGHT_LIM = 128; // TODO: use the env variable

        const auto &messages_send = comm_table.messages_send;
        const auto &messages_recv = comm_table.messages_recv;
        const auto &send_ids      = comm_table.send_message_global_ids;
        const auto &recv_ids      = comm_table.recv_message_global_ids;

        if (send_ids.size() != messages_send.size() || recv_ids.size() != messages_recv.size()) {
            throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                "The comm table global ids do not match its messages\n"
                "    messages_send = {}, send_message_global_ids = {}\n"
                "    messages_recv = {}, recv_message_global_ids = {}",
                messages_send.size(),
                send_ids.size(),
                messages_recv.size(),
                recv_ids.size()));
        }

        // Post the local sends and recvs merged by global message id, so that every rank posts
        // its requests in the same global order. The in-flight limiter below relies on this
        // ordering to avoid deadlocks.
        RequestList rqs;
        size_t i_send = 0;
        size_t i_recv = 0;
        while (i_send < messages_send.size() || i_recv < messages_recv.size()) {

            size_t gid_send = (i_send < messages_send.size()) ? send_ids[i_send] : SIZE_MAX;
            size_t gid_recv = (i_recv < messages_recv.size()) ? recv_ids[i_recv] : SIZE_MAX;
            size_t gid      = std::min(gid_send, gid_recv);

            if (gid_send == gid) {
                const auto &message_info = messages_send[i_send];
                auto off_info = shambase::get_check_ref(message_info.message_bytebuf_offset_send);
                auto ptr      = bytebuffer_send.at(off_info.buf_id) + off_info.data_offset;
                auto &rq      = rqs.new_request();
                shamcomm::mpi::Isend(
                    ptr,
                    shambase::narrow_or_throw<i32>(message_info.message_size),
                    MPI_BYTE,
                    message_info.rank_receiver,
                    shambase::get_check_ref(message_info.message_tag),
                    MPI_COMM_WORLD,
                    &rq);
                i_send++;
            }

            if (gid_recv == gid) {
                const auto &message_info = messages_recv[i_recv];
                auto off_info = shambase::get_check_ref(message_info.message_bytebuf_offset_recv);
                auto ptr      = bytebuffer_recv.at(off_info.buf_id) + off_info.data_offset;
                auto &rq      = rqs.new_request();
                shamcomm::mpi::Irecv(
                    ptr,
                    shambase::narrow_or_throw<i32>(message_info.message_size),
                    MPI_BYTE,
                    message_info.rank_sender,
                    shambase::get_check_ref(message_info.message_tag),
                    MPI_COMM_WORLD,
                    &rq);
                i_recv++;
            }

            rqs.spin_lock_partial_wait(SHAM_SPARSE_COMM_INFLIGHT_LIM, 120, 10);
        }
        rqs.wait_all();
    }

    template<sham::USMKindTarget target>
    void sparse_exchange(
        const std::shared_ptr<sham::DeviceScheduler> &dev_sched,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, target>>> &bytebuffer_send,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, target>>> &bytebuffer_recv,
        const CommTable &comm_table) {

        __shamrock_stack_entry();

        if (&bytebuffer_send == &bytebuffer_recv) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "In-place sparse_exchange is not supported. Send and receive buffers must be "
                "distinct.");
        }

        if (comm_table.send_total_sizes.size() != bytebuffer_send.size()) {
            throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                "The send total size is greater than the send buffer size\n"
                "    send_total_sizes = {}, send_buffer_size = {}",
                comm_table.send_total_sizes.size(),
                bytebuffer_send.size()));
        }

        if (comm_table.recv_total_sizes.size() != bytebuffer_recv.size()) {
            throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                "The recv total size is greater than the recv buffer size\n"
                "    recv_total_sizes = {}, recv_buffer_size = {}",
                comm_table.recv_total_sizes.size(),
                bytebuffer_recv.size()));
        }

        for (size_t i = 0; i < comm_table.send_total_sizes.size(); i++) {
            if (comm_table.send_total_sizes[i] > bytebuffer_send[i]->get_size()) {
                throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                    "The send total size is greater than the send buffer size\n"
                    "    send_total_sizes = {}, send_buffer_size = {}, buf_id = {}",
                    comm_table.send_total_sizes[i],
                    bytebuffer_send[i]->get_size(),
                    i));
            }
        }

        for (size_t i = 0; i < comm_table.recv_total_sizes.size(); i++) {
            if (comm_table.recv_total_sizes[i] > bytebuffer_recv[i]->get_size()) {
                throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                    "The recv total size is greater than the recv buffer size\n"
                    "    recv_total_sizes = {}, recv_buffer_size = {}, buf_id = {}",
                    comm_table.recv_total_sizes[i],
                    bytebuffer_recv[i]->get_size(),
                    i));
            }
        }

        bool direct_gpu_capable = dev_sched->ctx->device->mpi_prop.is_mpi_direct_capable;

        if (!direct_gpu_capable && target == sham::device) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "You are trying to use a device buffer on the device but the device is not "
                "direct "
                "GPU capable");
        }

        std::vector<const u8 *> send_ptrs(bytebuffer_send.size());
        std::vector<u8 *> recv_ptrs(bytebuffer_recv.size());

        sham::EventList depends_list;
        for (size_t i = 0; i < bytebuffer_send.size(); i++) {
            send_ptrs[i]
                = shambase::get_check_ref(bytebuffer_send[i]).get_read_access(depends_list);
        }

        for (size_t i = 0; i < bytebuffer_recv.size(); i++) {
            recv_ptrs[i]
                = shambase::get_check_ref(bytebuffer_recv[i]).get_write_access(depends_list);
        }
        depends_list.wait();

        sparse_exchange(dev_sched, send_ptrs, recv_ptrs, comm_table);

        for (size_t i = 0; i < bytebuffer_send.size(); i++) {
            shambase::get_check_ref(bytebuffer_send[i]).complete_event_state(sycl::event{});
        }

        for (size_t i = 0; i < bytebuffer_recv.size(); i++) {
            shambase::get_check_ref(bytebuffer_recv[i]).complete_event_state(sycl::event{});
        }
    }

    // template instantiations
    template void sparse_exchange<sham::device>(
        const std::shared_ptr<sham::DeviceScheduler> &dev_sched,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, sham::device>>> &bytebuffer_send,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, sham::device>>> &bytebuffer_recv,
        const CommTable &comm_table);

    template void sparse_exchange<sham::host>(
        const std::shared_ptr<sham::DeviceScheduler> &dev_sched,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, sham::host>>> &bytebuffer_send,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, sham::host>>> &bytebuffer_recv,
        const CommTable &comm_table);

} // namespace shamalgs::collective
