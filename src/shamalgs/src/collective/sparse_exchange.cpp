// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sparse_exchange.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/collective/sparse_exchange.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/worldInfo.hpp"
#include <stdexcept>

namespace shamalgs::collective {

    CommMessageInfo unpack(u64_2 comm_info) {
        u64 comm_vec        = comm_info.x();
        size_t message_size = comm_info.y();
        u32_2 comm_ranks    = sham::unpack32(comm_vec);
        u32 sender          = comm_ranks.x();
        u32 receiver        = comm_ranks.y();

        if (message_size == 0) {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "Message size is 0 for rank {}, sender = {}, receiver = {}",
                shamcomm::world_rank(),
                sender,
                receiver));
        }

        return CommMessageInfo{
            message_size,
            static_cast<i32>(sender),
            static_cast<i32>(receiver),
            std::nullopt,
            std::nullopt,
            std::nullopt};
    };

    CommTable build_sparse_exchange_table(const std::vector<CommMessageInfo> &messages_send) {
        __shamrock_stack_entry();

        ////////////////////////////////////////////////////////////
        // Pack the local data then allgatherv to get the global data
        ////////////////////////////////////////////////////////////

        std::vector<u64_2> local_data = std::vector<u64_2>(messages_send.size());

        for (size_t i = 0; i < messages_send.size(); i++) {
            u32 sender          = static_cast<u32>(messages_send[i].rank_sender);
            u32 receiver        = static_cast<u32>(messages_send[i].rank_receiver);
            size_t message_size = messages_send[i].message_size;

            if (sender != shamcomm::world_rank()) {
                throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                    "You are trying to send a message from a rank that does not posses it\n"
                    "    sender = {}, receiver = {}, world_rank = {}",
                    sender,
                    receiver,
                    shamcomm::world_rank()));
            }

            local_data[i] = u64_2{sham::pack32(sender, receiver), message_size};
        }

        std::vector<u64_2> global_data;
        vector_allgatherv(local_data, global_data, MPI_COMM_WORLD);

        ////////////////////////////////////////////////////////////
        // Unpack the global data and build the global message list
        ////////////////////////////////////////////////////////////

        std::vector<CommMessageInfo> message_all(global_data.size());

        std::vector<i32> tag_map(shamcomm::world_size(), 0);

        u32 send_idx = 0;
        u32 recv_idx = 0;

        size_t recv_offset = 0;
        size_t send_offset = 0;
        for (u64 i = 0; i < global_data.size(); i++) {
            auto message_info = unpack(global_data[i]);

            auto sender   = message_info.rank_sender;
            auto receiver = message_info.rank_receiver;

            i32 &tag_map_ref = tag_map[static_cast<size_t>(sender)];

            i32 tag = tag_map_ref;
            tag_map_ref++;

            message_info.message_tag = tag;

            if (sender == shamcomm::world_rank()) {
                message_info.message_bytebuf_offset_send = send_offset;
                send_offset += message_info.message_size;
                send_idx++;
            }

            if (receiver == shamcomm::world_rank()) {
                message_info.message_bytebuf_offset_recv = recv_offset;
                recv_offset += message_info.message_size;
                recv_idx++;
            }

            message_all[i] = message_info;
        }

        ////////////////////////////////////////////////////////////
        // now that all comm were computed we can build the send and recv message lists
        ////////////////////////////////////////////////////////////

        std::vector<CommMessageInfo> ret_message_send(send_idx);
        std::vector<CommMessageInfo> ret_message_recv(recv_idx);

        std::vector<size_t> send_message_global_ids(send_idx);
        std::vector<size_t> recv_message_global_ids(recv_idx);

        send_idx = 0;
        recv_idx = 0;

        for (size_t i = 0; i < message_all.size(); i++) {
            auto message_info = message_all[i];
            if (message_info.rank_sender == shamcomm::world_rank()) {

                // the sender shoudl have set the offset for all messages, otherwise throw
                auto expected_offset = shambase::get_check_ref(
                    messages_send.at(send_idx).message_bytebuf_offset_send);

                // check that the send offset match for good measure
                if (message_info.message_bytebuf_offset_send != expected_offset) {
                    throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                        "The sender has not set the offset for all messages, otherwise throw\n"
                        "    expected_offset = {}, actual_offset = {}",
                        expected_offset,
                        message_info.message_bytebuf_offset_send));
                }

                ret_message_send[send_idx]        = message_info;
                send_message_global_ids[send_idx] = i;
                send_idx++;
            }
            if (message_info.rank_receiver == shamcomm::world_rank()) {
                ret_message_recv[recv_idx]        = message_info;
                recv_message_global_ids[recv_idx] = i;
                recv_idx++;
            }
        }

        return CommTable{
            ret_message_send,
            message_all,
            ret_message_recv,
            send_message_global_ids,
            recv_message_global_ids,
            send_offset,
            recv_offset};
    }

    void sparse_exchange(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        sham::DeviceBuffer<u8> &bytebuffer_send,
        sham::DeviceBuffer<u8> &bytebuffer_recv,
        const CommTable &comm_table) {

        __shamrock_stack_entry();
    }

} // namespace shamalgs::collective
