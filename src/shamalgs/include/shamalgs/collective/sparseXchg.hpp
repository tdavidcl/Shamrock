// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sparseXchg.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/assert.hpp"
#include "shambase/integer.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/serialize.hpp"
#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include <mpi.h>
#include <string>
#include <vector>

namespace shamalgs::collective {

    struct SendPayload {
        i32 receiver_rank;
        std::unique_ptr<shamcomm::CommunicationBuffer> payload;
    };

    struct RecvPayload {
        i32 sender_ranks; // should not be plural
        std::unique_ptr<shamcomm::CommunicationBuffer> payload;
    };

    struct SparseCommTable {
        std::vector<u64> local_send_vec_comm_ranks;
        std::vector<u64> global_comm_ranks;

        void build(const std::vector<SendPayload> &message_send) {
            StackEntry stack_loc{};

            local_send_vec_comm_ranks.resize(message_send.size());

            i32 iterator = 0;
            for (u64 i = 0; i < message_send.size(); i++) {
                local_send_vec_comm_ranks[i]
                    = sham::pack32(shamcomm::world_rank(), message_send[i].receiver_rank);
            }

            vector_allgatherv(local_send_vec_comm_ranks, global_comm_ranks, MPI_COMM_WORLD);
        }
    };

    inline void sparse_comm_debug_infos(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        const SparseCommTable &comm_table) {
        StackEntry stack_loc{};

        // share comm list accros nodes
        const std::vector<u64> &send_vec_comm_ranks = comm_table.local_send_vec_comm_ranks;
        const std::vector<u64> &global_comm_ranks   = comm_table.global_comm_ranks;

        // Utility lambda for printing comm matrix
        auto print_comm_mat = [&]() {
            StackEntry stack_loc{};

            MPI_Barrier(MPI_COMM_WORLD);
            std::string accum = "";

            u32 send_idx = 0;
            for (u32 i = 0; i < global_comm_ranks.size(); i++) {
                u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

                if (comm_ranks.x() == shamcomm::world_rank()) {
                    accum += shambase::format(
                        "{} # {} # {}\n",
                        comm_ranks.x(),
                        comm_ranks.y(),
                        message_send[send_idx].payload->get_bytesize());

                    send_idx++;
                }
            }

            std::string matrix;
            shamcomm::gather_str(accum, matrix);

            matrix = "\n" + matrix;

            if (shamcomm::world_rank() == 0) {
                logger::raw_ln("comm matrix:", matrix);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        };

        // Enable this only to do debug
        print_comm_mat();

        auto show_alloc_state = [&]() {
            StackEntry stack_loc{};
            sham::MemPerfInfos mem_perf_infos_end = sham::details::get_mem_perf_info();

            std::string accum = shambase::format(
                "rank = {} maxmem = {}\n",
                shamcomm::world_rank(),
                shambase::readable_sizeof(mem_perf_infos_end.max_allocated_byte_device));

            MPI_Barrier(MPI_COMM_WORLD);
            std::string log;
            shamcomm::gather_str(accum, log);

            log = "\n" + log;

            if (shamcomm::world_rank() == 0) {
                logger::raw_ln("alloc state:", log);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        };

        // Enable this only to do debug
        show_alloc_state();
    }

    inline void sparse_comm_isend_probe_count_irecv(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        const SparseCommTable &comm_table) {
        StackEntry stack_loc{};

        // share comm list accros nodes
        const std::vector<u64> &send_vec_comm_ranks = comm_table.local_send_vec_comm_ranks;
        const std::vector<u64> &global_comm_ranks   = comm_table.global_comm_ranks;

        // Utility lambda for error reporting
        auto check_payload_size_is_int = [&](u64 bytesz) {
            u64 payload_sz = bytesz;

            if (payload_sz > std::numeric_limits<i32>::max()) {

                std::vector<u64> send_sizes;
                for (u32 i = 0; i < global_comm_ranks.size(); i++) {
                    u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

                    if (comm_ranks.x() == shamcomm::world_rank()) {
                        send_sizes.push_back(payload_sz);
                    }
                }

                shambase::throw_with_loc<std::runtime_error>(shambase::format(
                    "payload size {} is too large for MPI (max i32 is {})\n"
                    "message sizes to send: {}",
                    payload_sz,
                    std::numeric_limits<i32>::max(),
                    send_sizes));
            }

            return (i32) payload_sz;
        };

        // note the tag cannot be bigger than max_i32 because of the allgatherv

        std::vector<MPI_Request> rqs;

        // send step
        u32 send_idx = 0;
        for (u32 i = 0; i < global_comm_ranks.size(); i++) {
            u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

            if (comm_ranks.x() == shamcomm::world_rank()) {

                auto &payload = message_send[send_idx].payload;

                rqs.push_back(MPI_Request{});
                u32 rq_index = rqs.size() - 1;
                auto &rq     = rqs[rq_index];

                int send_sz = check_payload_size_is_int(payload->get_bytesize());

                // logger::raw_ln(shambase::format(
                //     "[{}] send {} bytes to rank {}, tag {}",
                //     shamcomm::world_rank(),
                //     payload->get_bytesize(),
                //     comm_ranks.y(),
                //     i));

                MPICHECK(MPI_Isend(
                    payload->get_ptr(), send_sz, MPI_BYTE, comm_ranks.y(), i, MPI_COMM_WORLD, &rq));

                send_idx++;
            }
        }

        // recv step
        for (u32 i = 0; i < global_comm_ranks.size(); i++) {
            u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

            if (comm_ranks.y() == shamcomm::world_rank()) {

                RecvPayload payload;
                payload.sender_ranks = comm_ranks.x();

                rqs.push_back(MPI_Request{});
                u32 rq_index = rqs.size() - 1;
                auto &rq     = rqs[rq_index];

                MPI_Status st;
                i32 cnt;
                MPICHECK(MPI_Probe(comm_ranks.x(), i, MPI_COMM_WORLD, &st));
                MPICHECK(MPI_Get_count(&st, MPI_BYTE, &cnt));

                payload.payload = std::make_unique<shamcomm::CommunicationBuffer>(cnt, dev_sched);

                // logger::raw_ln(shambase::format(
                //     "[{}] recv {} bytes from rank {}, tag {}",
                //     shamcomm::world_rank(),
                //     cnt,
                //     comm_ranks.x(),
                //     i));

                MPICHECK(MPI_Irecv(
                    payload.payload->get_ptr(),
                    cnt,
                    MPI_BYTE,
                    comm_ranks.x(),
                    i,
                    MPI_COMM_WORLD,
                    &rq));

                message_recv.push_back(std::move(payload));
            }
        }

        std::vector<MPI_Status> st_lst(rqs.size());
        MPICHECK(MPI_Waitall(rqs.size(), rqs.data(), st_lst.data()));
    }

    inline void sparse_comm_allgather_isend_irecv(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        const SparseCommTable &comm_table) {
        StackEntry stack_loc{};

        // share comm list accros nodes
        const std::vector<u64> &send_vec_comm_ranks = comm_table.local_send_vec_comm_ranks;
        const std::vector<u64> &global_comm_ranks   = comm_table.global_comm_ranks;

        // Utility lambda for error reporting
        auto check_payload_size_is_int = [&](u64 bytesz) {
            u64 payload_sz = bytesz;

            if (payload_sz > std::numeric_limits<i32>::max()) {

                std::vector<u64> send_sizes;
                for (u32 i = 0; i < global_comm_ranks.size(); i++) {
                    u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

                    if (comm_ranks.x() == shamcomm::world_rank()) {
                        send_sizes.push_back(payload_sz);
                    }
                }

                shambase::throw_with_loc<std::runtime_error>(shambase::format(
                    "payload size {} is too large for MPI (max i32 is {})\n"
                    "message sizes to send: {}",
                    payload_sz,
                    std::numeric_limits<i32>::max(),
                    send_sizes));
            }

            return (i32) payload_sz;
        };

        // Build global comm size table
        std::vector<int> comm_sizes_loc = {};
        comm_sizes_loc.resize(message_send.size());
        for (u64 i = 0; i < message_send.size(); i++) {
            comm_sizes_loc[i] = check_payload_size_is_int(message_send[i].payload->get_bytesize());
        }

        std::vector<int> comm_sizes = {};
        vector_allgatherv(comm_sizes_loc, comm_sizes, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        if (shamcomm::world_rank() == 0) {
            logger::raw_ln(shambase::format("sparse comm start"));
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // note the tag cannot be bigger than max_i32 because of the allgatherv

        std::vector<MPI_Request> rqs;

        // send step
        u32 send_idx = 0;
        for (u32 i = 0; i < global_comm_ranks.size(); i++) {
            u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

            if (comm_ranks.x() == shamcomm::world_rank()) {

                auto &payload = message_send[send_idx].payload;

                rqs.push_back(MPI_Request{});
                u32 rq_index = rqs.size() - 1;
                auto &rq     = rqs[rq_index];

                SHAM_ASSERT(payload->get_bytesize() == comm_sizes_loc[send_idx]);

                // logger::raw_ln(shambase::format(
                //     "[{}] send {} bytes to rank {}, tag {}",
                //     shamcomm::world_rank(),
                //     payload->get_bytesize(),
                //     comm_ranks.y(),
                //     i));

                MPICHECK(MPI_Isend(
                    payload->get_ptr(),
                    comm_sizes_loc[send_idx],
                    MPI_BYTE,
                    comm_ranks.y(),
                    i,
                    MPI_COMM_WORLD,
                    &rq));

                send_idx++;
            }
        }

        // recv step
        for (u32 i = 0; i < global_comm_ranks.size(); i++) {
            u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

            if (comm_ranks.y() == shamcomm::world_rank()) {

                RecvPayload payload;
                payload.sender_ranks = comm_ranks.x();

                rqs.push_back(MPI_Request{});
                u32 rq_index = rqs.size() - 1;
                auto &rq     = rqs[rq_index];

                i32 cnt = comm_sizes[i];

                payload.payload = std::make_unique<shamcomm::CommunicationBuffer>(cnt, dev_sched);

                // logger::raw_ln(shambase::format(
                //     "[{}] recv {} bytes from rank {}, tag {}",
                //     shamcomm::world_rank(),
                //     cnt,
                //     comm_ranks.x(),
                //     i));

                MPICHECK(MPI_Irecv(
                    payload.payload->get_ptr(),
                    cnt,
                    MPI_BYTE,
                    comm_ranks.x(),
                    i,
                    MPI_COMM_WORLD,
                    &rq));

                message_recv.push_back(std::move(payload));
            }
        }

        std::vector<MPI_Status> st_lst(rqs.size());
        MPICHECK(MPI_Waitall(rqs.size(), rqs.data(), st_lst.data()));

        MPI_Barrier(MPI_COMM_WORLD);
        if (shamcomm::world_rank() == 0) {
            logger::raw_ln(shambase::format("sparse comm done"));
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    inline void sparse_comm_c(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        const SparseCommTable &comm_table) {
        // sparse_comm_debug_infos(dev_sched, message_send, message_recv, comm_table);
        //   sparse_comm_isend_probe_count_irecv(dev_sched, message_send, message_recv, comm_table);
        sparse_comm_allgather_isend_irecv(dev_sched, message_send, message_recv, comm_table);
    }

    inline void base_sparse_comm(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv) {
        StackEntry stack_loc{};

        SparseCommTable comm_table;

        comm_table.build(message_send);

        sparse_comm_c(dev_sched, message_send, message_recv, comm_table);
    }

} // namespace shamalgs::collective
