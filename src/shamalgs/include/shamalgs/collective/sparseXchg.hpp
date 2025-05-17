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
#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/serialize.hpp"
#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include <string_view>
#include <functional>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <thread>
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

        auto get_hash_comm_map = [](const std::vector<u64> &vec) {
            std::string s = "";
            s.resize(vec.size() * sizeof(u64));
            std::memcpy(s.data(), vec.data(), vec.size() * sizeof(u64));
            auto ret = std::hash<std::string>{}(s);
            return ret;
        };

        auto check_comm_hash = [&]() {
            auto hash = get_hash_comm_map(global_comm_ranks);
            // logger::raw_ln("global_comm_ranks hash", hash);

            auto max_hash = allreduce_max(hash);
            auto min_hash = allreduce_min(hash);

            if (max_hash != min_hash) {
                std::string msg = shambase::format(
                    "hash mismatch {} != {}, local hash = {}", max_hash, min_hash, hash);
                logger::err_ln("Sparse comm", msg);
                MPI_Barrier(MPI_COMM_WORLD);
                shambase::throw_with_loc<std::runtime_error>(msg);
            }
        };

        check_comm_hash();

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

        // MPI_Barrier(MPI_COMM_WORLD);
        // if (shamcomm::world_rank() == 0) {
        //     logger::raw_ln(shambase::format("sparse comm start"));
        // }
        // MPI_Barrier(MPI_COMM_WORLD);

        // note the tag cannot be bigger than max_i32 because of the allgatherv

        // if(message_send.size() > 1 && shamcomm::world_rank() == 11){
        //     //sleep
        //     std::this_thread::sleep_for(std::chrono::seconds(30));
        // }

        struct rq_info {
            u32 sender;
            u32 receiver;
            u64 size;
            u32 tag;
            bool is_send;
            bool is_recv;
        };

        std::vector<MPI_Request> rqs;
        std::vector<rq_info> rqs_infos;

        // send step
        u32 send_idx = 0;
        for (u32 i = 0; i < global_comm_ranks.size(); i++) {
            u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

            if (comm_ranks.x() == shamcomm::world_rank()) {

                auto &payload = message_send[send_idx].payload;

                rqs.push_back(MPI_Request{});
                u32 rq_index = rqs.size() - 1;
                auto &rq     = rqs[rq_index];

                rqs_infos.push_back(
                    {comm_ranks.x(), comm_ranks.y(), payload->get_bytesize(), i, true, false});

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

            //  }
            //
            //     // recv step
            //    for (u32 i = 0; i < global_comm_ranks.size(); i++) {
            //      u32_2 comm_ranks = sham::unpack32(global_comm_ranks[i]);

            if (comm_ranks.y() == shamcomm::world_rank()) {

                RecvPayload payload;
                payload.sender_ranks = comm_ranks.x();

                rqs.push_back(MPI_Request{});
                u32 rq_index = rqs.size() - 1;
                auto &rq     = rqs[rq_index];

                rqs_infos.push_back(
                    {comm_ranks.x(), comm_ranks.y(), u64(comm_sizes[i]), i, false, true});

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

        auto test_event_completions = [&]() {
            shambase::Timer twait;
            twait.start();
            f64 timeout_t = 20;

            std::vector<bool> done_map = {};
            done_map.resize(rqs.size());
            for (u32 i = 0; i < rqs.size(); i++) {
                done_map[i] = false;
            }

            bool done = false;
            while (!done) {
                bool loc_done = true;
                for (u32 i = 0; i < rqs.size(); i++) {
                    if (done_map[i]) {
                        continue;
                    }

                    auto &rq = rqs[i];

                    MPI_Status st;
                    int ready;
                    MPICHECK(MPI_Test(&rq, &ready, MPI_STATUS_IGNORE));
                    if (!ready) {
                        loc_done = false;
                        // logger::raw_ln(shambase::format(
                        //     "communication pending : send {} -> {} tag {} size {}",
                        //     rqs_infos[i].sender,
                        //     rqs_infos[i].receiver,
                        //     rqs_infos[i].tag,
                        //     rqs_infos[i].size));
                    } else {
                        done_map[i] = true;
                        // logger::raw_ln(shambase::format(
                        //     "communication done : send {} -> {} tag {} size {}",
                        //     rqs_infos[i].sender,
                        //     rqs_infos[i].receiver,
                        //     rqs_infos[i].tag,
                        //     rqs_infos[i].size));
                    }
                }
                if (loc_done) {
                    done = true;
                }

                twait.end();
                if (twait.elasped_sec() > timeout_t) {
                    std::string err_msg = "";
                    for (u32 i = 0; i < rqs.size(); i++) {
                        if (!done_map[i]) {

                            if (rqs_infos[i].is_send) {
                                err_msg += shambase::format(
                                    "communication timeout : send {} -> {} tag {} size {}\n",
                                    rqs_infos[i].sender,
                                    rqs_infos[i].receiver,
                                    rqs_infos[i].tag,
                                    rqs_infos[i].size);
                            } else {
                                err_msg += shambase::format(
                                    "communication timeout : recv {} -> {} tag {} size {}\n",
                                    rqs_infos[i].sender,
                                    rqs_infos[i].receiver,
                                    rqs_infos[i].tag,
                                    rqs_infos[i].size);
                            }
                        }
                    }
                    std::string msg = shambase::format("communication timeout : \n{}", err_msg);
                    logger::err_ln("Sparse comm", msg);
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    shambase::throw_with_loc<std::runtime_error>(msg);
                }
            }
        };

        test_event_completions();

        std::vector<MPI_Status> st_lst(rqs.size());
        MPICHECK(MPI_Waitall(rqs.size(), rqs.data(), st_lst.data()));

        // MPI_Barrier(MPI_COMM_WORLD);
        // if (shamcomm::world_rank() == 0) {
        //     logger::raw_ln(shambase::format("sparse comm done"));
        // }
        // MPI_Barrier(MPI_COMM_WORLD);
    }

    inline void sparse_comm_c(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        const std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        const SparseCommTable &comm_table) {
        // sparse_comm_debug_infos(dev_sched, message_send, message_recv, comm_table);
        //    sparse_comm_isend_probe_count_irecv(dev_sched, message_send, message_recv,
        //    comm_table);
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

    inline void base_sparse_comm_max_comm(
        std::shared_ptr<sham::DeviceScheduler> dev_sched,
        std::vector<SendPayload> &message_send,
        std::vector<RecvPayload> &message_recv,
        u32 max_simultaneous_send) {

        int send_loc = message_send.size();
        int send_max_count;
        MPI_Allreduce(&send_loc, &send_max_count, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // logger::raw_ln(send_loc, send_max_count);

        StackEntry stack_loc{};

        int i = 0;
        while (i < send_max_count) {

            if (i > 0) {
                ON_RANK_0(
                    logger::warn_ln("SparseComm", "Splitted sparse comm", i, "/", send_max_count));
            }

            std::vector<SendPayload> message_send_tmp;
            std::vector<RecvPayload> message_recv_tmp;

            for (int j = i; (j < (i + max_simultaneous_send)) && (j < message_send.size()); j++) {
                // logger::raw_ln("emplace message", j);
                message_send_tmp.emplace_back(std::move(message_send[j]));
            }

            base_sparse_comm(dev_sched, message_send_tmp, message_recv_tmp);

            for (int j = 0; j < message_recv_tmp.size(); j++) {
                message_recv.emplace_back(std::move(message_recv_tmp[j]));
            }

            i += max_simultaneous_send;
        }
    }

} // namespace shamalgs::collective
