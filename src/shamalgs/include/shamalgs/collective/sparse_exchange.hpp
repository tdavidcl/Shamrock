// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sparse_exchange.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/typeAliasVec.hpp"
#include <optional>
#include <vector>

namespace shamalgs::collective {

    struct CommMessageBufOffset {
        size_t buf_id;
        size_t data_offset;

        friend bool operator==(const CommMessageBufOffset &a, const CommMessageBufOffset &b) {
            return a.buf_id == b.buf_id && a.data_offset == b.data_offset;
        }
        friend bool operator!=(const CommMessageBufOffset &a, const CommMessageBufOffset &b) {
            return !(a == b);
        }
    };

    struct CommMessageInfo {
        size_t message_size;            ///< Size of the MPI message
        i32 rank_sender;                ///< Rank of the sender
        i32 rank_receiver;              ///< Rank of the receiver
        std::optional<i32> message_tag; ///< Tag of the MPI message

        std::optional<CommMessageBufOffset>
            message_bytebuf_offset_send; ///< Offset of the MPI message in the send buffer
        std::optional<CommMessageBufOffset>
            message_bytebuf_offset_recv; ///< Offset of the MPI message in the recv buffer
    };

    /**
     * @brief Communication table of the local rank
     *
     * Only the messages sent or received by the local rank are stored. The global message list
     * (allgatherv of every rank's messages_send, in rank order) is never materialized; each local
     * message is identified by its index in it, stored in send_message_global_ids and
     * recv_message_global_ids (both sorted in increasing order).
     */
    struct CommTable {
        std::vector<CommMessageInfo> messages_send;  ///< Messages to send
        std::vector<CommMessageInfo> messages_recv;  ///< Messages to recv
        std::vector<size_t> send_message_global_ids; ///< global ids of messages_send
        std::vector<size_t> recv_message_global_ids; ///< global ids of messages_recv

        std::vector<size_t> send_total_sizes; ///< Total size of the send buffer
        std::vector<size_t> recv_total_sizes; ///< Total size of the recv buffer
    };

    CommTable build_sparse_exchange_table(
        const std::vector<CommMessageInfo> &messages_send, size_t max_alloc_size);

    namespace details {

        /**
         * @brief Build the communication table of `world_rank` from the gathered message list
         *
         * Single pass over the packed global message list: only the messages sent or received by
         * `world_rank` are decoded. The tag of a message is its index within its sender's block,
         * which is `global_id - displs[sender]`.
         *
         * @param global_data packed messages `{pack32(sender, receiver), size}` of every rank,
         *        concatenated in rank order (the output of vector_allgatherv), so that the block
         *        of rank `r` starts at `displs[r]` and only holds messages sent by `r`
         * @param displs start of the block of each rank in global_data
         * @param world_rank the rank to build the table for
         * @param max_alloc_size max size of a single send/recv buffer
         */
        CommTable build_sparse_exchange_table_from_global(
            const std::vector<u64_2> &global_data,
            const std::vector<int> &displs,
            i32 world_rank,
            size_t max_alloc_size);

    } // namespace details

    template<sham::USMKindTarget target>
    void sparse_exchange(
        const std::shared_ptr<sham::DeviceScheduler> &dev_sched,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, target>>> &bytebuffer_send,
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, target>>> &bytebuffer_recv,
        const CommTable &comm_table);

} // namespace shamalgs::collective
