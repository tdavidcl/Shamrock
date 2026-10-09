// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/narrowing.hpp"
#include "shambase/time.hpp"
#include "shamalgs/collective/sparse_exchange.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shamalgs/primitives/equals.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <fmt/std.h>
#include <algorithm>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace {

    struct TestElement {
        i32 sender, receiver;
        u32 size;
    };

} // namespace

void reorder_msg(std::vector<TestElement> &test_elements) {
    std::sort(test_elements.begin(), test_elements.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.sender
               < rhs.sender; //|| (lhs.sender == rhs.sender && lhs.receiver < rhs.receiver);
    });
}

#if false
void validate_comm_table(
    const std::vector<TestElement> &test_elements,
    const shamalgs::collective::CommTable &comm_table,
    size_t max_alloc_size) {

    std::vector<shamalgs::collective::CommMessageInfo> messages_send;

    std::vector<size_t> total_send_sizes = {0};
    std::vector<size_t> total_recv_sizes = {0};
    shamalgs::collective::sequentialize([&]() {
        u32 send_buf_id    = 0;
        u32 recv_buf_id    = 0;
        size_t send_offset = 0;
        size_t recv_offset = 0;
        for (u32 i = 0; i < test_elements.size(); i++) {
            if (test_elements[i].sender == shamcomm::world_rank()) {
                messages_send.push_back(
                    shamalgs::collective::CommMessageInfo{
                        test_elements[i].size,
                        test_elements[i].sender,
                        test_elements[i].receiver,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                    });

                logger::info_ln(
                    "sparse exchange test",
                    "rank :",
                    shamcomm::world_rank(),
                    "send message : (",
                    test_elements[i].sender,
                    "->",
                    test_elements[i].receiver,
                    ")");

                if (send_offset + test_elements[i].size > max_alloc_size) {
                    send_buf_id++;
                    send_offset = 0;
                    total_send_sizes.push_back(0);
                }

                total_send_sizes.at(send_buf_id) += test_elements[i].size;
            }
            if (test_elements[i].receiver == shamcomm::world_rank()) {
                if (recv_offset + test_elements[i].size > max_alloc_size) {
                    recv_buf_id++;
                    recv_offset = 0;
                    total_recv_sizes.push_back(0);
                }

                total_recv_sizes.at(recv_buf_id) += test_elements[i].size;
            }
        }
    });

    REQUIRE_EQUAL(comm_table.send_total_sizes, total_send_sizes);
    REQUIRE_EQUAL(comm_table.recv_total_sizes, total_recv_sizes);

    shamalgs::collective::sequentialize([&]() {
        size_t send_msg_idx = 0;
        size_t recv_msg_idx = 0;
        for (u32 i = 0; i < test_elements.size(); i++) {
            if (test_elements[i].sender == shamcomm::world_rank()) {
                REQUIRE_EQUAL(
                    comm_table.messages_send[send_msg_idx].message_size, test_elements[i].size);
                REQUIRE_EQUAL(
                    comm_table.messages_send[send_msg_idx].rank_sender, test_elements[i].sender);
                REQUIRE_EQUAL(
                    comm_table.messages_send[send_msg_idx].rank_receiver,
                    test_elements[i].receiver);

                send_msg_idx++;
            }
            if (test_elements[i].receiver == shamcomm::world_rank()) {
                REQUIRE_EQUAL(
                    comm_table.messages_recv[recv_msg_idx].message_size, test_elements[i].size);
                REQUIRE_EQUAL(
                    comm_table.messages_recv[recv_msg_idx].rank_sender, test_elements[i].sender);
                REQUIRE_EQUAL(
                    comm_table.messages_recv[recv_msg_idx].rank_receiver,
                    test_elements[i].receiver);

                auto &ref_buf = all_bufs[i];
                sham::DeviceBuffer<u8> recov(test_elements[i].size, dev_sched);
                auto off_info = shambase::get_check_ref(
                    comm_table.messages_recv[recv_msg_idx].message_bytebuf_offset_recv);
                size_t begin = off_info.data_offset;
                size_t end   = begin + test_elements[i].size;
                shambase::get_check_ref(recv_bufs.at(off_info.buf_id))
                    .copy_range(begin, end, recov);

                logger::info_ln(
                    "sparse exchange test",
                    "rank :",
                    shamcomm::world_rank(),
                    "recv message : (",
                    test_elements[i].sender,
                    "->",
                    test_elements[i].receiver,
                    ") data :",
                    recov.copy_to_stdvec());

                REQUIRE_EQUAL(recov.copy_to_stdvec(), ref_buf.copy_to_stdvec());

                recv_msg_idx++;
            }
        }
    });
}

template<>
struct fmt::formatter<shamalgs::collective::CommMessageBufOffset> {

    template<typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(shamalgs::collective::CommMessageBufOffset c, FormatContext &ctx) const {
        return fmt::format_to(
            ctx.out(), "Offset(buf_id : {}, data_offset : {})", c.buf_id, c.data_offset);
    }
};

template<>
struct fmt::formatter<shamalgs::collective::CommMessageInfo> {

    template<typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(shamalgs::collective::CommMessageInfo c, FormatContext &ctx) const {
        return fmt::format_to(
            ctx.out(),
            "Info(size : {}, sender : {}, receiver : {}, offset_send : {}, offset_recv : {})",
            c.message_size,
            c.rank_sender,
            c.rank_receiver,
            c.message_bytebuf_offset_send,
            c.message_bytebuf_offset_recv);
    }
};

void print_comm_table(const shamalgs::collective::CommTable &comm_table) {
    std::stringstream ss;
    ss << sham::format(
        "messages_send : [\n    {}\n]\n", fmt::join(comm_table.messages_send, "\n    "));
    ss << sham::format(
        "messages_recv : [\n    {}\n]\n", fmt::join(comm_table.messages_recv, "\n    "));
    ss << sham::format("send_message_global_ids : {}\n", comm_table.send_message_global_ids);
    ss << sham::format("recv_message_global_ids : {}\n", comm_table.recv_message_global_ids);
    ss << sham::format("send_total_sizes : {}\n", comm_table.send_total_sizes);
    ss << sham::format("recv_total_sizes : {}\n", comm_table.recv_total_sizes);
    logger::info_ln(
        "sparse exchange test", "rank :", shamcomm::world_rank(), "comm table :", "\n" + ss.str());
}
#endif

void test_sparse_exchange(std::vector<TestElement> test_elements, size_t max_alloc_size) {
    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    reorder_msg(test_elements);

    std::vector<sham::DeviceBuffer<u8>> all_bufs;

    std::mt19937 eng(0x123);
    for (const auto &test_element : test_elements) {
        all_bufs.push_back(
            shamalgs::random::mock_buffer_usm<u8>(dev_sched, eng(), test_element.size));
    }

    std::vector<shamalgs::collective::CommMessageInfo> messages_send;

    for (u32 i = 0; i < test_elements.size(); i++) {
        if (test_elements[i].sender == shamcomm::world_rank()) {
            messages_send.push_back(
                shamalgs::collective::CommMessageInfo{
                    .message_size                = test_elements[i].size,
                    .rank_sender                 = test_elements[i].sender,
                    .rank_receiver               = test_elements[i].receiver,
                    .message_tag                 = std::nullopt,
                    .message_bytebuf_offset_send = std::nullopt,
                    .message_bytebuf_offset_recv = std::nullopt,
                });
        }
    }

    shamalgs::collective::CommTable comm_table
        = shamalgs::collective::build_sparse_exchange_table(messages_send, max_alloc_size);

    // print_comm_table(comm_table);

    // check the local tables against the global message list
    {
        auto expected_tag = [&](size_t global_msg_id) {
            i32 tag = 0;
            for (size_t j = 0; j < global_msg_id; j++) {
                tag += (test_elements[j].sender == test_elements[global_msg_id].sender) ? 1 : 0;
            }
            return tag;
        };

        auto check_messages = [&](const std::vector<shamalgs::collective::CommMessageInfo> &msgs,
                                  const std::vector<size_t> &global_ids,
                                  bool is_send) {
            REQUIRE_EQUAL(msgs.size(), global_ids.size());
            size_t expected_count = 0;
            for (const auto &elem : test_elements) {
                i32 rank = (is_send) ? elem.sender : elem.receiver;
                expected_count += (rank == shamcomm::world_rank()) ? 1 : 0;
            }
            REQUIRE_EQUAL(msgs.size(), expected_count);
            for (size_t i = 0; i < msgs.size(); i++) {
                size_t gid = global_ids[i];
                REQUIRE(gid < test_elements.size());
                if (i > 0) {
                    REQUIRE(global_ids[i - 1] < gid);
                }
                REQUIRE_EQUAL(msgs[i].message_size, test_elements[gid].size);
                REQUIRE_EQUAL(msgs[i].rank_sender, test_elements[gid].sender);
                REQUIRE_EQUAL(msgs[i].rank_receiver, test_elements[gid].receiver);
                REQUIRE_EQUAL(shambase::get_check_ref(msgs[i].message_tag), expected_tag(gid));
            }
        };

        check_messages(comm_table.messages_send, comm_table.send_message_global_ids, true);
        check_messages(comm_table.messages_recv, comm_table.recv_message_global_ids, false);
    }

    // allocate send bufs
    std::vector<std::unique_ptr<sham::DeviceBuffer<u8>>> send_bufs{};

    for (size_t i = 0; i < comm_table.send_total_sizes.size(); i++) {
        send_bufs.push_back(
            std::make_unique<sham::DeviceBuffer<u8>>(comm_table.send_total_sizes[i], dev_sched));
    }

    // push data to the comm buf
    for (size_t i = 0; i < comm_table.messages_send.size(); i++) {
        auto msg_info        = comm_table.messages_send[i];
        size_t global_msg_id = comm_table.send_message_global_ids[i];

        auto off_info = shambase::get_check_ref(msg_info.message_bytebuf_offset_send);

        auto &source = all_bufs.at(global_msg_id);
        auto &dest   = shambase::get_check_ref(send_bufs.at(off_info.buf_id));

        source.copy_range_offset(0, source.get_size(), dest, off_info.data_offset);
    }

    // allocate recv bufs
    std::vector<std::unique_ptr<sham::DeviceBuffer<u8>>> recv_bufs{};

    for (size_t i = 0; i < comm_table.recv_total_sizes.size(); i++) {
        recv_bufs.push_back(
            std::make_unique<sham::DeviceBuffer<u8>>(comm_table.recv_total_sizes[i], dev_sched));
    }

    // do the comm
    if (dev_sched->ctx->device->mpi_prop.is_mpi_direct_capable) {
        shamalgs::collective::sparse_exchange(dev_sched, send_bufs, recv_bufs, comm_table);
    } else {
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, sham::host>>> send_bufs_host{};
        std::vector<std::unique_ptr<sham::DeviceBuffer<u8, sham::host>>> recv_bufs_host{};

        for (size_t i = 0; i < comm_table.send_total_sizes.size(); i++) {
            send_bufs_host.push_back(
                std::make_unique<sham::DeviceBuffer<u8, sham::host>>(
                    send_bufs[i]->copy_to<sham::host>()));
        }
        for (size_t i = 0; i < comm_table.recv_total_sizes.size(); i++) {
            recv_bufs_host.push_back(
                std::make_unique<sham::DeviceBuffer<u8, sham::host>>(
                    comm_table.recv_total_sizes[i], dev_sched));
        }

        shamalgs::collective::sparse_exchange(
            dev_sched, send_bufs_host, recv_bufs_host, comm_table);
        for (size_t i = 0; i < comm_table.recv_total_sizes.size(); i++) {
            recv_bufs[i]->copy_from(*recv_bufs_host[i]);
        }
    }

    {
        std::stringstream ss;
        ss << "send bufs :\n";
        for (size_t i = 0; i < send_bufs.size(); i++) {
            ss << "buf " << i << " : " << sham::format("{}", send_bufs[i]->copy_to_stdvec())
               << "\n";
        }
        ss << "recv bufs :\n";
        for (size_t i = 0; i < recv_bufs.size(); i++) {
            ss << "buf " << i << " : " << sham::format("{}", recv_bufs[i]->copy_to_stdvec())
               << "\n";
        }
        logger::info_ln("sparse exchange test", "rank :", shamcomm::world_rank(), ss.str());
    }

    // time to check
    std::vector<sham::DeviceBuffer<u8>> recv_messages;

    for (size_t i = 0; i < comm_table.messages_recv.size(); i++) {
        auto msg_info        = comm_table.messages_recv[i];
        size_t global_msg_id = comm_table.recv_message_global_ids[i];

        auto off_info
            = shambase::get_check_ref(comm_table.messages_recv[i].message_bytebuf_offset_recv);

        sham::DeviceBuffer<u8> recov(test_elements[global_msg_id].size, dev_sched);

        size_t begin = off_info.data_offset;
        size_t end   = begin + test_elements[global_msg_id].size;
        shambase::get_check_ref(recv_bufs.at(off_info.buf_id)).copy_range(begin, end, recov);
        recv_messages.push_back(std::move(recov));
    }

    // validate
    u32 recv_idx = 0;
    for (size_t i = 0; i < test_elements.size(); i++) {
        if (test_elements[i].receiver == shamcomm::world_rank()) {
            REQUIRE_EQUAL(recv_messages[recv_idx].copy_to_stdvec(), all_bufs[i].copy_to_stdvec());
            logger::info_ln(
                "sparse exchange test",
                "rank :",
                shamcomm::world_rank(),
                "recv message : (",
                test_elements[i].sender,
                "->",
                test_elements[i].receiver,
                ") data :",
                recv_messages[recv_idx].copy_to_stdvec(),
                "data ref :",
                all_bufs[i].copy_to_stdvec(),
                "valid :",
                recv_messages[recv_idx].copy_to_stdvec() == all_bufs[i].copy_to_stdvec());
            recv_idx++;
        }
    }
}

NEW_TEST(Unittest, "shamalgs/collective/test_sparse_exchange", -1) {

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("sparse exchange test", "empty comm");
    }

    test_sparse_exchange({}, i32_max);

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("sparse exchange test", "send to self");
    }

    {
        // everyone send to itself
        std::mt19937 eng(0x123);
        std::vector<TestElement> test_elements;
        for (i32 i = 0; i < shamcomm::world_size(); i++) {
            test_elements.push_back(
                TestElement{
                    .sender   = i,
                    .receiver = i,
                    .size     = shamalgs::primitives::mock_value<u32>(eng, 1, 10)});
        }
        test_sparse_exchange(test_elements, i32_max);
    }

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("sparse exchange test", "send to next");
    }

    {
        // everyone send to next one
        std::mt19937 eng(0x123);
        std::vector<TestElement> test_elements;
        for (i32 i = 0; i < shamcomm::world_size(); i++) {
            test_elements.push_back(
                TestElement{
                    .sender   = i,
                    .receiver = (i + 1) % shamcomm::world_size(),
                    .size     = shamalgs::primitives::mock_value<u32>(eng, 1, 10)});
        }
        test_sparse_exchange(test_elements, i32_max);
    }

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("sparse exchange test", "random test");
    }

    {
        // random test
        std::mt19937 eng(0x123);
        std::vector<TestElement> test_elements;
        for (u32 i = 0; i < 3 * shamcomm::world_size(); i++) {
            test_elements.push_back(
                TestElement{
                    .sender
                    = shamalgs::primitives::mock_value<i32>(eng, 0, shamcomm::world_size() - 1),
                    .receiver
                    = shamalgs::primitives::mock_value<i32>(eng, 0, shamcomm::world_size() - 1),
                    .size = shamalgs::primitives::mock_value<u32>(eng, 1, 10)});
        }
        test_sparse_exchange(test_elements, i32_max);
    }

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("sparse exchange test", "random test (force multiple bufs)");
    }

    {
        // random test
        std::mt19937 eng(0x123);
        std::vector<TestElement> test_elements;
        for (u32 i = 0; i < 3 * shamcomm::world_size(); i++) {
            test_elements.push_back(
                TestElement{
                    .sender
                    = shamalgs::primitives::mock_value<i32>(eng, 0, shamcomm::world_size() - 1),
                    .receiver
                    = shamalgs::primitives::mock_value<i32>(eng, 0, shamcomm::world_size() - 1),
                    .size = shamalgs::primitives::mock_value<u32>(eng, 1, 10)});
        }
        test_sparse_exchange(test_elements, 20);
    }
}

namespace {

    /// Reference implementation of the comm table (previous multi-pass version), kept to validate
    /// and benchmark details::build_sparse_exchange_table_from_global
    struct ReferenceCommTable {
        std::vector<shamalgs::collective::CommMessageInfo> message_all;
        shamalgs::collective::CommTable table;
    };

    ReferenceCommTable reference_build_sparse_exchange_table(
        const std::vector<u64_2> &global_data,
        i32 world_rank,
        i32 world_size,
        size_t max_alloc_size) {

        using namespace shamalgs::collective;

        // decode
        std::vector<CommMessageInfo> message_all(global_data.size());
        for (u64 i = 0; i < global_data.size(); i++) {
            u32_2 comm_ranks = sham::unpack32(global_data[i].x());
            message_all[i]   = CommMessageInfo{
                .message_size                = global_data[i].y(),
                .rank_sender                 = static_cast<i32>(comm_ranks.x()),
                .rank_receiver               = static_cast<i32>(comm_ranks.y()),
                .message_tag                 = std::nullopt,
                .message_bytebuf_offset_send = std::nullopt,
                .message_bytebuf_offset_recv = std::nullopt};
        }

        // tags
        std::vector<i32> tag_map(static_cast<size_t>(world_size), 0);
        for (auto &message_info : message_all) {
            message_info.message_tag = tag_map[static_cast<size_t>(message_info.rank_sender)]++;
        }

        // offsets
        std::vector<size_t> send_buf_sizes{};
        std::vector<size_t> recv_buf_sizes{};
        u32 send_idx = 0;
        u32 recv_idx = 0;
        {
            size_t tmp_recv_offset = 0;
            size_t tmp_send_offset = 0;
            size_t send_buf_id     = 0;
            size_t recv_buf_id     = 0;
            for (auto &message_info : message_all) {
                if (message_info.rank_sender == world_rank) {
                    if (send_buf_sizes.size() == 0) {
                        send_buf_sizes.push_back(0);
                    }
                    if (tmp_send_offset + message_info.message_size >= max_alloc_size) {
                        send_buf_id++;
                        tmp_send_offset = 0;
                        send_buf_sizes.push_back(0);
                    }
                    message_info.message_bytebuf_offset_send
                        = {.buf_id = send_buf_id, .data_offset = tmp_send_offset};
                    tmp_send_offset += message_info.message_size;
                    send_buf_sizes.at(send_buf_id) += message_info.message_size;
                    send_idx++;
                }
                if (message_info.rank_receiver == world_rank) {
                    if (recv_buf_sizes.size() == 0) {
                        recv_buf_sizes.push_back(0);
                    }
                    if (tmp_recv_offset + message_info.message_size >= max_alloc_size) {
                        recv_buf_id++;
                        tmp_recv_offset = 0;
                        recv_buf_sizes.push_back(0);
                    }
                    message_info.message_bytebuf_offset_recv
                        = {.buf_id = recv_buf_id, .data_offset = tmp_recv_offset};
                    tmp_recv_offset += message_info.message_size;
                    recv_buf_sizes.at(recv_buf_id) += message_info.message_size;
                    recv_idx++;
                }
            }
        }

        // split
        ReferenceCommTable ret{};
        auto &table = ret.table;
        table.messages_send.resize(send_idx);
        table.messages_recv.resize(recv_idx);
        table.send_message_global_ids.resize(send_idx);
        table.recv_message_global_ids.resize(recv_idx);
        send_idx = 0;
        recv_idx = 0;
        for (size_t i = 0; i < message_all.size(); i++) {
            auto message_info = message_all[i];
            if (message_info.rank_sender == world_rank) {
                table.messages_send[send_idx]           = message_info;
                table.send_message_global_ids[send_idx] = i;
                send_idx++;
            }
            if (message_info.rank_receiver == world_rank) {
                table.messages_recv[recv_idx]           = message_info;
                table.recv_message_global_ids[recv_idx] = i;
                recv_idx++;
            }
        }
        table.send_total_sizes = send_buf_sizes;
        table.recv_total_sizes = recv_buf_sizes;
        ret.message_all        = message_all; // the old CommTable stored a copy
        return ret;
    }

    /// Previous sparse_exchange loop without the MPI calls (walk every message to find the local
    /// ones), to benchmark the host cost of the old exchange loop
    size_t reference_walk_all_messages(
        const std::vector<shamalgs::collective::CommMessageInfo> &message_all, i32 world_rank) {
        size_t posted = 0;
        for (size_t i = 0; i < message_all.size(); i++) {
            auto message_info = message_all[i];
            if (message_info.rank_sender == world_rank) {
                posted
                    += shambase::get_check_ref(message_info.message_bytebuf_offset_send).buf_id + 1;
            }
            if (message_info.rank_receiver == world_rank) {
                posted
                    += shambase::get_check_ref(message_info.message_bytebuf_offset_recv).buf_id + 1;
            }
        }
        return posted;
    }

    /// New sparse_exchange loop without the MPI calls (merge of the local sends and recvs by
    /// global id)
    size_t walk_local_messages(const shamalgs::collective::CommTable &table) {
        size_t posted = 0;
        size_t i_send = 0;
        size_t i_recv = 0;
        while (i_send < table.messages_send.size() || i_recv < table.messages_recv.size()) {
            size_t gid_send = (i_send < table.messages_send.size())
                                  ? table.send_message_global_ids[i_send]
                                  : SIZE_MAX;
            size_t gid_recv = (i_recv < table.messages_recv.size())
                                  ? table.recv_message_global_ids[i_recv]
                                  : SIZE_MAX;
            size_t gid      = std::min(gid_send, gid_recv);
            if (gid_send == gid) {
                posted += shambase::get_check_ref(
                              table.messages_send[i_send].message_bytebuf_offset_send)
                              .buf_id
                          + 1;
                i_send++;
            }
            if (gid_recv == gid) {
                posted += shambase::get_check_ref(
                              table.messages_recv[i_recv].message_bytebuf_offset_recv)
                              .buf_id
                          + 1;
                i_recv++;
            }
        }
        return posted;
    }

    /// Synthetic gathered message list: each rank sends to `msg_per_rank` random ranks on average
    /// (at least one), concatenated in rank order like vector_allgatherv
    struct SyntheticGlobalData {
        std::vector<u64_2> global_data;
        std::vector<int> displs;
    };

    SyntheticGlobalData make_synthetic_global_data(
        std::mt19937 &eng, u32 world_size, u32 msg_per_rank, u32 max_msg_size) {
        SyntheticGlobalData ret{};
        ret.displs.resize(world_size);
        std::uniform_int_distribution<u32> dist_rank(0, world_size - 1);
        std::uniform_int_distribution<u32> dist_count(1, 2 * msg_per_rank - 1);
        std::uniform_int_distribution<u64> dist_size(1, max_msg_size);
        for (u32 sender = 0; sender < world_size; sender++) {
            ret.displs[sender] = shambase::narrow_or_throw<int>(ret.global_data.size());
            u32 count          = dist_count(eng);
            for (u32 j = 0; j < count; j++) {
                ret.global_data.push_back(
                    u64_2{sham::pack32(sender, dist_rank(eng)), dist_size(eng)});
            }
        }
        return ret;
    }

    void require_same_message_list(
        const std::vector<shamalgs::collective::CommMessageInfo> &a,
        const std::vector<shamalgs::collective::CommMessageInfo> &b) {
        REQUIRE_EQUAL(a.size(), b.size());
        for (size_t i = 0; i < std::min(a.size(), b.size()); i++) {
            REQUIRE_EQUAL(a[i].message_size, b[i].message_size);
            REQUIRE_EQUAL(a[i].rank_sender, b[i].rank_sender);
            REQUIRE_EQUAL(a[i].rank_receiver, b[i].rank_receiver);
            REQUIRE(a[i].message_tag == b[i].message_tag);
            REQUIRE(a[i].message_bytebuf_offset_send == b[i].message_bytebuf_offset_send);
            REQUIRE(a[i].message_bytebuf_offset_recv == b[i].message_bytebuf_offset_recv);
        }
    }

} // namespace

NEW_TEST(Unittest, "shamalgs/collective/build_sparse_exchange_table_from_global", 1) {

    std::mt19937 eng(0x111);

    for (u32 world_size : {1_u32, 2_u32, 7_u32, 64_u32}) {
        for (u32 msg_per_rank : {1_u32, 3_u32, 10_u32}) {
            SyntheticGlobalData data = make_synthetic_global_data(eng, world_size, msg_per_rank, 9);

            for (size_t max_alloc_size : {size_t{10}, size_t{25}, size_t{i32_max}}) {
                for (u32 rank = 0; rank < world_size; rank++) {
                    auto ref = reference_build_sparse_exchange_table(
                        data.global_data,
                        static_cast<i32>(rank),
                        static_cast<i32>(world_size),
                        max_alloc_size);
                    auto table
                        = shamalgs::collective::details::build_sparse_exchange_table_from_global(
                            data.global_data, data.displs, static_cast<i32>(rank), max_alloc_size);

                    require_same_message_list(table.messages_send, ref.table.messages_send);
                    require_same_message_list(table.messages_recv, ref.table.messages_recv);
                    REQUIRE_EQUAL(table.send_message_global_ids, ref.table.send_message_global_ids);
                    REQUIRE_EQUAL(table.recv_message_global_ids, ref.table.recv_message_global_ids);
                    REQUIRE_EQUAL(table.send_total_sizes, ref.table.send_total_sizes);
                    REQUIRE_EQUAL(table.recv_total_sizes, ref.table.recv_total_sizes);
                    REQUIRE_EQUAL(
                        walk_local_messages(table),
                        reference_walk_all_messages(ref.message_all, static_cast<i32>(rank)));
                }
            }
        }
    }

    // empty exchange
    auto table = shamalgs::collective::details::build_sparse_exchange_table_from_global(
        {}, {}, 0, i32_max);
    REQUIRE(table.messages_send.empty());
    REQUIRE(table.messages_recv.empty());
    REQUIRE(table.send_total_sizes.empty());
    REQUIRE(table.recv_total_sizes.empty());
}

NEW_TEST(Benchmark, "shamalgs/collective/build_sparse_exchange_table_benchmark", 1) {

    // M = world_size * msg_per_rank messages in the gathered list, up to ~5M as projected for
    // 800k patches over 100k ranks
    std::mt19937 eng(0x222);
    for (auto [world_size, msg_per_rank] :
         std::vector<std::pair<u32, u32>>{{24576, 32}, {65536, 32}, {100000, 20}, {100000, 50}}) {
        SyntheticGlobalData data
            = make_synthetic_global_data(eng, world_size, msg_per_rank, 1 << 20);

        // a rank in the middle, its messages are spread through the global list
        i32 rank = static_cast<i32>(world_size / 2);

        size_t check_old = 0;
        size_t check_new = 0;

        f64 t_old = shambase::timeitfor([&]() {
            auto ref = reference_build_sparse_exchange_table(
                data.global_data, rank, static_cast<i32>(world_size), i32_max);
            check_old = reference_walk_all_messages(ref.message_all, rank);
        });

        f64 t_new = shambase::timeitfor([&]() {
            auto table = shamalgs::collective::details::build_sparse_exchange_table_from_global(
                data.global_data, data.displs, rank, i32_max);
            check_new = walk_local_messages(table);
        });

        REQUIRE_EQUAL(check_old, check_new);

        logger::info_ln(
            "build_sparse_exchange_table_benchmark",
            "world_size",
            world_size,
            "messages",
            data.global_data.size(),
            "old (ms)",
            t_old * 1e3,
            "new (ms)",
            t_new * 1e3,
            "speedup",
            t_old / t_new);
    }
}
