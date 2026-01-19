// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/collective/sparse_exchange.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shamalgs/primitives/equals.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
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

    std::vector<size_t> total_send_sizes = {0};
    std::vector<size_t> total_recv_sizes = {0};
    {
        u32 send_buf_id    = 0;
        size_t send_offset = 0;
        u32 recv_buf_id    = 0;
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
                    ") data :",
                    all_bufs[i].copy_to_stdvec());

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
    }

    shamalgs::collective::CommTable comm_table
        = shamalgs::collective::build_sparse_exchange_table(messages_send, max_alloc_size);

    REQUIRE_EQUAL(comm_table.send_total_sizes, total_send_sizes);
    REQUIRE_EQUAL(comm_table.recv_total_sizes, total_recv_sizes);

    // allocate send and receive bufs
    std::vector<std::unique_ptr<sham::DeviceBuffer<u8>>> send_bufs{};

    for (size_t i = 0; i < comm_table.send_total_sizes.size(); i++) {
        send_bufs.push_back(
            std::make_unique<sham::DeviceBuffer<u8>>(comm_table.send_total_sizes[i], dev_sched));
    }

    std::vector<std::unique_ptr<sham::DeviceBuffer<u8>>> recv_bufs{};

    for (size_t i = 0; i < comm_table.recv_total_sizes.size(); i++) {
        recv_bufs.push_back(
            std::make_unique<sham::DeviceBuffer<u8>>(comm_table.recv_total_sizes[i], dev_sched));
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

    // time to check

    size_t send_msg_idx = 0;
    size_t recv_msg_idx = 0;
    for (u32 i = 0; i < test_elements.size(); i++) {
        if (test_elements[i].sender == shamcomm::world_rank()) {
            REQUIRE_EQUAL(
                comm_table.messages_send[send_msg_idx].message_size, test_elements[i].size);
            REQUIRE_EQUAL(
                comm_table.messages_send[send_msg_idx].rank_sender, test_elements[i].sender);
            REQUIRE_EQUAL(
                comm_table.messages_send[send_msg_idx].rank_receiver, test_elements[i].receiver);

            send_msg_idx++;
        }
        if (test_elements[i].receiver == shamcomm::world_rank()) {
            REQUIRE_EQUAL(
                comm_table.messages_recv[recv_msg_idx].message_size, test_elements[i].size);
            REQUIRE_EQUAL(
                comm_table.messages_recv[recv_msg_idx].rank_sender, test_elements[i].sender);
            REQUIRE_EQUAL(
                comm_table.messages_recv[recv_msg_idx].rank_receiver, test_elements[i].receiver);

            auto &ref_buf = all_bufs[i];
            sham::DeviceBuffer<u8> recov(test_elements[i].size, dev_sched);
            auto off_info = shambase::get_check_ref(
                comm_table.messages_recv[recv_msg_idx].message_bytebuf_offset_recv);
            size_t begin = off_info.data_offset;
            size_t end   = begin + test_elements[i].size;
            shambase::get_check_ref(recv_bufs.at(off_info.buf_id)).copy_range(begin, end, recov);

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
        REQUIRE_EQUAL(comm_table.message_all[i].message_size, test_elements[i].size);
        REQUIRE_EQUAL(comm_table.message_all[i].rank_sender, test_elements[i].sender);
        REQUIRE_EQUAL(comm_table.message_all[i].rank_receiver, test_elements[i].receiver);
    }
}

TestStart(Unittest, "shamalgs/collective/test_sparse_exchange", testsparsexchg_2, -1) {

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
                TestElement{i, i, shamalgs::primitives::mock_value<u32>(eng, 1, 10)});
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
                    i,
                    (i + 1) % shamcomm::world_size(),
                    shamalgs::primitives::mock_value<u32>(eng, 1, 10)});
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
                    shamalgs::primitives::mock_value<i32>(eng, 0, shamcomm::world_size() - 1),
                    shamalgs::primitives::mock_value<i32>(eng, 0, shamcomm::world_size() - 1),
                    shamalgs::primitives::mock_value<u32>(eng, 1, 10)});
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
                    shamalgs::primitives::mock_value<i32>(eng, 0, shamcomm::world_size() - 1),
                    shamalgs::primitives::mock_value<i32>(eng, 0, shamcomm::world_size() - 1),
                    shamalgs::primitives::mock_value<u32>(eng, 1, 10)});
        }
        test_sparse_exchange(test_elements, 20);
    }
}
