// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shambase/DistributedDataShared.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shamalgs/primitives/mock_value.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/solvergraph/ExchangeGhostLayer.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamrock/solvergraph/RankGetter.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

using namespace shamrock::solvergraph;
using namespace shamrock::patch;

NEW_TEST(Unittest, "shamrock/solvergraph/ExchangeGhostLayer", -1) {

    std::string test_name = "shamrock/solvergraph/ExchangeGhostLayer";

    // Create a simple layout with a few fields
    auto layout = std::make_shared<PatchDataLayerLayout>();
    layout->add_field<f32>("density", 1);
    layout->add_field<f32_3>("velocity", 1);
    layout->add_field<u32>("level", 1);

    // Seed for reproducible random data
    u64 seed = 0x1111;
    std::mt19937 eng(seed);

    u32 npatch = 100;

    // Generate a random original rank owner
    std::vector<u32> rank_owner;
    for (u64 i = 0; i < npatch; i++) {
        rank_owner.push_back(
            shamalgs::primitives::mock_value<u32>(eng, 0_u32, u32(shamcomm::world_size() - 1)));
    }

    // create reference data
    shambase::DistributedDataShared<PatchDataLayer> reference_data;
    for (u64 i = 0; i < npatch; i++) {
        // generate a random message
        u32 obj_count   = shamalgs::primitives::mock_value<u32>(eng, 0_u32, 100_u32);
        auto patch_data = PatchDataLayer::mock_patchdata(eng(), obj_count, layout);

        u32 sender   = shamalgs::primitives::mock_value<u32>(eng, 0_u32, npatch - 1);
        u32 receiver = shamalgs::primitives::mock_value<u32>(eng, 0_u32, npatch - 1);

        shamlog_debug_ln(
            test_name,
            "reference_data: ",
            sender,
            " receiver: ",
            receiver,
            " rank sender: ",
            rank_owner[sender],
            " rank receiver: ",
            rank_owner[receiver],
            " pdat: ",
            patch_data.get_obj_cnt());

        reference_data.add_obj(sender, receiver, std::move(patch_data));
    }

    // select the local data
    std::shared_ptr<shamrock::solvergraph::PatchDataLayerDDShared> exchange_gz_edge
        = std::make_shared<shamrock::solvergraph::PatchDataLayerDDShared>("", "");

    reference_data.for_each([&](u64 sender, u64 receiver, PatchDataLayer &pdat) {
        if (rank_owner[sender] == shamcomm::world_rank()) {
            exchange_gz_edge->patchdatas.add_obj(sender, receiver, pdat.duplicate());

            shamlog_debug_ln(
                test_name,
                "exchange_gz_edge: ",
                sender,
                " receiver: ",
                receiver,
                " rank sender: ",
                rank_owner[sender],
                " rank receiver: ",
                rank_owner[receiver],
                " pdat: ",
                pdat.get_obj_cnt());
        }
    });

    // create the rank owner edge
    auto rank_owner_edge = std::make_shared<shamrock::solvergraph::RankGetter>(
        [&](u64 patch_id) {
            return rank_owner[patch_id];
        },
        "rank_owner",
        "RO");

    std::shared_ptr<ExchangeGhostLayer> exchange_gz_node
        = std::make_shared<ExchangeGhostLayer>(layout);
    exchange_gz_node->set_edges(rank_owner_edge, exchange_gz_edge);

    exchange_gz_node->evaluate();

    shambase::DistributedDataShared<PatchDataLayer> interf_pdat
        = std::move(exchange_gz_edge->patchdatas);

    // generate the check dataset
    shambase::DistributedDataShared<PatchDataLayer> check_dataset{};
    reference_data.for_each([&](u64 sender, u64 receiver, PatchDataLayer &pdat) {
        if (rank_owner[receiver] == shamcomm::world_rank()) {
            check_dataset.add_obj(sender, receiver, pdat.duplicate());

            shamlog_debug_ln(
                test_name,
                "check_dataset: ",
                sender,
                " receiver: ",
                receiver,
                " rank sender: ",
                rank_owner[sender],
                " rank receiver: ",
                rank_owner[receiver],
                " pdat: ",
                pdat.get_obj_cnt());
        }
    });

    // verify the data
    auto check_dataset_it = check_dataset.begin();
    auto interf_pdat_it   = interf_pdat.begin();
    for (; check_dataset_it != check_dataset.end(); ++check_dataset_it, ++interf_pdat_it) {
        auto [sender, receiver] = check_dataset_it->first;

        auto &pdat        = check_dataset_it->second;
        auto &interf_pdat = interf_pdat_it->second;

        shamlog_debug_ln(
            test_name,
            "sender: ",
            sender,
            " receiver: ",
            receiver,
            "rank sender: ",
            rank_owner[sender],
            " rank receiver: ",
            rank_owner[receiver],
            " pdat: ",
            pdat.get_obj_cnt(),
            " interf_pdat: ",
            interf_pdat.get_obj_cnt());

        REQUIRE(pdat == interf_pdat);
    }
}

NEW_TEST(Unittest, "shamrock/solvergraph/ExchangeGhostLayer(reused edge)", -1) {

    // Several exchanges through the same edge, as in a solver where the edge is kept between
    // steps: the layers held by the edge (received during the previous round, with other sizes)
    // are reused to hold the data to send and the received data.

    auto layout = std::make_shared<PatchDataLayerLayout>();
    layout->add_field<f32>("density", 1);
    layout->add_field<f32_3>("velocity", 2);
    layout->add_field<u32>("level", 1);

    u32 npatch = 40;

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    auto edge = std::make_shared<PatchDataLayerDDShared>("gz", "GZ");

    std::vector<u32> rank_owner;
    auto rank_owner_edge = std::make_shared<RankGetter>(
        [&](u64 patch_id) {
            return rank_owner[patch_id];
        },
        "rank_owner",
        "RO");

    ExchangeGhostLayer node(layout);
    node.set_edges(rank_owner_edge, edge);

    for (u64 round = 0; round < 4; round++) {
        std::mt19937 eng(0x4444 + round);

        rank_owner.clear();
        for (u64 i = 0; i < npatch; i++) {
            rank_owner.push_back(
                shamalgs::primitives::mock_value<u32>(eng, 0_u32, u32(shamcomm::world_size() - 1)));
        }

        // the same reference data is generated on every rank
        shambase::DistributedDataShared<PatchDataLayer> reference_data;
        for (u64 i = 0; i < npatch; i++) {
            u32 obj_count = shamalgs::primitives::mock_value<u32>(eng, 0_u32, 100_u32);
            auto pdat     = PatchDataLayer::mock_patchdata(eng(), obj_count, layout);
            u32 sender    = shamalgs::primitives::mock_value<u32>(eng, 0_u32, npatch - 1);
            u32 receiver  = shamalgs::primitives::mock_value<u32>(eng, 0_u32, npatch - 1);
            reference_data.add_obj(sender, receiver, std::move(pdat));
        }

        // data to send by this rank
        shambase::DistributedDataShared<PatchDataLayer *> send_data;
        reference_data.for_each([&](u64 sender, u64 receiver, PatchDataLayer &pdat) {
            if (rank_owner[sender] == shamcomm::world_rank()) {
                send_data.add_obj(sender, receiver, &pdat);
            }
        });

        // fill the edge reusing the layers it holds, like ExtractGhostLayer
        edge->patchdatas.reshape_like(
            send_data,
            [&]() {
                return PatchDataLayer(layout);
            },
            [&](u64, u64, PatchDataLayer &out, PatchDataLayer *const &src) {
                u32 cnt = src->get_obj_cnt();
                std::vector<u32> idxs(cnt);
                std::iota(idxs.begin(), idxs.end(), 0);
                sham::DeviceBuffer<u32> idxs_buf(cnt, dev_sched);
                idxs_buf.copy_from_stdvec(idxs);
                src->assign_subset_to(idxs_buf, cnt, out);
            });

        node.evaluate();

        // expected data received by this rank
        std::vector<std::tuple<u64, u64, PatchDataLayer *>> expected;
        reference_data.for_each([&](u64 sender, u64 receiver, PatchDataLayer &pdat) {
            if (rank_owner[receiver] == shamcomm::world_rank()) {
                expected.emplace_back(sender, receiver, &pdat);
            }
        });

        REQUIRE_EQUAL(edge->patchdatas.get_element_count(), expected.size());

        size_t i = 0;
        edge->patchdatas.for_each([&](u64 sender, u64 receiver, PatchDataLayer &pdat) {
            auto [exp_sender, exp_receiver, exp_pdat] = expected.at(i++);
            REQUIRE_EQUAL(sender, exp_sender);
            REQUIRE_EQUAL(receiver, exp_receiver);
            REQUIRE(*exp_pdat == pdat);
        });
    }
}
