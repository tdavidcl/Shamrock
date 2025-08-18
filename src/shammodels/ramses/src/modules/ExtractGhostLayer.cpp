// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ExtractGhostLayer.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the ExtractGhostLayer solver graph node.
 */

#include "shambase/assert.hpp"
#include "shambase/exception.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shammath/AABB.hpp"
#include "shammath/paving_function.hpp"
#include "shammodels/ramses/modules/ExtractGhostLayer.hpp"
#include "shammodels/ramses/modules/FindGhostLayerCandidates.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/NodeInstance.hpp"
#include <stdexcept>

void shammodels::basegodunov::modules::ExtractGhostLayer::_impl_evaluate_internal() {
    StackEntry stack_loc{};

    auto edges = get_edges();

    // inputs
    auto &patch_data_layers = edges.patch_data_layers;
    auto &idx_in_ghost      = edges.idx_in_ghost;

    // outputs
    auto &ghost_layer = edges.ghost_layer;

    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

    // extract the ghost layers
    auto idx_in_ghost_it = idx_in_ghost.buffers.begin();

    // iterate on both DDShared containers
    for (; idx_in_ghost_it != idx_in_ghost.buffers.end(); ++idx_in_ghost_it) {

        auto [sender, receiver] = idx_in_ghost_it->first;

        const sham::DeviceBuffer<u32> &sender_idx_in_ghost = idx_in_ghost_it->second;

        shamrock::patch::PatchDataLayer ghost_zone(ghost_layer_layout);

        // extract the actual data
        patch_data_layers.get(sender).append_subset_to(
            sender_idx_in_ghost, u32(sender_idx_in_ghost.get_size()), ghost_zone);

        ghost_layer.patchdatas.add_obj(sender, receiver, std::move(ghost_zone));
    }
}

std::string shammodels::basegodunov::modules::ExtractGhostLayer::_impl_get_tex() { return "TODO"; }
