// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ExtractGhostLayer.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the ExtractGhostLayer solver graph node.
 */

#include "shammodels/ramses/modules/ExtractGhostLayer.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"

void shammodels::basegodunov::modules::ExtractGhostLayer::_impl_evaluate_internal() {
    __shamrock_stack_entry();

    auto edges = get_edges();

    // inputs
    auto &patch_data_layers = edges.patch_data_layers;
    auto &idx_in_ghost      = edges.idx_in_ghost;

    // outputs
    auto &ghost_layer = edges.ghost_layer;

    // one ghost layer per entry of idx_in_ghost, the layers held by the edge (from the previous
    // evaluation) are reused instead of being reallocated
    ghost_layer.patchdatas.reshape_like(
        idx_in_ghost.buffers,
        [&]() {
            return shamrock::patch::PatchDataLayer(ghost_layer_layout);
        },
        [&](u64 sender,
            [[maybe_unused]] u64 receiver,
            shamrock::patch::PatchDataLayer &ghost_zone,
            const sham::DeviceBuffer<u32> &sender_idx_in_ghost) {
            if (ghost_zone.get_layout_ptr().get() != ghost_layer_layout.get()) {
                ghost_zone = shamrock::patch::PatchDataLayer(ghost_layer_layout);
            }

            // extract the actual data
            patch_data_layers.get(sender).assign_subset_to(
                sender_idx_in_ghost, u32(sender_idx_in_ghost.get_size()), ghost_zone);
        });
}

std::string shammodels::basegodunov::modules::ExtractGhostLayer::_impl_get_tex() const {
    return "TODO";
}
