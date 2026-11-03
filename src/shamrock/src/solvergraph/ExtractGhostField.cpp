// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ExtractGhostField.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief  This module allows to extract ghosts for a generic field such as density, velocity, etc
 *
 */

#include "shambase/stacktrace.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/ExtractGhostField.hpp"

namespace shamrock::solvergraph {
    template<class T>
    void ExtractGhostField<T>::_impl_evaluate_internal() {
        StackEntry stack_loc{};

        auto edges = get_edges();

        auto &orig_data_fields = edges.original_fields;
        auto &idx_in_ghost     = edges.idx_in_ghosts;
        auto &ghost_fields     = edges.ghost_fields;

        // one ghost field per entry of idx_in_ghost, the fields held by the edge (from the
        // previous evaluation) are reused instead of being reallocated
        ghost_fields.patchdata_fields.reshape_like(
            idx_in_ghost.buffers,
            []() {
                // name and number of variables are set by assign_subset_to
                return PatchDataField<T>("", 1);
            },
            [&](u64 sender,
                [[maybe_unused]] u64 receiver,
                PatchDataField<T> &gz_field,
                const sham::DeviceBuffer<u32> &sender_idx_in_ghost) {
                // TODO: replace cast by narrowing when added
                orig_data_fields.get_field(sender).assign_subset_to(
                    sender_idx_in_ghost,
                    static_cast<u32>(sender_idx_in_ghost.get_size()),
                    gz_field);
            });
    }

    template class ExtractGhostField<f64>;
    template class ExtractGhostField<f64_3>;

} // namespace shamrock::solvergraph
