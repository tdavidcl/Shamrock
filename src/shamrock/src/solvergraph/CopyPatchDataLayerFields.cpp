// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CopyPatchDataLayerFields.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the CopyPatchDataLayerFields class for copying fields between patch data layers.
 *
 */

#include "shamrock/solvergraph/CopyPatchDataLayerFields.hpp"

namespace shamrock::solvergraph {

    void CopyPatchDataLayerFields::_impl_evaluate_internal() {
        StackEntry stack_loc{};

        auto edges = get_edges();

        // Ensures that the layout are all matching sources and targets
        edges.original.get_const_refs().for_each(
            [&](u64 id_patch, const patch::PatchDataLayer &pdat) {
                if (pdat.get_layout_ptr().get() != layout_source.get()) {
                    throw shambase::make_except_with_loc<std::invalid_argument>("layout mismatch");
                }
            });

        if (edges.target.get_layout_ptr().get() != layout_target.get()) {
            throw shambase::make_except_with_loc<std::invalid_argument>("layout mismatch");
        }

        // Resize the target layers in place (reusing their allocations from the previous
        // evaluation), then copy the fields from the original to the target
        edges.target.ensure_sizes(edges.original.get_const_refs().map<u32>(
            [&](u64 id_patch, const patch::PatchDataLayer &pdat) {
                return pdat.get_obj_cnt();
            }));

        edges.original.get_const_refs().for_each(
            [&](u64 id_patch, const patch::PatchDataLayer &pdat) {
                patch::PatchDataLayer &pdat_target = edges.target.get(id_patch);

                pdat_target.for_each_field_any([&](auto &field) {
                    using T = typename std::remove_reference<decltype(field)>::type::Field_type;
                    const PatchDataField<T> &field_source = pdat.get_field<T>(field.get_name());
                    field.overwrite(field_source, field_source.get_obj_cnt());
                });

                pdat_target.check_field_obj_cnt_match();
            });
    }

} // namespace shamrock::solvergraph
