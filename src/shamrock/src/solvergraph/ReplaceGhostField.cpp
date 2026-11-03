// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ReplaceGhostField.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief   This module allows replacing ghost values for a generic field with updated values.
 *
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/ReplaceGhostField.hpp"
#include <map>

namespace shamrock::solvergraph {
    template<class T>
    void shamrock::solvergraph::ReplaceGhostField<T>::_impl_evaluate_internal() {

        StackEntry stack_loc{};
        auto edges = get_edges();

        auto &ghost_fields = edges.ghost_fields;
        auto &fields       = edges.fields;

        // number of ghost objects received by each patch
        std::map<u64, u32> gz_map;
        ghost_fields.patchdata_fields.for_each(
            [&](u64 s, u64 r, const PatchDataField<T> &pdat_field) {
                gz_map[r] += pdat_field.get_obj_cnt();
            });

        // the ghost zones are stored at the end of the fields, compute where they start
        // TODO: currently we guess the GZ size using the input, we should use in place ghost
        // zones really ...
        std::map<u64, u32> gz_offset;
        fields.get_refs().for_each([&](u64 id_patch, PatchDataField<T> &field) {
            u32 gz_cnt  = gz_map.at(id_patch);
            u32 obj_cnt = field.get_obj_cnt();
            if (gz_cnt > obj_cnt) {
                throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                    "more ghost objects ({}) than objects in the field ({}) for patch {}",
                    gz_cnt,
                    obj_cnt,
                    id_patch));
            }
            gz_offset[id_patch] = obj_cnt - gz_cnt;
        });

        // overwrite the old ghost zones in place with the new ones, in the same order as they
        // would be appended, this avoids shrinking then expanding the fields
        ghost_fields.patchdata_fields.for_each(
            [&](u64 s, u64 r, const PatchDataField<T> &pdat_field) {
                PatchDataField<T> &field = fields.get_field(r);
                u32 &offset              = gz_offset.at(r);

                if (pdat_field.get_nvar() != field.get_nvar()) {
                    throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                        "nvar mismatch between the ghost field ({}) and the field ({})",
                        pdat_field.get_nvar(),
                        field.get_nvar()));
                }

                pdat_field.get_buf().copy_range_offset(
                    0,
                    pdat_field.get_val_cnt(),
                    field.get_buf(),
                    size_t(offset) * size_t(field.get_nvar()));

                offset += pdat_field.get_obj_cnt();
            });
    }

    template class shamrock::solvergraph::ReplaceGhostField<f64>;
    template class shamrock::solvergraph::ReplaceGhostField<f64_3>;

} // namespace shamrock::solvergraph
