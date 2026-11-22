// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TransformShearSource.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the TransformShearSource solver graph node
 */

#include "shambase/exception.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/ramses/modules/TransformShearSource.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/NodeInstance.hpp"

template<class TgridVec>
void shammodels::basegodunov::modules::TransformShearSource<TgridVec>::_impl_evaluate_internal() {
    __shamrock_stack_entry();
    auto edges = get_edges();

    auto &infos       = edges.shear_src_infos.values;
    auto &ghost_layer = edges.ghost_layer.patchdatas;

    if (ghost_layer.get_element_count() != infos.get_element_count()) {
        shambase::throw_with_loc<std::runtime_error>(sham::format(
            "the shear source layers and their infos do not match: {} != {}",
            ghost_layer.get_element_count(),
            infos.get_element_count()));
    }

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    u32 iblock_min = ghost_layer_layout->get_field_idx<TgridVec>("cell_min");
    u32 iblock_max = ghost_layer_layout->get_field_idx<TgridVec>("cell_max");

    // both containers were filled in the same order (see FindShearSourceGhosts)
    auto layer_it = ghost_layer.begin();
    auto info_it  = infos.begin();
    for (; layer_it != ghost_layer.end(); ++layer_it, ++info_it) {

        if (layer_it->first != info_it->first) {
            shambase::throw_with_loc<std::runtime_error>(
                "the shear source layers and their infos are not in the same order");
        }

        shamrock::patch::PatchDataLayer &layer = layer_it->second;
        TgridVec offset                        = info_it->second.offset;

        u32 cnt = layer.get_obj_cnt();
        if (cnt == 0) {
            continue;
        }

        auto &block_min = layer.get_field<TgridVec>(iblock_min).get_buf();
        auto &block_max = layer.get_field<TgridVec>(iblock_max).get_buf();

        sham::kernel_call(
            q,
            sham::MultiRef{},
            sham::MultiRef{block_min, block_max},
            cnt,
            [offset](u32 i, TgridVec *__restrict bmin, TgridVec *__restrict bmax) {
                bmin[i] += offset;
                bmax[i] += offset;
            });
    }
}

template<class TgridVec>
std::string shammodels::basegodunov::modules::TransformShearSource<TgridVec>::_impl_get_tex()
    const {
    return "TODO";
}

template class shammodels::basegodunov::modules::TransformShearSource<i64_3>;
