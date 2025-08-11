// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GenerateGhostLayer.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/assert.hpp"
#include "shambase/exception.hpp"
#include "shammath/AABB.hpp"
#include "shammath/paving_function.hpp"
#include "shammodels/ramses/modules/GenerateGhostLayer.hpp"
#include <stdexcept>

namespace {

    using namespace shammodels::basegodunov::modules;

    template<class TgridVec>
    shammath::paving_function_general_3d<TgridVec>
    get_paving(GhostLayerGenMode mode, shammath::AABB<TgridVec> sim_box) {

        TgridVec box_size   = sim_box.max - sim_box.min;
        TgridVec box_center = (sim_box.max + sim_box.min) / 2;

        SHAM_ASSERT(sim_box.is_volume_not_null());

        { // check that rebuildind the AABB from size and center gives the same AABB
            shammath::AABB<TgridVec> new_box
                = {box_center - box_size / 2, box_center + box_size / 2};
            if (new_box != sim_box) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "Rebuilding AABB from size and center gives a different AABB");
            }
        }

        return shammath::paving_function_general_3d<TgridVec>{
            box_size,
            box_center,
            mode.ghost_type_x == GhostType::Periodic,
            mode.ghost_type_y == GhostType::Periodic,
            mode.ghost_type_z == GhostType::Periodic};
    }

    template<class Func>
    void for_each_paving_tile(GhostLayerGenMode mode, Func &&func) {

        // if the ghost type is none, we do not need to repeat as there is no ghost layer
        i32 repetition_x = mode.ghost_type_x != GhostType::None;
        i32 repetition_y = mode.ghost_type_y != GhostType::None;
        i32 repetition_z = mode.ghost_type_z != GhostType::None;

        for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
            for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
                for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {
                    func(xoff, yoff, zoff);
                }
            }
        }
    }

} // namespace

template<class Tvec, class TgridVec>
void GenerateGhostLayer<Tvec, TgridVec>::_impl_evaluate_internal() {
    auto edges = get_edges();

    // inputs
    auto &sim_box           = edges.sim_box.value;
    auto &patch_data_layers = edges.patch_data_layers;
    auto &patch_tree        = edges.patch_tree.get_patch_tree();
    auto &patch_boxes       = edges.patch_boxes;

    using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

    // outputs
    auto &ghost_layers = edges.ghost_layers.patchdatas;

    auto paving = get_paving(mode, sim_box);

    using namespace shamrock::patch;

    struct FoundGhostLayer {
        u64 sender;
        u64 receiver;

        i32 xoff;
        i32 yoff;
        i32 zoff;
    };

    std::vector<FoundGhostLayer> found_ghost_layers;

    // for each repetitions
    for_each_paving_tile(mode, [&](i32 xoff, i32 yoff, i32 zoff) {
        // for all local patches
        patch_data_layers.patchdatas.for_each([&](u64 id, PatchDataLayer &pdat) {
            // get the current patch box
            auto &patch_box = patch_boxes.get(id);

            // f(patch)
            auto patch_box_mapped = paving.f_aabb(patch_box, xoff, yoff, zoff);

            patch_tree.host_for_each_leafs(
                [&](u64 tree_id, PtNode n) {
                    shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};

                    // f(patch) V box =! empty (a surface is not an empty set btw)
                    // <=> is ghost layer != empty
                    return tree_cell.get_intersect(patch_box_mapped).is_not_empty();
                },
                [&](u64 id_found, PtNode n) {
                    // skip self intersection (but not if we are through a boundary)
                    if ((id_found == id) && (xoff == 0) && (yoff == 0) && (zoff == 0)) {
                        return;
                    }

                    // we have an ghost layer between
                    // patch `id` and patch `id_found` for this offset
                    // so we store that
                    found_ghost_layers.push_back({id, id_found, xoff, yoff, zoff});
                });
        });
    });
}

template<class Tvec, class TgridVec>
std::string GenerateGhostLayer<Tvec, TgridVec>::_impl_get_tex() {
    return "TODO";
}
