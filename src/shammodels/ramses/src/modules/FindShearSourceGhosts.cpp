// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FindShearSourceGhosts.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the FindShearSourceGhosts solver graph node
 */

#include "shammodels/ramses/modules/FindShearSourceGhosts.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/NodeInstance.hpp"
#include <algorithm>

namespace {

    /// Largest extent of a block over all the ranks (in grid units)
    template<class TgridVec>
    shambase::VecComponent<TgridVec> get_global_max_block_extent(
        const sham::DeviceScheduler_ptr &dev_sched,
        const shamrock::solvergraph::IPatchDataLayerRefs &patch_data_layers) {

        using Tgridscal = shambase::VecComponent<TgridVec>;

        sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

        Tgridscal local_max = 0;

        patch_data_layers.get_const_refs().for_each([&](u64 id, const auto &pdat_ref) {
            shamrock::patch::PatchDataLayer &pdat = pdat_ref.get();
            u32 cnt                               = pdat.get_obj_cnt();
            if (cnt == 0) {
                return;
            }

            auto &block_min = pdat.template get_field<TgridVec>(0).get_buf();
            auto &block_max = pdat.template get_field<TgridVec>(1).get_buf();

            sham::DeviceBuffer<Tgridscal> extent(cnt, dev_sched);

            sham::kernel_call(
                q,
                sham::MultiRef{block_min, block_max},
                sham::MultiRef{extent},
                cnt,
                [](u32 i,
                   const TgridVec *__restrict bmin,
                   const TgridVec *__restrict bmax,
                   Tgridscal *__restrict ext) {
                    TgridVec d = bmax[i] - bmin[i];
                    ext[i]     = sycl::max(d[0], sycl::max(d[1], d[2]));
                });

            local_max = std::max(local_max, shamalgs::primitives::max(dev_sched, extent, 0, cnt));
        });

        return shamalgs::collective::allreduce_max(local_max);
    }

    /// Indexes of the blocks that, once translated by `offset`, intersect `region` with a volume
    template<class TgridVec>
    sham::DeviceBuffer<u32> get_ids_in_region(
        const sham::DeviceScheduler_ptr &dev_sched,
        const sham::DeviceBuffer<TgridVec> &block_min,
        const sham::DeviceBuffer<TgridVec> &block_max,
        TgridVec offset,
        shammath::AABB<TgridVec> region,
        u32 obj_cnt) {

        if (obj_cnt == 0) {
            return sham::DeviceBuffer<u32>(0, dev_sched);
        }

        sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

        sham::DeviceBuffer<u32> mask(obj_cnt, dev_sched);

        sham::kernel_call(
            q,
            sham::MultiRef{block_min, block_max},
            sham::MultiRef{mask},
            obj_cnt,
            [offset, region](
                u32 i,
                const TgridVec *__restrict bmin,
                const TgridVec *__restrict bmax,
                u32 *__restrict in_region) {
                shammath::AABB<TgridVec> block{bmin[i] + offset, bmax[i] + offset};
                in_region[i] = block.get_intersect(region).is_volume_not_null();
            });

        return shamalgs::stream_compact(dev_sched, mask, obj_cnt);
    }

} // namespace

template<class TgridVec>
void shammodels::basegodunov::modules::FindShearSourceGhosts<TgridVec>::_impl_evaluate_internal() {
    __shamrock_stack_entry();
    auto edges = get_edges();

    using Tgridscal = shambase::VecComponent<TgridVec>;
    using AABB      = shammath::AABB<TgridVec>;
    using PtNode    = typename SerialPatchTree<TgridVec>::PtNode;

    // inputs
    const AABB &sim_box     = edges.sim_box.value;
    auto &patch_tree        = edges.patch_tree.get_patch_tree();
    auto &patch_boxes       = edges.patch_boxes;
    auto &patch_data_layers = edges.patch_data_layers;
    const f64 offset_grid   = edges.shear_offset.value;

    // outputs
    auto &infos_out = edges.shear_src_infos.values;
    auto &idx_out   = edges.idx_shear_src.buffers;
    infos_out       = {};
    idx_out         = {};

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    const u32 ib = geom.base_axis;
    const u32 id = geom.dir_axis;
    const u32 it = get_third_axis(geom);

    const TgridVec box_size = sim_box.upper - sim_box.lower;

    // Depth of the ghost layer (one block) and padding for the slopes (one block). Mesh refinement
    // is refused with this boundary, so every block has this extent.
    const Tgridscal D = get_global_max_block_extent<TgridVec>(dev_sched, patch_data_layers);

    if (D <= 0) {
        return; // no block anywhere
    }

    // Region (in the receiver frame) that the source cells must cover for the ghosts of the
    // receiver patch `rbox` beyond the face on side `side`: the ghost slab of depth D (the
    // ghosts touch the patch, so extend D around it), plus one block and the residual shift
    // (< 1) along the shear direction.
    auto get_needed_region = [&](const AABB &rbox, i32 side) -> AABB {
        TgridVec lo = rbox.lower - D;
        TgridVec hi = rbox.upper + D;

        lo[id] -= D + 1;
        hi[id] += D + 1;

        if (side > 0) {
            lo[ib] = std::max(lo[ib], sim_box.upper[ib]);
            hi[ib] = std::min(hi[ib], sim_box.upper[ib] + D);
        } else {
            lo[ib] = std::max(lo[ib], sim_box.lower[ib] - D);
            hi[ib] = std::min(hi[ib], sim_box.lower[ib]);
        }

        return {lo, hi};
    };

    // conservative margin used to prune the patch tree
    const Tgridscal margin = 2 * D + 1;

    const i32 rep_third = geom.third_periodic ? 1 : 0;

    patch_data_layers.get_const_refs().for_each([&](u64 sender, const auto &pdat_ref) {
        shamrock::patch::PatchDataLayer &pdat = pdat_ref.get();
        u32 cnt                               = pdat.get_obj_cnt();
        if (cnt == 0) {
            return;
        }

        auto &block_min = pdat.template get_field<TgridVec>(0).get_buf();
        auto &block_max = pdat.template get_field<TgridVec>(1).get_buf();

        const AABB sender_box = patch_boxes.values.get(sender);

        for (i32 side : {-1, 1}) {

            // the sender must touch the face opposite to the ghosts it feeds
            if (side > 0 && sender_box.lower[ib] >= sim_box.lower[ib] + D) {
                continue;
            }
            if (side < 0 && sender_box.upper[ib] <= sim_box.upper[ib] - D) {
                continue;
            }

            const ShearSideShift shift = get_shear_side_shift(offset_grid, side);

            for (i32 jdir = -1; jdir <= 1; jdir++) {
                for (i32 kthird = -rep_third; kthird <= rep_third; kthird++) {

                    TgridVec offset = {0, 0, 0};
                    offset[ib]      = side * box_size[ib];
                    offset[id]      = shift.int_shift + jdir * box_size[id];
                    offset[it]      = kthird * box_size[it];

                    const AABB placed{sender_box.lower + offset, sender_box.upper + offset};

                    patch_tree.host_for_each_leafs(
                        [&](u64 tree_id, PtNode n) {
                            AABB node{n.box_min - margin, n.box_max + margin};
                            return node.get_intersect(placed).is_not_empty();
                        },
                        [&](u64 receiver, PtNode n) {
                            const AABB needed
                                = get_needed_region(patch_boxes.values.get(receiver), side);

                            if (!needed.is_volume_not_null()
                                || !needed.get_intersect(placed).is_volume_not_null()) {
                                return;
                            }

                            sham::DeviceBuffer<u32> ids = get_ids_in_region(
                                dev_sched, block_min, block_max, offset, needed, cnt);

                            if (ids.get_size() == 0) {
                                return;
                            }

                            // both containers are filled in the same order, so that the n-th
                            // interface of a (sender, receiver) pair matches in both
                            infos_out.add_obj(
                                sender, receiver, ShearSourceInfos<TgridVec>{side, offset});
                            idx_out.add_obj(sender, receiver, std::move(ids));
                        });
                }
            }
        }
    });
}

template<class TgridVec>
std::string shammodels::basegodunov::modules::FindShearSourceGhosts<TgridVec>::_impl_get_tex()
    const {
    return "TODO";
}

template class shammodels::basegodunov::modules::FindShearSourceGhosts<i64_3>;
