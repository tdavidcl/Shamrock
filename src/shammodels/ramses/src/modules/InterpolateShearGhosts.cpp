// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file InterpolateShearGhosts.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the InterpolateShearGhosts solver graph node
 */

#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/ramses/modules/InterpolateShearGhosts.hpp"
#include "shammodels/ramses/modules/SlopeLimitedGradientUtilities.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include <type_traits>

namespace {

    /// marker of a missing neighbour
    constexpr u32 no_neigh = 0xFFFFFFFF;

    /// Integer AABB of the cell `loc` of the block [bmin, bmax)
    template<class Block, class TgridVec>
    inline shammath::AABB<TgridVec> get_cell_aabb(TgridVec bmin, TgridVec bmax, u32 loc) {
        TgridVec delta       = (bmax - bmin) / Block::side_size;
        std::array<u32, 3> c = Block::get_coord(loc);
        TgridVec lower       = bmin + delta * TgridVec{c[0], c[1], c[2]};
        return {lower, lower + delta};
    }

    /// Slope limiter selected at runtime
    template<class T>
    inline T limit_slope(shammodels::basegodunov::SlopeMode mode, T sL, T sR) {
        using SM = shammodels::basegodunov::SlopeMode;
        switch (mode) {
        case SM::VanLeer_f  : return slope_function<T, SM::VanLeer_f>(sL, sR);
        case SM::VanLeer_std: return slope_function<T, SM::VanLeer_std>(sL, sR);
        case SM::VanLeer_sym: return slope_function<T, SM::VanLeer_sym>(sL, sR);
        case SM::Minmod     : return slope_function<T, SM::Minmod>(sL, sR);
        default             : break;
        }
        return sham::VectorProperties<T>::get_zero();
    }

} // namespace

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::InterpolateShearGhosts<Tvec, TgridVec>::
    _impl_evaluate_internal() {
    __shamrock_stack_entry();
    auto edges = get_edges();

    using AABB          = shammath::AABB<TgridVec>;
    using Block         = shammodels::amr::AMRBlock<Tvec, TgridVec, 1>;
    using RTree         = shamtree::CompressedLeafBVH<u64, TgridVec, 3>;
    constexpr u32 bsize = Block::block_size;

    const AABB sim_box    = edges.sim_box.value;
    const f64 offset_grid = edges.shear_offset.value;
    auto &shear_src       = edges.shear_src.patchdatas;

    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

    // copies for the kernels (no capture of this)
    const u32 ib         = geom.base_axis;
    const u32 idir       = geom.dir_axis;
    const Tscal phi_m    = Tscal(get_shear_side_shift(offset_grid, -1).phi);
    const Tscal phi_p    = Tscal(get_shear_side_shift(offset_grid, +1).phi);
    const Tscal v_shear  = shear_speed;
    const SlopeMode mode = slope_mode;
    const i64 face_lower = sim_box.lower[ib];
    const i64 face_upper = sim_box.upper[ib];
    const u32 iblock_min = ghost_layer_layout->get_field_idx<TgridVec>("cell_min");
    const u32 iblock_max = ghost_layer_layout->get_field_idx<TgridVec>("cell_max");

    edges.patch_data_layers.get_refs().for_each([&](u64 patch_id, auto &pdat_ref) {
        shamrock::patch::PatchDataLayer &tgt = pdat_ref.get();

        // gather the source blocks received for this patch (deterministic multimap order)
        shamrock::patch::PatchDataLayer src(ghost_layer_layout);
        shear_src.for_each(
            [&](u64 sender, u64 receiver, const shamrock::patch::PatchDataLayer &layer) {
                if (receiver == patch_id) {
                    src.insert_elements(layer);
                }
            });

        const u32 n_sblocks = src.get_obj_cnt();
        const u32 n_tblocks = tgt.get_obj_cnt();
        if (n_sblocks == 0 || n_tblocks == 0) {
            return;
        }
        const u32 n_scells = n_sblocks * bsize;
        const u32 n_tcells = n_tblocks * bsize;

        auto &src_bmin = src.get_field<TgridVec>(iblock_min).get_buf();
        auto &src_bmax = src.get_field<TgridVec>(iblock_max).get_buf();
        auto &tgt_bmin = tgt.get_field<TgridVec>(iblock_min).get_buf();
        auto &tgt_bmax = tgt.get_field<TgridVec>(iblock_max).get_buf();

        ////////////////////////////////////////////////////////////////////////////////////////
        // source cells bounds and tree
        ////////////////////////////////////////////////////////////////////////////////////////

        sham::DeviceBuffer<TgridVec> src_cmin(n_scells, dev_sched);
        sham::DeviceBuffer<TgridVec> src_cmax(n_scells, dev_sched);

        sham::kernel_call(
            q,
            sham::MultiRef{src_bmin, src_bmax},
            sham::MultiRef{src_cmin, src_cmax},
            n_scells,
            [](u32 i,
               const TgridVec *__restrict bmin,
               const TgridVec *__restrict bmax,
               TgridVec *__restrict cmin,
               TgridVec *__restrict cmax) {
                u32 block = i / bsize;
                AABB cell = get_cell_aabb<Block>(bmin[block], bmax[block], i % bsize);
                cmin[i]   = cell.lower;
                cmax[i]   = cell.upper;
            });

        AABB tree_box{
            shamalgs::primitives::min(dev_sched, src_cmin, 0, n_scells),
            shamalgs::primitives::max(dev_sched, src_cmax, 0, n_scells)};

        { // the integer Morton codes need power of two extents (same as NodeBuildTrees)
            TgridVec diff  = tree_box.upper - tree_box.lower;
            diff.x()       = shambase::roundup_pow2(diff.x());
            diff.y()       = shambase::roundup_pow2(diff.y());
            diff.z()       = shambase::roundup_pow2(diff.z());
            tree_box.upper = tree_box.lower + diff;
        }

        RTree tree = RTree::make_empty(dev_sched);
        tree.rebuild_from_position_range(src_cmin, src_cmax, n_scells, tree_box, 0);

        auto obj_it = tree.get_object_iterator();

        ////////////////////////////////////////////////////////////////////////////////////////
        // neighbours of the source cells along the shear direction (for the slopes)
        ////////////////////////////////////////////////////////////////////////////////////////

        sham::DeviceBuffer<u32> neigh_lo(n_scells, dev_sched);
        sham::DeviceBuffer<u32> neigh_hi(n_scells, dev_sched);

        sham::kernel_call(
            q,
            sham::MultiRef{src_cmin, src_cmax, obj_it},
            sham::MultiRef{neigh_lo, neigh_hi},
            n_scells,
            [idir](
                u32 i,
                const TgridVec *__restrict cmin,
                const TgridVec *__restrict cmax,
                auto cell_looper,
                u32 *__restrict lo,
                u32 *__restrict hi) {
                AABB cell{cmin[i], cmax[i]};

                // smallest index among the cells intersecting the probe (unique if uniform)
                auto find = [&](AABB probe) -> u32 {
                    u32 found = no_neigh;
                    cell_looper.rtree_for(
                        [&](u32 node_id, AABB node_aabb) -> bool {
                            return node_aabb.get_intersect(probe).is_volume_not_null();
                        },
                        [&](u32 j) {
                            if (j != i
                                && AABB{cmin[j], cmax[j]}
                                       .get_intersect(probe)
                                       .is_volume_not_null()) {
                                found = sycl::min(found, j);
                            }
                        });
                    return found;
                };

                AABB probe_lo        = cell;
                probe_lo.upper[idir] = cell.lower[idir];
                probe_lo.lower[idir] = cell.lower[idir] - 1;

                AABB probe_hi        = cell;
                probe_hi.lower[idir] = cell.upper[idir];
                probe_hi.upper[idir] = cell.upper[idir] + 1;

                lo[i] = find(probe_lo);
                hi[i] = find(probe_hi);
            });

        ////////////////////////////////////////////////////////////////////////////////////////
        // interpolation stencil of the target cells (CSR)
        ////////////////////////////////////////////////////////////////////////////////////////

        // side of the face beyond which a target cell is (+1 upper, -1 lower, 0 not a target)
        auto get_side = [face_lower, face_upper, ib](const AABB &cell) -> i32 {
            if (cell.lower[ib] >= face_upper) {
                return 1;
            }
            if (cell.upper[ib] <= face_lower) {
                return -1;
            }
            return 0;
        };

        auto get_query = [idir](AABB cell, Tscal phi) -> AABB {
            if (phi > 0) {
                cell.upper[idir] += 1;
            }
            return cell;
        };

        sham::DeviceBuffer<u32> counts(n_tcells + 1, dev_sched);
        sham::DeviceBuffer<i32> tside(n_tcells, dev_sched);

        sham::kernel_call(
            q,
            sham::MultiRef{tgt_bmin, tgt_bmax, src_cmin, src_cmax, obj_it},
            sham::MultiRef{counts, tside},
            n_tcells,
            [idir, phi_m, phi_p, get_side, get_query](
                u32 t,
                const TgridVec *__restrict tbmin,
                const TgridVec *__restrict tbmax,
                const TgridVec *__restrict cmin,
                const TgridVec *__restrict cmax,
                auto cell_looper,
                u32 *__restrict cnt,
                i32 *__restrict side_out) {
                cnt[t]      = 0;
                side_out[t] = 0;

                u32 block = t / bsize;
                AABB tc   = get_cell_aabb<Block>(tbmin[block], tbmax[block], t % bsize);
                i32 side  = get_side(tc);
                if (side == 0) {
                    return;
                }

                Tscal phi   = (side > 0) ? phi_p : phi_m;
                AABB query  = get_query(tc, phi);
                TgridVec dt = tc.delt();
                i64 vol_t   = dt[0] * dt[1] * dt[2];

                u32 c        = 0;
                i64 sum_int  = 0;
                i64 sum_coef = 0;

                cell_looper.rtree_for(
                    [&](u32 node_id, AABB node_aabb) -> bool {
                        return node_aabb.get_intersect(query).is_volume_not_null();
                    },
                    [&](u32 j) {
                        auto ov
                            = compute_shear_overlap<Tscal>(tc, AABB{cmin[j], cmax[j]}, idir, phi);
                        if (ov.positive) {
                            c++;
                            sum_int += ov.vol_int;
                            sum_coef += ov.vol_coef;
                        }
                    });

                // exact coverage test, otherwise the reflective values are kept
                if (sum_int == vol_t && sum_coef == 0) {
                    cnt[t]      = c;
                    side_out[t] = side;
                }
            });

        counts.set_val_at_idx(n_tcells, 0);
        sham::DeviceBuffer<u32> offsets
            = shamalgs::numeric::scan_exclusive(dev_sched, counts, n_tcells + 1);
        const u32 n_entries = offsets.get_val_at_idx(n_tcells);

        if (n_entries == 0) {
            return;
        }

        sham::DeviceBuffer<u32> entry_src(n_entries, dev_sched);
        sham::DeviceBuffer<Tscal> entry_weight(n_entries, dev_sched);
        sham::DeviceBuffer<Tscal> entry_dr(n_entries, dev_sched);

        sham::kernel_call(
            q,
            sham::MultiRef{tgt_bmin, tgt_bmax, src_cmin, src_cmax, obj_it, tside, offsets},
            sham::MultiRef{entry_src, entry_weight, entry_dr},
            n_tcells,
            [idir, phi_m, phi_p, get_query](
                u32 t,
                const TgridVec *__restrict tbmin,
                const TgridVec *__restrict tbmax,
                const TgridVec *__restrict cmin,
                const TgridVec *__restrict cmax,
                auto cell_looper,
                const i32 *__restrict side_in,
                const u32 *__restrict offs,
                u32 *__restrict e_src,
                Tscal *__restrict e_weight,
                Tscal *__restrict e_dr) {
                i32 side = side_in[t];
                if (side == 0) {
                    return;
                }

                u32 block = t / bsize;
                AABB tc   = get_cell_aabb<Block>(tbmin[block], tbmax[block], t % bsize);

                Tscal phi       = (side > 0) ? phi_p : phi_m;
                AABB query      = get_query(tc, phi);
                TgridVec dt     = tc.delt();
                Tscal inv_vol_t = Tscal(1) / Tscal(dt[0] * dt[1] * dt[2]);

                u32 k = offs[t];

                // same traversal as the counting kernel, hence the same entries and order
                cell_looper.rtree_for(
                    [&](u32 node_id, AABB node_aabb) -> bool {
                        return node_aabb.get_intersect(query).is_volume_not_null();
                    },
                    [&](u32 j) {
                        auto ov
                            = compute_shear_overlap<Tscal>(tc, AABB{cmin[j], cmax[j]}, idir, phi);
                        if (ov.positive) {
                            e_src[k] = j;
                            e_weight[k]
                                = (Tscal(ov.vol_int) + phi * Tscal(ov.vol_coef)) * inv_vol_t;
                            e_dr[k] = ov.dr;
                            k++;
                        }
                    });
            });

        ////////////////////////////////////////////////////////////////////////////////////////
        // remap of every field (slope-limited linear reconstruction along dir)
        ////////////////////////////////////////////////////////////////////////////////////////

        auto remap_field = [&](auto &tfield, auto &sfield, u32 nvar_per_cell) {
            using T = typename std::decay_t<decltype(tfield)>::Field_type;

            const u32 m = nvar_per_cell;

            sham::DeviceBuffer<T> slopes(n_scells * m, dev_sched);

            sham::kernel_call(
                q,
                sham::MultiRef{sfield.get_buf(), neigh_lo, neigh_hi, src_cmin, src_cmax},
                sham::MultiRef{slopes},
                n_scells * m,
                [m, idir, mode](
                    u32 gid,
                    const T *__restrict U,
                    const u32 *__restrict lo,
                    const u32 *__restrict hi,
                    const TgridVec *__restrict cmin,
                    const TgridVec *__restrict cmax,
                    T *__restrict G) {
                    u32 cell = gid / m;
                    u32 v    = gid % m;
                    u32 l    = lo[cell];
                    u32 h    = hi[cell];

                    if (l == no_neigh || h == no_neigh) {
                        G[gid] = sham::VectorProperties<T>::get_zero();
                        return;
                    }

                    // distances between the cell centres, in grid units
                    i64 c2       = cmin[cell][idir] + cmax[cell][idir];
                    Tscal dist_l = Tscal(c2 - (cmin[l][idir] + cmax[l][idir])) * Tscal(0.5);
                    Tscal dist_h = Tscal((cmin[h][idir] + cmax[h][idir]) - c2) * Tscal(0.5);

                    T u  = U[gid];
                    T dL = (u - U[l * m + v]) / dist_l;
                    T dR = (U[h * m + v] - u) / dist_h;

                    G[gid] = limit_slope<T>(mode, dL, dR);
                });

            sham::kernel_call(
                q,
                sham::MultiRef{
                    sfield.get_buf(), slopes, counts, offsets, entry_src, entry_weight, entry_dr},
                sham::MultiRef{tfield.get_buf()},
                n_tcells * m,
                [m](u32 gid,
                    const T *__restrict U,
                    const T *__restrict G,
                    const u32 *__restrict cnt,
                    const u32 *__restrict offs,
                    const u32 *__restrict e_src,
                    const Tscal *__restrict e_weight,
                    const Tscal *__restrict e_dr,
                    T *__restrict out) {
                    u32 t = gid / m;
                    u32 v = gid % m;
                    u32 c = cnt[t];
                    if (c == 0) {
                        return;
                    }

                    T acc  = sham::VectorProperties<T>::get_zero();
                    u32 k0 = offs[t];
                    for (u32 k = k0; k < k0 + c; k++) {
                        u32 s = e_src[k] * m + v;
                        acc += e_weight[k] * (U[s] + G[s] * e_dr[k]);
                    }
                    out[gid] = acc;
                });
        };

        tgt.for_each_field_any([&](auto &tfield) {
            using T = typename std::decay_t<decltype(tfield)>::Field_type;

            if constexpr (std::is_same_v<T, Tscal> || std::is_same_v<T, Tvec>) {
                u32 nvar = tfield.get_nvar();
                if (nvar % bsize != 0) {
                    shambase::throw_with_loc<std::runtime_error>(sham::format(
                        "field {} has {} variables, not a multiple of the block size {}",
                        tfield.get_name(),
                        nvar,
                        bsize));
                }
                auto &sfield = src.get_field<T>(src.pdl().get_field_idx<T>(tfield.get_name()));
                remap_field(tfield, sfield, nvar / bsize);
            } else if constexpr (std::is_same_v<T, TgridVec>) {
                // block bounds, the geometry of the ghosts is kept
            } else {
                shambase::throw_unimplemented(
                    "unsupported field type in the shearing-periodic interpolation");
            }
        });

        ////////////////////////////////////////////////////////////////////////////////////////
        // velocity jump across the shearing faces
        ////////////////////////////////////////////////////////////////////////////////////////

        auto &layout = tgt.pdl();

        {
            auto &rho  = tgt.get_field<Tscal>(layout.get_field_idx<Tscal>("rho")).get_buf();
            auto &rhov = tgt.get_field<Tvec>(layout.get_field_idx<Tvec>("rhovel")).get_buf();
            auto &rhoe = tgt.get_field<Tscal>(layout.get_field_idx<Tscal>("rhoetot")).get_buf();

            sham::kernel_call(
                q,
                sham::MultiRef{tside, rho},
                sham::MultiRef{rhov, rhoe},
                n_tcells,
                [v_shear, idir](
                    u32 t,
                    const i32 *__restrict side,
                    const Tscal *__restrict rho,
                    Tvec *__restrict rhov,
                    Tscal *__restrict rhoe) {
                    i32 s = side[t];
                    if (s == 0) {
                        return;
                    }
                    Tscal w   = Tscal(s) * v_shear;
                    Tscal r   = rho[t];
                    Tvec mom  = rhov[t];
                    rhoe[t]   = rhoe[t] + w * mom[idir] + Tscal(0.5) * r * w * w;
                    mom[idir] = mom[idir] + r * w;
                    rhov[t]   = mom;
                });
        }

        if (layout.has_field_name("rho_dust") && layout.has_field_name("rhovel_dust")) {
            auto &rho_d  = tgt.get_field<Tscal>(layout.get_field_idx<Tscal>("rho_dust"));
            auto &rhov_d = tgt.get_field<Tvec>(layout.get_field_idx<Tvec>("rhovel_dust"));

            const u32 ndust = rho_d.get_nvar() / bsize;

            sham::kernel_call(
                q,
                sham::MultiRef{tside, rho_d.get_buf()},
                sham::MultiRef{rhov_d.get_buf()},
                n_tcells * ndust,
                [v_shear, idir, ndust](
                    u32 gid,
                    const i32 *__restrict side,
                    const Tscal *__restrict rho,
                    Tvec *__restrict rhov) {
                    i32 s = side[gid / ndust];
                    if (s == 0) {
                        return;
                    }
                    Tscal w   = Tscal(s) * v_shear;
                    Tvec mom  = rhov[gid];
                    mom[idir] = mom[idir] + rho[gid] * w;
                    rhov[gid] = mom;
                });
        }
    });
}

template<class Tvec, class TgridVec>
std::string shammodels::basegodunov::modules::InterpolateShearGhosts<Tvec, TgridVec>::
    _impl_get_tex() const {
    return "TODO";
}

template class shammodels::basegodunov::modules::InterpolateShearGhosts<f64_3, i64_3>;
