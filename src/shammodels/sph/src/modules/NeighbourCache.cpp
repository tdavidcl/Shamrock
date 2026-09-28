// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NeighbourCache.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/assert.hpp"
#include "shambase/memory.hpp"
#include "shambase/numeric_limits.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/impl_variants.hpp"
#include "shammodels/sph/modules/NeighbourCache.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/TreeTraversal.hpp"
#include "shamtree/kernels/geometry_utils.hpp"
#include "shamunits/Constants.hpp"

template<class Tvec, class Tmorton, template<class> class SPHKernel>
void shammodels::sph::modules::NeighbourCache<Tvec, Tmorton, SPHKernel>::start_neighbors_cache() {

    // interface_control
    using GhostHandle      = sph::BasicSPHGhostHandler<Tvec>;
    using GhostHandleCache = typename GhostHandle::CacheMap;
    using RTree            = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    auto build_neigh_cache = [&](u64 patch_id) {
        shamlog_debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        auto &mfield = storage.merged_xyzh.get().get(patch_id);

        sham::DeviceBuffer<Tvec> &buf_xyz    = mfield.template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tscal> &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);

        sham::DeviceBuffer<Tscal> &tree_field_rint
            = storage.rtree_rint_field.get().get(patch_id).buf_field;

        RTree &tree = storage.merged_pos_trees.get().get(patch_id);
        auto obj_it = tree.get_object_iterator();

        u32 obj_cnt = shambase::get_check_ref(storage.part_counts).indexes.get(patch_id);

        sycl::range range_npart{obj_cnt};

        Tscal h_tolerance = solver_config.htol_up_coarse_cycle;

        NamedStackEntry stack_loc1{"init cache"};

        using namespace shamrock;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::DeviceBuffer<u32> neigh_count(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamlog_debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);
        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, buf_hpart, tree_field_rint, obj_it},
            sham::MultiRef{neigh_count},
            obj_cnt,
            [h_tolerance](
                u32 id_a,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const Tscal *__restrict rint_tree,
                auto particle_looper,
                u32 *__restrict neigh_cnt) {
                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                u32 cnt = 0;

                particle_looper.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        using namespace walker::interaction_crit;

                        return sph_radix_cell_crit(
                            xyz_a,
                            inter_box_a_min,
                            inter_box_a_max,
                            node_aabb.lower,
                            node_aabb.upper,
                            int_r_max_cell);
                    },
                    [&](u32 id_b) {
                        // compute only omega_a
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact
                            = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        cnt += (no_interact) ? 0 : 1;
                    });

                neigh_cnt[id_a] = cnt;
            });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc2{"fill cache"};
        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, buf_hpart, tree_field_rint, pcache.scanned_cnt, obj_it},
            sham::MultiRef{pcache.index_neigh_map},
            obj_cnt,
            [h_tolerance](
                u32 id_a,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const Tscal *__restrict rint_tree,
                const u32 *__restrict scanned_neigh_cnt,
                auto particle_looper,
                u32 *__restrict neigh) {
                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                u32 cnt = scanned_neigh_cnt[id_a];

                particle_looper.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        using namespace walker::interaction_crit;

                        return sph_radix_cell_crit(
                            xyz_a,
                            inter_box_a_min,
                            inter_box_a_max,
                            node_aabb.lower,
                            node_aabb.upper,
                            int_r_max_cell);
                    },
                    [&](u32 id_b) {
                        // compute only omega_a
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact
                            = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        if (!no_interact) {
                            neigh[cnt] = id_b;
                        }
                        cnt += (no_interact) ? 0 : 1;
                    });
            });

        return pcache;
    };

    shambase::get_check_ref(storage.neigh_cache).free_alloc();

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        auto &ncache = shambase::get_check_ref(storage.neigh_cache);
        ncache.neigh_cache.add_obj(cur_p.id_patch, build_neigh_cache(cur_p.id_patch));
    });

    time_neigh.stop();
    storage.timings_details.neighbors += time_neigh.elapsed_sec();
}

template<class Tvec, class Tmorton, template<class> class SPHKernel>
void shammodels::sph::modules::NeighbourCache<Tvec, Tmorton, SPHKernel>::
    start_neighbors_cache_2stages() {

    // interface_control
    using GhostHandle      = sph::BasicSPHGhostHandler<Tvec>;
    using GhostHandleCache = typename GhostHandle::CacheMap;
    using RTree            = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    auto build_neigh_cache = [&](u64 patch_id) {
        shamlog_debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        auto &mfield = storage.merged_xyzh.get().get(patch_id);

        sham::DeviceBuffer<Tvec> &buf_xyz    = mfield.template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tscal> &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);

        sham::DeviceBuffer<Tscal> &tree_field_rint
            = storage.rtree_rint_field.get().get(patch_id).buf_field;

        RTree &tree  = storage.merged_pos_trees.get().get(patch_id);
        auto obj_it  = tree.get_object_iterator();
        auto leaf_it = tree.get_traverser();

        u32 leaf_cnt    = tree.get_leaf_cell_count();
        u32 intnode_cnt = tree.get_internal_cell_count();

        u32 obj_cnt = shambase::get_check_ref(storage.part_counts).indexes.get(patch_id);

        sycl::range range_nleaf{leaf_cnt};
        sycl::range range_nobj{obj_cnt};
        using namespace shamrock;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        Tscal h_tolerance = solver_config.htol_up_coarse_cycle;

        NamedStackEntry stack_loc1{"init cache"};

        // leaf level neighbour search : does leaf node_id possibly interact with leaf id_a
        auto leaf_interact_crit =
            [intnode_cnt](u32 id_a, const Tscal *__restrict rint_tree, const auto &leaf_looper) {
                u32 offset_leaf = intnode_cnt;

                Tscal leaf_a_rint    = rint_tree[offset_leaf + id_a] * Kernel::Rkern;
                Tvec leaf_a_bmin     = leaf_looper.aabb_min[offset_leaf + id_a];
                Tvec leaf_a_bmax     = leaf_looper.aabb_max[offset_leaf + id_a];
                Tvec leaf_a_bmin_ext = leaf_a_bmin - leaf_a_rint;
                Tvec leaf_a_bmax_ext = leaf_a_bmax + leaf_a_rint;

                return [=](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                    Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                    Tvec ext_bmin = node_aabb.lower - int_r_max_cell;
                    Tvec ext_bmax = node_aabb.upper + int_r_max_cell;

                    return BBAA::cella_neigh_b(leaf_a_bmin, leaf_a_bmax, ext_bmin, ext_bmax)
                           || BBAA::cella_neigh_b(
                               leaf_a_bmin_ext, leaf_a_bmax_ext, node_aabb.lower, node_aabb.upper);
                };
            };

        // reference : one traversal to count the leaf neighbours, one to store them
        auto leaf_passes_count_fill = [&]() -> tree::ObjectCache {
            // start by counting number of leaf neighbours

            sham::DeviceBuffer<u32> neigh_count_leaf(
                leaf_cnt, shamsys::instance::get_compute_scheduler_ptr());

            shamlog_debug_sycl_ln("Cache", "generate cache for Nleaf=", leaf_cnt);

            sham::kernel_call(
                q,
                sham::MultiRef{tree_field_rint, leaf_it},
                sham::MultiRef{neigh_count_leaf},
                leaf_cnt,
                [intnode_cnt](
                    u32 id_a,
                    const Tscal *__restrict rint_tree,
                    auto leaf_looper,
                    u32 *__restrict neigh_cnt) {
                    u32 offset_leaf = intnode_cnt;

                    Tscal leaf_a_rint    = rint_tree[offset_leaf + id_a] * Kernel::Rkern;
                    Tvec leaf_a_bmin     = leaf_looper.aabb_min[offset_leaf + id_a];
                    Tvec leaf_a_bmax     = leaf_looper.aabb_max[offset_leaf + id_a];
                    Tvec leaf_a_bmin_ext = leaf_a_bmin - leaf_a_rint;
                    Tvec leaf_a_bmax_ext = leaf_a_bmax + leaf_a_rint;

                    u32 cnt = 0;

                    leaf_looper.rtree_for(
                        [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                            Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                            Tvec ext_bmin = node_aabb.lower - int_r_max_cell;
                            Tvec ext_bmax = node_aabb.upper + int_r_max_cell;

                            return BBAA::cella_neigh_b(leaf_a_bmin, leaf_a_bmax, ext_bmin, ext_bmax)
                                   || BBAA::cella_neigh_b(
                                       leaf_a_bmin_ext,
                                       leaf_a_bmax_ext,
                                       node_aabb.lower,
                                       node_aabb.upper);
                        },
                        [&](u32 leaf_b) {
                            cnt++;
                        });

                    neigh_cnt[id_a] = cnt;
                });

            //{
            //    u32 offset_leaf = intnode_cnt;
            //    sycl::host_accessor neigh_cnt{neigh_count_leaf};
            //    sycl::host_accessor pos_min_cell
            //    {shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt)};
            //    sycl::host_accessor pos_max_cell
            //    {shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt)};
            //
            //    for (u32 i = 0; i < 1000; i++) {
            //        if(neigh_cnt[i] > 30){
            //            logger::raw_ln(i, neigh_cnt[i], pos_max_cell[i+offset_leaf] -
            //            pos_min_cell[i+offset_leaf]);
            //        }
            //    }
            //}

            tree::ObjectCache pleaf_cache
                = tree::prepare_object_cache(std::move(neigh_count_leaf), leaf_cnt);

            // fill ids of leaf neighbours

            NamedStackEntry stack_loc2{"fill cache"};

            sham::kernel_call(
                q,
                sham::MultiRef{tree_field_rint, pleaf_cache.scanned_cnt, leaf_it},
                sham::MultiRef{pleaf_cache.index_neigh_map},
                leaf_cnt,
                [intnode_cnt](
                    u32 id_a,
                    const Tscal *__restrict rint_tree,
                    const u32 *__restrict scanned_neigh_cnt,
                    auto leaf_looper,
                    u32 *__restrict neigh) {
                    u32 offset_leaf = intnode_cnt;

                    Tscal leaf_a_rint    = rint_tree[offset_leaf + id_a] * Kernel::Rkern;
                    Tvec leaf_a_bmin     = leaf_looper.aabb_min[offset_leaf + id_a];
                    Tvec leaf_a_bmax     = leaf_looper.aabb_max[offset_leaf + id_a];
                    Tvec leaf_a_bmin_ext = leaf_a_bmin - leaf_a_rint;
                    Tvec leaf_a_bmax_ext = leaf_a_bmax + leaf_a_rint;

                    u32 cnt = scanned_neigh_cnt[id_a];

                    leaf_looper.rtree_for(
                        [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                            Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                            Tvec ext_bmin = node_aabb.lower - int_r_max_cell;
                            Tvec ext_bmax = node_aabb.upper + int_r_max_cell;

                            return BBAA::cella_neigh_b(leaf_a_bmin, leaf_a_bmax, ext_bmin, ext_bmax)
                                   || BBAA::cella_neigh_b(
                                       leaf_a_bmin_ext,
                                       leaf_a_bmax_ext,
                                       node_aabb.lower,
                                       node_aabb.upper);
                        },
                        [&](u32 leaf_b) {
                            neigh[cnt] = leaf_b;
                            cnt++;
                        });
                });

            return pleaf_cache;
        };

        // single traversal : the leaf neighbours are stored in a buffer of fixed capacity per
        // leaf while counting them, the fill pass then only copies them (traversing the tree
        // again only for the leaves having more neighbours than the capacity). The traversal
        // being the same, the resulting lists are identical to the reference.
        auto leaf_passes_buffered = [&](u32 capacity) -> tree::ObjectCache {
            sham::DeviceBuffer<u32> neigh_count_leaf(
                leaf_cnt, shamsys::instance::get_compute_scheduler_ptr());

            sham::DeviceBuffer<u32> neigh_buf(
                size_t(leaf_cnt) * capacity, shamsys::instance::get_compute_scheduler_ptr());

            sham::kernel_call(
                q,
                sham::MultiRef{tree_field_rint, leaf_it},
                sham::MultiRef{neigh_count_leaf, neigh_buf},
                leaf_cnt,
                [leaf_interact_crit, capacity](
                    u32 id_a,
                    const Tscal *__restrict rint_tree,
                    auto leaf_looper,
                    u32 *__restrict neigh_cnt,
                    u32 *__restrict buf) {
                    u32 cnt     = 0;
                    u64 buf_off = u64(id_a) * capacity;

                    leaf_looper.rtree_for(
                        leaf_interact_crit(id_a, rint_tree, leaf_looper), [&](u32 leaf_b) {
                            if (cnt < capacity) {
                                buf[buf_off + cnt] = leaf_b;
                            }
                            cnt++;
                        });

                    neigh_cnt[id_a] = cnt;
                });

            tree::ObjectCache pleaf_cache
                = tree::prepare_object_cache(std::move(neigh_count_leaf), leaf_cnt);

            NamedStackEntry stack_loc2{"fill cache"};

            sham::kernel_call(
                q,
                sham::MultiRef{
                    tree_field_rint,
                    pleaf_cache.scanned_cnt,
                    pleaf_cache.cnt_neigh,
                    neigh_buf,
                    leaf_it},
                sham::MultiRef{pleaf_cache.index_neigh_map},
                leaf_cnt,
                [leaf_interact_crit, capacity](
                    u32 id_a,
                    const Tscal *__restrict rint_tree,
                    const u32 *__restrict scanned_neigh_cnt,
                    const u32 *__restrict neigh_cnt,
                    const u32 *__restrict buf,
                    auto leaf_looper,
                    u32 *__restrict neigh) {
                    u32 offset = scanned_neigh_cnt[id_a];
                    u32 cnt    = neigh_cnt[id_a];

                    if (cnt <= capacity) {
                        u64 buf_off = u64(id_a) * capacity;
                        for (u32 i = 0; i < cnt; i++) {
                            neigh[offset + i] = buf[buf_off + i];
                        }
                    } else {
                        u32 idx = offset;
                        leaf_looper.rtree_for(
                            leaf_interact_crit(id_a, rint_tree, leaf_looper), [&](u32 leaf_b) {
                                neigh[idx] = leaf_b;
                                idx++;
                            });
                    }
                });

            return pleaf_cache;
        };

        tree::ObjectCache pleaf_cache = std::visit(
            [&](const auto &variant) -> tree::ObjectCache {
                using V = std::decay_t<decltype(variant)>;
                if constexpr (std::is_same_v<V, impl::neigh_cache_leaf_pass::CountThenFill>) {
                    return leaf_passes_count_fill();
                } else {
                    return leaf_passes_buffered(variant.capacity);
                }
            },
            impl::get_impl_neigh_cache_leaf_pass());

        shamlog_debug_ln(
            "Cache",
            "leaf neighbour cache : leaf count =",
            leaf_cnt,
            "mean leaf neighbours =",
            f64(pleaf_cache.sum_neigh_cnt) / f64(leaf_cnt),
            "mean particles per leaf =",
            f64(obj_cnt) / f64(leaf_cnt));

        // search in which leaf each parts are
        sham::DeviceBuffer<u32> leaf_part_id(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, leaf_it},
            sham::MultiRef{leaf_part_id},
            obj_cnt,
            [intnode_cnt](
                u32 id_a, const Tvec *__restrict xyz, auto leaf_looper, u32 *__restrict found_id) {
                u32 offset_leaf = intnode_cnt;

                Tvec r_a = xyz[id_a];

                u32 found_id_ = i32_max; // to ensure a crash because of out of bound
                                         // access if not found

                leaf_looper.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        return BBAA::is_coord_in_range_incl_max(
                            r_a, node_aabb.lower, node_aabb.upper);
                    },
                    [&](u32 leaf_b) {
                        found_id_ = leaf_b - offset_leaf;
                    });

                SHAM_ASSERT(found_id_ < offset_leaf + 1);

                found_id[id_a] = found_id_;
            });

        //{
        //    sycl::host_accessor xyz{buf_xyz};
        //    sycl::host_accessor acc {leaf_part_id};
        //
        //    for(u32 i = 0; i < obj_cnt; i++){
        //        u32 leaf_id = acc[i];
        //        if(leaf_id >= leaf_cnt){
        //            logger::raw_ln("error : i=",i,"r=",xyz[i],"leaf_id=",leaf_id);
        //        }
        //    }
        //}

        // particle level passes (count then fill). When pruning is enabled, a neighbour leaf is
        // skipped altogether if no particle of it can interact with particle a, that is if the
        // distance from a to the leaf AABB is larger than both the interaction radius of a and
        // the largest interaction radius of the leaf. The leaf AABB being the tight bounding box
        // of its particles and the floating point operations used being monotonic, this test
        // never removes an interacting particle, hence the resulting neighbour lists (and their
        // ordering) are identical with or without pruning.
        auto particle_passes = [&](auto prune_tag, u32 slot_capacity) -> tree::ObjectCache {
            constexpr bool prune = decltype(prune_tag)::value;

            // test whether the neighbour leaf leaf_b (node id) can be skipped for particle a
            auto can_skip_leaf = [](const Tvec &xyz_a,
                                    Tscal rint_a_sq_R2,
                                    u32 leaf_b,
                                    const Tscal *__restrict rint_tree,
                                    const Tvec *__restrict aabb_min,
                                    const Tvec *__restrict aabb_max) -> bool {
                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;
                // tiny safety margin on top of the (already conservative) test
                constexpr Tscal margin = Tscal{1} - Tscal{1e-12};

                Tvec d = sham::max(
                    sham::max(aabb_min[leaf_b] - xyz_a, xyz_a - aabb_max[leaf_b]), Tvec{0, 0, 0});
                Tscal dist2 = sycl::dot(d, d) * margin;

                Tscal rint_leaf = rint_tree[leaf_b];
                return dist2 > rint_a_sq_R2 && dist2 > rint_leaf * rint_leaf * Rker2;
            };

            sham::DeviceBuffer<u32> neigh_count(
                obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

            shamlog_debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);

            // slotted layout : a single pass counts the neighbours and stores them in fixed size
            // slots (particle i owns [i*capacity, (i+1)*capacity)), the count pass being also the
            // fill pass. The lists are the same as with the compact layout, only their location
            // in the cache buffer differs. If a particle has more neighbours than the capacity,
            // the counts are kept and the compact layout is built instead.
            bool counted   = false;
            bool use_slots = slot_capacity > 0
                             && u64(obj_cnt) * u64(slot_capacity) < u64(shambase::get_max<u32>());
            if (use_slots) {
                sham::DeviceBuffer<u32> slots(
                    size_t(obj_cnt) * slot_capacity,
                    shamsys::instance::get_compute_scheduler_ptr());

                // candidate data : either read through the tree sort map (reference), or from a
                // packed copy of the positions & smoothing lengths stored in tree (morton) order,
                // making the candidate loops contiguous. The values, hence the lists, are the same.
                bool use_sorted_copy
                    = std::holds_alternative<impl::neigh_cache_candidate_data::LeafSortedCopy>(
                        impl::get_impl_neigh_cache_candidate_data());

                if (use_sorted_copy) {
                    using Tvec4 = sycl::vec<Tscal, 4>;

                    u32 merged_cnt = mfield.get_obj_cnt();
                    u32 sorted_len = obj_it.cell_iterator.buf_sort_index_map.get_size();

                    sham::DeviceBuffer<Tvec4> xyzh_sorted(
                        sorted_len, shamsys::instance::get_compute_scheduler_ptr());

                    sham::kernel_call(
                        q,
                        sham::MultiRef{buf_xyz, buf_hpart, obj_it.cell_iterator},
                        sham::MultiRef{xyzh_sorted},
                        sorted_len,
                        [merged_cnt](
                            u32 k,
                            const Tvec *__restrict xyz,
                            const Tscal *__restrict hpart,
                            auto cell_it,
                            Tvec4 *__restrict out) {
                            u32 id = cell_it.sort_index_map[k];
                            if (id < merged_cnt) {
                                Tvec r = xyz[id];
                                out[k] = Tvec4{r.x(), r.y(), r.z(), hpart[id]};
                            } else {
                                out[k] = Tvec4{0, 0, 0, 0};
                            }
                        });

                    sham::kernel_call(
                        q,
                        sham::MultiRef{
                            buf_xyz,
                            buf_hpart,
                            pleaf_cache,
                            obj_it.cell_iterator,
                            leaf_part_id,
                            tree_field_rint,
                            tree.aabbs.buf_aabb_min,
                            tree.aabbs.buf_aabb_max,
                            xyzh_sorted},
                        sham::MultiRef{neigh_count, slots},
                        obj_cnt,
                        [intnode_cnt, h_tolerance, can_skip_leaf, slot_capacity](
                            u32 id_a,
                            const Tvec *__restrict xyz,
                            const Tscal *__restrict hpart,
                            auto acc_neigh_leaf_looper,
                            auto cell_it,
                            const u32 *__restrict leaf_owner,
                            const Tscal *__restrict rint_tree,
                            const Tvec *__restrict aabb_min,
                            const Tvec *__restrict aabb_max,
                            const Tvec4 *__restrict xyzh_s,
                            u32 *__restrict neigh_cnt,
                            u32 *__restrict slot_neigh) {
                            tree::ObjectCacheIterator neigh_leaf_looper(acc_neigh_leaf_looper);

                            u32 offset_leaf = intnode_cnt;

                            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                            Tscal rint_a = hpart[id_a] * h_tolerance;

                            Tvec xyz_a = xyz[id_a];

                            Tscal rint_a_sq_R2 = rint_a * rint_a * Rker2;

                            u32 cnt      = 0;
                            u64 slot_off = u64(id_a) * slot_capacity;

                            u32 leaf_own_a = leaf_owner[id_a];

                            neigh_leaf_looper.for_each_object(leaf_own_a, [&](u32 leaf_b) {
                                if constexpr (prune) {
                                    if (can_skip_leaf(
                                            xyz_a,
                                            rint_a_sq_R2,
                                            leaf_b,
                                            rint_tree,
                                            aabb_min,
                                            aabb_max)) {
                                        return;
                                    }
                                }

                                u32 cell = leaf_b - offset_leaf;
                                u32 kmin = cell_it.reduc_index_map[cell];
                                u32 kmax = cell_it.reduc_index_map[cell + 1];

                                for (u32 k = kmin; k < kmax; k++) {
                                    Tvec4 p      = xyzh_s[k];
                                    Tvec xyz_b   = {p.x(), p.y(), p.z()};
                                    Tvec dr      = xyz_a - xyz_b;
                                    Tscal rab2   = sycl::dot(dr, dr);
                                    Tscal rint_b = p.w() * h_tolerance;

                                    bool no_interact = rab2 > rint_a * rint_a * Rker2
                                                       && rab2 > rint_b * rint_b * Rker2;

                                    if (!no_interact && cnt < slot_capacity) {
                                        slot_neigh[slot_off + cnt] = cell_it.sort_index_map[k];
                                    }
                                    cnt += (no_interact) ? 0 : 1;
                                }
                            });

                            neigh_cnt[id_a] = cnt;
                        });
                } else {
                    sham::kernel_call(
                        q,
                        sham::MultiRef{
                            buf_xyz,
                            buf_hpart,
                            pleaf_cache,
                            obj_it.cell_iterator,
                            leaf_part_id,
                            tree_field_rint,
                            tree.aabbs.buf_aabb_min,
                            tree.aabbs.buf_aabb_max},
                        sham::MultiRef{neigh_count, slots},
                        obj_cnt,
                        [intnode_cnt, h_tolerance, can_skip_leaf, slot_capacity](
                            u32 id_a,
                            const Tvec *__restrict xyz,
                            const Tscal *__restrict hpart,
                            auto acc_neigh_leaf_looper,
                            auto particle_looper,
                            const u32 *__restrict leaf_owner,
                            const Tscal *__restrict rint_tree,
                            const Tvec *__restrict aabb_min,
                            const Tvec *__restrict aabb_max,
                            u32 *__restrict neigh_cnt,
                            u32 *__restrict slot_neigh) {
                            tree::ObjectCacheIterator neigh_leaf_looper(acc_neigh_leaf_looper);

                            u32 offset_leaf = intnode_cnt;

                            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                            Tscal rint_a = hpart[id_a] * h_tolerance;

                            Tvec xyz_a = xyz[id_a];

                            Tscal rint_a_sq_R2 = rint_a * rint_a * Rker2;

                            u32 cnt      = 0;
                            u64 slot_off = u64(id_a) * slot_capacity;

                            u32 leaf_own_a = leaf_owner[id_a];

                            neigh_leaf_looper.for_each_object(leaf_own_a, [&](u32 leaf_b) {
                                SHAM_ASSERT(leaf_b >= offset_leaf);

                                if constexpr (prune) {
                                    if (can_skip_leaf(
                                            xyz_a,
                                            rint_a_sq_R2,
                                            leaf_b,
                                            rint_tree,
                                            aabb_min,
                                            aabb_max)) {
                                        return;
                                    }
                                }

                                particle_looper.for_each_in_leaf_cell(
                                    leaf_b - offset_leaf, [&](u32 id_b) {
                                        Tvec dr      = xyz_a - xyz[id_b];
                                        Tscal rab2   = sycl::dot(dr, dr);
                                        Tscal rint_b = hpart[id_b] * h_tolerance;

                                        bool no_interact = rab2 > rint_a * rint_a * Rker2
                                                           && rab2 > rint_b * rint_b * Rker2;

                                        if (!no_interact && cnt < slot_capacity) {
                                            slot_neigh[slot_off + cnt] = id_b;
                                        }
                                        cnt += (no_interact) ? 0 : 1;
                                    });
                            });

                            neigh_cnt[id_a] = cnt;
                        });
                }

                u32 max_cnt = shamalgs::primitives::max(
                    shamsys::instance::get_compute_scheduler_ptr(), neigh_count, 0, obj_cnt);

                if (max_cnt <= slot_capacity) {
                    sham::DeviceBuffer<u32> slot_offsets(
                        obj_cnt, shamsys::instance::get_compute_scheduler_ptr());
                    sham::kernel_call(
                        q,
                        sham::MultiRef{},
                        sham::MultiRef{slot_offsets},
                        obj_cnt,
                        [slot_capacity](u32 i, u32 *__restrict offs) {
                            offs[i] = i * slot_capacity;
                        });

                    return tree::ObjectCache{
                        .cnt_neigh       = std::move(neigh_count),
                        .scanned_cnt     = std::move(slot_offsets),
                        .sum_neigh_cnt   = obj_cnt * slot_capacity,
                        .index_neigh_map = std::move(slots)};
                }

                shamlog_debug_ln(
                    "Cache",
                    "neighbour count",
                    max_cnt,
                    "larger than the slot capacity",
                    slot_capacity,
                    ", using the compact layout");
                counted = true;
            }

            if (!counted) {
                sham::kernel_call(
                    q,
                    sham::MultiRef{
                        buf_xyz,
                        buf_hpart,
                        pleaf_cache,
                        obj_it.cell_iterator,
                        leaf_part_id,
                        tree_field_rint,
                        tree.aabbs.buf_aabb_min,
                        tree.aabbs.buf_aabb_max},
                    sham::MultiRef{neigh_count},
                    obj_cnt,
                    [intnode_cnt, h_tolerance, can_skip_leaf](
                        u32 id_a,
                        const Tvec *__restrict xyz,
                        const Tscal *__restrict hpart,
                        auto acc_neigh_leaf_looper,
                        auto particle_looper,
                        const u32 *__restrict leaf_owner,
                        const Tscal *__restrict rint_tree,
                        const Tvec *__restrict aabb_min,
                        const Tvec *__restrict aabb_max,
                        u32 *__restrict neigh_cnt) {
                        tree::ObjectCacheIterator neigh_leaf_looper(acc_neigh_leaf_looper);

                        u32 offset_leaf = intnode_cnt;

                        constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                        Tscal rint_a = hpart[id_a] * h_tolerance;

                        Tvec xyz_a = xyz[id_a];

                        Tscal rint_a_sq_R2 = rint_a * rint_a * Rker2;

                        u32 cnt = 0;

                        u32 leaf_own_a = leaf_owner[id_a];

                        neigh_leaf_looper.for_each_object(leaf_own_a, [&](u32 leaf_b) {
                            SHAM_ASSERT(leaf_b >= offset_leaf);

                            if constexpr (prune) {
                                if (can_skip_leaf(
                                        xyz_a,
                                        rint_a_sq_R2,
                                        leaf_b,
                                        rint_tree,
                                        aabb_min,
                                        aabb_max)) {
                                    return;
                                }
                            }

                            particle_looper.for_each_in_leaf_cell(
                                leaf_b - offset_leaf, [&](u32 id_b) {
                                    Tvec dr      = xyz_a - xyz[id_b];
                                    Tscal rab2   = sycl::dot(dr, dr);
                                    Tscal rint_b = hpart[id_b] * h_tolerance;

                                    bool no_interact = rab2 > rint_a * rint_a * Rker2
                                                       && rab2 > rint_b * rint_b * Rker2;

                                    cnt += (no_interact) ? 0 : 1;
                                });
                        });

                        neigh_cnt[id_a] = cnt;
                    });
            }

            tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

            NamedStackEntry stack_loc3{"fill cache"};

            sham::kernel_call(
                q,
                sham::MultiRef{
                    buf_xyz,
                    buf_hpart,
                    pleaf_cache,
                    pcache.scanned_cnt,
                    obj_it.cell_iterator,
                    leaf_part_id,
                    tree_field_rint,
                    tree.aabbs.buf_aabb_min,
                    tree.aabbs.buf_aabb_max},
                sham::MultiRef{pcache.index_neigh_map},
                obj_cnt,
                [intnode_cnt, h_tolerance, can_skip_leaf](
                    u32 id_a,
                    const Tvec *__restrict xyz,
                    const Tscal *__restrict hpart,
                    auto acc_neigh_leaf_looper,
                    const u32 *__restrict scanned_neigh_cnt,
                    auto particle_looper,
                    const u32 *__restrict leaf_owner,
                    const Tscal *__restrict rint_tree,
                    const Tvec *__restrict aabb_min,
                    const Tvec *__restrict aabb_max,
                    u32 *__restrict neigh) {
                    tree::ObjectCacheIterator neigh_leaf_looper(acc_neigh_leaf_looper);

                    u32 offset_leaf = intnode_cnt;

                    constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                    Tscal rint_a = hpart[id_a] * h_tolerance;

                    Tvec xyz_a = xyz[id_a];

                    Tscal rint_a_sq_R2 = rint_a * rint_a * Rker2;

                    u32 cnt = scanned_neigh_cnt[id_a];

                    u32 leaf_own_a = leaf_owner[id_a];

                    neigh_leaf_looper.for_each_object(leaf_own_a, [&](u32 leaf_b) {
                        SHAM_ASSERT(leaf_b >= offset_leaf);

                        if constexpr (prune) {
                            if (can_skip_leaf(
                                    xyz_a, rint_a_sq_R2, leaf_b, rint_tree, aabb_min, aabb_max)) {
                                return;
                            }
                        }

                        particle_looper.for_each_in_leaf_cell(leaf_b - offset_leaf, [&](u32 id_b) {
                            Tvec dr      = xyz_a - xyz[id_b];
                            Tscal rab2   = sycl::dot(dr, dr);
                            Tscal rint_b = hpart[id_b] * h_tolerance;

                            bool no_interact
                                = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                            if (!no_interact) {
                                neigh[cnt] = id_b;
                            }
                            cnt += (no_interact) ? 0 : 1;
                        });
                    });
                });

            return pcache;
        };

        bool prune_leaves = std::holds_alternative<impl::neigh_cache_particle_pass::PruneLeaves>(
            impl::get_impl_neigh_cache_particle_pass());

        u32 slot_capacity = std::visit(
            [](const auto &variant) -> u32 {
                using V = std::decay_t<decltype(variant)>;
                if constexpr (std::is_same_v<V, impl::neigh_cache_particle_layout::Slots>) {
                    return variant.capacity;
                } else {
                    return 0;
                }
            },
            impl::get_impl_neigh_cache_particle_layout());

        if (prune_leaves) {
            return particle_passes(std::true_type{}, slot_capacity);
        } else {
            return particle_passes(std::false_type{}, slot_capacity);
        }
    };

    shambase::get_check_ref(storage.neigh_cache).free_alloc();

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        auto &ncache = shambase::get_check_ref(storage.neigh_cache);
        ncache.neigh_cache.add_obj(cur_p.id_patch, build_neigh_cache(cur_p.id_patch));
    });

    time_neigh.stop();
    storage.timings_details.neighbors += time_neigh.elapsed_sec();
}

using namespace shammath;
template class shammodels::sph::modules::NeighbourCache<f64_3, u32, M4>;
template class shammodels::sph::modules::NeighbourCache<f64_3, u32, M6>;
template class shammodels::sph::modules::NeighbourCache<f64_3, u32, M8>;

template class shammodels::sph::modules::NeighbourCache<f64_3, u32, C2>;
template class shammodels::sph::modules::NeighbourCache<f64_3, u32, C4>;
template class shammodels::sph::modules::NeighbourCache<f64_3, u32, C6>;
