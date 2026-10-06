// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FindBlockNeighOpt.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Optimised variant of FindBlockNeigh (NeighGraphStrategy::NeighGraphOpt), builds the same
 * block graph
 */

#include "shambase/stacktrace.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/make_ndrange.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/ramses/modules/FindBlockNeighOpt.hpp"
#include "shammodels/ramses/modules/details/compute_neigh_graph.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace {

    /// work group size of the fused block graph kernels
    constexpr u32 block_finder_group_size = 64;

    /// number of traversal stack entries (from the bottom) kept in work-group local memory
    constexpr u32 block_finder_shared_stack_depth = 24;

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec, class Tmorton>
    class FindBlockNeighOpt<Tvec, TgridVec, Tmorton>::AMRBlockFinder {
        public:
        using acc_u32 = sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device>;
        using acc_u8  = sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device>;
        using acc_grid
            = sycl::accessor<TgridVec, 1, sycl::access::mode::read, sycl::target::device>;

        static constexpr u32 tree_depth = RTree::tree_depth;
        static constexpr u32 _nindex    = 4294967295;

        // radix tree (same data as shamrock::tree::ObjectIterator)
        acc_u32 particle_index_map;
        acc_u32 cell_index_map;
        acc_u32 rchild_id;
        acc_u32 lchild_id;
        acc_u8 rchild_flag;
        acc_u8 lchild_flag;
        acc_grid pos_min_cell;
        acc_grid pos_max_cell;
        u32 leaf_offset;

        acc_grid acc_block_min;
        acc_grid acc_block_max;

        TgridVec dir_offset;

        AMRBlockFinder(
            sycl::handler &cgh,
            const RTree &tree,
            sycl::buffer<TgridVec> &buf_block_min,
            sycl::buffer<TgridVec> &buf_block_max,
            TgridVec dir_offset)
            : particle_index_map{
                  shambase::get_check_ref(tree.tree_morton_codes.buf_particle_index_map),
                  cgh,
                  sycl::read_only},
              cell_index_map{
                  shambase::get_check_ref(tree.tree_reduced_morton_codes.buf_reduc_index_map),
                  cgh,
                  sycl::read_only},
              rchild_id{
                  shambase::get_check_ref(tree.tree_struct.buf_rchild_id), cgh, sycl::read_only},
              lchild_id{
                  shambase::get_check_ref(tree.tree_struct.buf_lchild_id), cgh, sycl::read_only},
              rchild_flag{
                  shambase::get_check_ref(tree.tree_struct.buf_rchild_flag), cgh, sycl::read_only},
              lchild_flag{
                  shambase::get_check_ref(tree.tree_struct.buf_lchild_flag), cgh, sycl::read_only},
              pos_min_cell{
                  shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt),
                  cgh,
                  sycl::read_only},
              pos_max_cell{
                  shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt),
                  cgh,
                  sycl::read_only},
              leaf_offset(tree.tree_struct.internal_cell_count),
              acc_block_min{buf_block_min, cgh, sycl::read_only},
              acc_block_max{buf_block_max, cgh, sycl::read_only},
              dir_offset(std::move(dir_offset)) {}

        template<class IndexFunctor>
        void for_each_other_index(u32 id_a, IndexFunctor &&fct) const {

            // current block AABB
            shammath::AABB<TgridVec> block_aabb{acc_block_min[id_a], acc_block_max[id_a]};

            // The wanted AABB (the block we look for)
            shammath::AABB<TgridVec> check_aabb{
                block_aabb.lower + dir_offset, block_aabb.upper + dir_offset};

            auto node_test = [&](u32 node_id) -> bool {
                return shammath::AABB<TgridVec>{pos_min_cell[node_id], pos_max_cell[node_id]}
                    .get_intersect(check_aabb)
                    .is_volume_not_null();
            };

            auto on_object = [&](u32 id_b) {
                bool interact = shammath::AABB<TgridVec>{acc_block_min[id_b], acc_block_max[id_b]}
                                    .get_intersect(check_aabb)
                                    .is_volume_not_null()
                                && id_b != id_a;

                if (interact) {
                    fct(id_b);
                }
            };

            // Same depth first traversal as ObjectIterator::rtree_for, but the inner loop stops
            // on the first hit leaf, whose objects are scanned once the warp has reconverged
            u32 stack_cursor = tree_depth - 1;
            std::array<u32, tree_depth> id_stack;
            id_stack[stack_cursor] = 0;

            while (stack_cursor < tree_depth) {

                u32 found_leaf = _nindex;

                while (stack_cursor < tree_depth) {
                    u32 current_node_id = id_stack[stack_cursor];
                    stack_cursor++;

                    if (node_test(current_node_id)) {
                        if (current_node_id >= leaf_offset) {
                            found_leaf = current_node_id;
                            break;
                        }

                        u32 lid = lchild_id[current_node_id]
                                  + leaf_offset * lchild_flag[current_node_id];
                        u32 rid = rchild_id[current_node_id]
                                  + leaf_offset * rchild_flag[current_node_id];

                        id_stack[stack_cursor - 1] = rid;
                        stack_cursor--;

                        id_stack[stack_cursor - 1] = lid;
                        stack_cursor--;
                    }
                }

                if (found_leaf != _nindex) {
                    u32 min_ids = cell_index_map[found_leaf - leaf_offset];
                    u32 max_ids = cell_index_map[found_leaf + 1 - leaf_offset];
                    for (u32 id_s = min_ids; id_s < max_ids; id_s++) {
                        on_object(particle_index_map[id_s]);
                    }
                }
            }
        }
    };

    /**
     * @brief Block finder working on packed i32 records relative to the first block of the patch,
     * for the 6 directions at once
     *
     * node_rec holds two vec<i32, 4> per tree node: {lower - origin, a} and {upper - origin, b},
     * with (a, b) the left & right child ids of an internal node, or the range of objects of a
     * leaf. obj_rec holds two vec<i32, 4> per object, in leaf order: {lower - origin, block id}
     * and {upper - origin, 0}. The intersection tests are translation invariant on integers, so
     * as long as every coordinate fits (checked when building the records) they give exactly the
     * same results as the TgridVec ones of AMRBlockFinder.
     *
     * One traversal serves the 6 directions of OrientedAMRGraph::offset_check: each stack entry
     * carries the mask of the directions for which the node and all its ancestors are hit, so a
     * direction sees exactly the nodes, leaves and objects of its own traversal, in the same
     * order.
     */
    template<class Tvec, class TgridVec, class Tmorton>
    class FindBlockNeighOpt<Tvec, TgridVec, Tmorton>::AMRBlockFinderI32 {
        public:
        static constexpr u32 tree_depth = RTree::tree_depth;

        /// stack entries are (direction mask << dir_mask_shift) | node id
        static constexpr u32 dir_mask_shift = 26;
        static constexpr u32 node_id_mask   = (u32(1) << dir_mask_shift) - 1;
        static constexpr u32 all_dirs       = 0x3F;

        sham::DeviceBuffer<sycl::vec<i32, 4>> &node_rec;
        sham::DeviceBuffer<sycl::vec<i32, 4>> &obj_rec;
        sham::DeviceBuffer<TgridVec> &buf_block_min;
        sham::DeviceBuffer<TgridVec> &buf_block_max;
        u32 leaf_offset;
        TgridVec origin;

        struct ro_access {
            const sycl::vec<i32, 4> *node_rec;
            const sycl::vec<i32, 4> *obj_rec;
            const TgridVec *block_min;
            const TgridVec *block_max;
            u32 leaf_offset;
            TgridVec origin;

            /**
             * @brief Call fct(dir_mask, id_b) on every block id_b intersecting the block id_a
             * shifted in one of the directions, dir_mask holding these directions (bit d for
             * offset_check[d] = +x, -x, +y, -y, +z, -z)
             */
            template<class Func>
            void for_each_neigh_6dir(u32 id_a, u32 *sh_stack, u32 sh_stride, Func &&fct) const {

                TgridVec qlo64 = block_min[id_a] - origin;
                TgridVec qup64 = block_max[id_a] - origin;

                i32_3 qlo = {i32(qlo64.x()), i32(qlo64.y()), i32(qlo64.z())};
                i32_3 qup = {i32(qup64.x()), i32(qup64.y()), i32(qup64.z())};

                // directions whose shifted query box intersects the box [lo, up] (non null
                // volume), from the overlaps per axis of the query shifted by -1, 0 or +1
                auto dir_hits = [&](sycl::vec<i32, 4> lo, sycl::vec<i32, 4> up) -> u32 {
                    auto ov = [](i32 lo, i32 up, i32 qlo, i32 qup, i32 s) -> bool {
                        return sycl::min(up, qup + s) > sycl::max(lo, qlo + s);
                    };

                    bool x0 = ov(lo.x(), up.x(), qlo.x(), qup.x(), 0);
                    bool y0 = ov(lo.y(), up.y(), qlo.y(), qup.y(), 0);
                    bool z0 = ov(lo.z(), up.z(), qlo.z(), qup.z(), 0);

                    u32 m = 0;
                    if (y0 && z0) {
                        m |= u32(ov(lo.x(), up.x(), qlo.x(), qup.x(), 1)) << 0;
                        m |= u32(ov(lo.x(), up.x(), qlo.x(), qup.x(), -1)) << 1;
                    }
                    if (x0 && z0) {
                        m |= u32(ov(lo.y(), up.y(), qlo.y(), qup.y(), 1)) << 2;
                        m |= u32(ov(lo.y(), up.y(), qlo.y(), qup.y(), -1)) << 3;
                    }
                    if (x0 && y0) {
                        m |= u32(ov(lo.z(), up.z(), qlo.z(), qup.z(), 1)) << 4;
                        m |= u32(ov(lo.z(), up.z(), qlo.z(), qup.z(), -1)) << 5;
                    }
                    return m;
                };

                // Same traversal as AMRBlockFinder::for_each_other_index, the direction mask
                // travelling with the node ids. The stack grows down from tree_depth - 1: its
                // first block_finder_shared_stack_depth entries live in work-group local memory
                // (sh_stack, entry j at sh_stack[j * sh_stride]), deeper ones in id_stack.
                constexpr u32 sh_lo = tree_depth - block_finder_shared_stack_depth;

                std::array<u32, tree_depth> id_stack;

                auto stack_get = [&](u32 idx) -> u32 {
                    return (idx >= sh_lo) ? sh_stack[(idx - sh_lo) * sh_stride] : id_stack[idx];
                };
                auto stack_set = [&](u32 idx, u32 val) {
                    if (idx >= sh_lo) {
                        sh_stack[(idx - sh_lo) * sh_stride] = val;
                    } else {
                        id_stack[idx] = val;
                    }
                };

                u32 stack_cursor = tree_depth - 1;
                stack_set(stack_cursor, (all_dirs << dir_mask_shift) | 0);

                while (stack_cursor < tree_depth) {

                    u32 leaf_dirs = 0;
                    u32 obj_begin = 0;
                    u32 obj_end   = 0;

                    while (stack_cursor < tree_depth) {
                        u32 entry = stack_get(stack_cursor);
                        stack_cursor++;

                        u32 current_node_id = entry & node_id_mask;

                        sycl::vec<i32, 4> lo = node_rec[2 * current_node_id];
                        sycl::vec<i32, 4> up = node_rec[2 * current_node_id + 1];

                        u32 dirs = (entry >> dir_mask_shift) & dir_hits(lo, up);

                        if (dirs != 0) {
                            if (current_node_id >= leaf_offset) {
                                leaf_dirs = dirs;
                                obj_begin = u32(lo.w());
                                obj_end   = u32(up.w());
                                break;
                            }

                            stack_set(stack_cursor - 1, (dirs << dir_mask_shift) | u32(up.w()));
                            stack_cursor--;

                            stack_set(stack_cursor - 1, (dirs << dir_mask_shift) | u32(lo.w()));
                            stack_cursor--;
                        }
                    }

                    if (leaf_dirs != 0) {
                        for (u32 id_s = obj_begin; id_s < obj_end; id_s++) {
                            sycl::vec<i32, 4> lo = obj_rec[2 * id_s];
                            sycl::vec<i32, 4> up = obj_rec[2 * id_s + 1];
                            u32 id_b             = u32(lo.w());

                            u32 dirs = (id_b != id_a) ? (leaf_dirs & dir_hits(lo, up)) : 0;

                            if (dirs != 0) {
                                fct(dirs, id_b);
                            }
                        }
                    }
                }
            }
        };

        ro_access get_read_access(sham::EventList &deps) {
            return ro_access{
                node_rec.get_read_access(deps),
                obj_rec.get_read_access(deps),
                buf_block_min.get_read_access(deps),
                buf_block_max.get_read_access(deps),
                leaf_offset,
                origin};
        }

        void complete_event_state(sycl::event &e) {
            node_rec.complete_event_state(e);
            obj_rec.complete_event_state(e);
            buf_block_min.complete_event_state(e);
            buf_block_max.complete_event_state(e);
        }
    };

    template<class Tvec, class TgridVec, class Tmorton>
    void FindBlockNeighOpt<Tvec, TgridVec, Tmorton>::_impl_evaluate_internal() {
        __shamrock_stack_entry();

        auto edges = get_edges();

        edges.spans_block_min.check_sizes(edges.sizes.indexes);
        edges.spans_block_max.check_sizes(edges.sizes.indexes);

        shambase::DistributedData<OrientedAMRGraph> graph;

        edges.trees.trees.for_each([&](u64 id, const RTree &tree) {
            u32 leaf_count          = tree.tree_reduced_morton_codes.tree_leaf_count;
            u32 internal_cell_count = tree.tree_struct.internal_cell_count;
            u32 tot_count           = leaf_count + internal_cell_count;

            OrientedAMRGraph result;

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sycl::buffer<TgridVec> &tree_bmin
                = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt);
            sycl::buffer<TgridVec> &tree_bmax
                = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt);

            PatchDataField<TgridVec> &block_min = edges.spans_block_min.get_refs().get(id);
            PatchDataField<TgridVec> &block_max = edges.spans_block_max.get_refs().get(id);

            u32 block_count = edges.sizes.indexes.get(id);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            // packed i32 records of the tree nodes and of the objects (blocks), relative to the
            // first block of the patch. The flag is raised if a coordinate does not fit, in which
            // case the TgridVec path is used for this patch.
            TgridVec origin
                = (block_count > 0) ? block_min.get_buf().get_val_at_idx(0) : TgridVec{};

            sham::DeviceBuffer<sycl::vec<i32, 4>> node_rec(2 * tot_count, dev_sched);
            sham::DeviceBuffer<sycl::vec<i32, 4>> obj_rec(2 * block_count, dev_sched);
            sham::DeviceBuffer<u32> out_of_range(1, dev_sched);
            out_of_range.set_val_at_idx(0, 0);

            if (block_count > 0) {
                // bound on the relative coordinates, so that the query boxes (+-1) and every
                // comparison stay far from the i32 limits
                constexpr i64 max_rel = i64(1) << 30;

                auto to_rel = [](TgridVec v, TgridVec origin, bool &bad) -> sycl::vec<i32, 3> {
                    TgridVec d = v - origin;
                    bad = bad || (d.x() < -max_rel) || (d.x() > max_rel) || (d.y() < -max_rel)
                          || (d.y() > max_rel) || (d.z() < -max_rel) || (d.z() > max_rel);
                    return {i32(d.x()), i32(d.y()), i32(d.z())};
                };

                sham::EventList deps;
                auto rec  = node_rec.get_write_access(deps);
                auto flag = out_of_range.get_write_access(deps);

                auto e = q.submit(deps, [&, origin, tot_count](sycl::handler &cgh) {
                    sycl::accessor pos_min{tree_bmin, cgh, sycl::read_only};
                    sycl::accessor pos_max{tree_bmax, cgh, sycl::read_only};
                    sycl::accessor lchild_id{
                        shambase::get_check_ref(tree.tree_struct.buf_lchild_id),
                        cgh,
                        sycl::read_only};
                    sycl::accessor rchild_id{
                        shambase::get_check_ref(tree.tree_struct.buf_rchild_id),
                        cgh,
                        sycl::read_only};
                    sycl::accessor lchild_flag{
                        shambase::get_check_ref(tree.tree_struct.buf_lchild_flag),
                        cgh,
                        sycl::read_only};
                    sycl::accessor rchild_flag{
                        shambase::get_check_ref(tree.tree_struct.buf_rchild_flag),
                        cgh,
                        sycl::read_only};
                    sycl::accessor cell_index_map{
                        shambase::get_check_ref(tree.tree_reduced_morton_codes.buf_reduc_index_map),
                        cgh,
                        sycl::read_only};
                    u32 leaf_offset = internal_cell_count;

                    shambase::parallel_for(cgh, tot_count, "pack i32 node records", [=](u64 gid) {
                        u32 node = (u32) gid;

                        bool bad             = false;
                        sycl::vec<i32, 3> lo = to_rel(pos_min[node], origin, bad);
                        sycl::vec<i32, 3> up = to_rel(pos_max[node], origin, bad);

                        u32 a, b;
                        if (node >= leaf_offset) {
                            a = cell_index_map[node - leaf_offset];
                            b = cell_index_map[node + 1 - leaf_offset];
                        } else {
                            a = lchild_id[node] + leaf_offset * lchild_flag[node];
                            b = rchild_id[node] + leaf_offset * rchild_flag[node];
                        }

                        rec[2 * node]     = {lo.x(), lo.y(), lo.z(), i32(a)};
                        rec[2 * node + 1] = {up.x(), up.y(), up.z(), i32(b)};

                        if (bad) {
                            flag[0] = 1;
                        }
                    });
                });

                node_rec.complete_event_state(e);
                out_of_range.complete_event_state(e);

                sham::EventList deps2;
                auto orec  = obj_rec.get_write_access(deps2);
                auto flag2 = out_of_range.get_write_access(deps2);
                auto bmin  = block_min.get_buf().get_read_access(deps2);
                auto bmax  = block_max.get_buf().get_read_access(deps2);

                auto e2 = q.submit(deps2, [&, origin, block_count](sycl::handler &cgh) {
                    sycl::accessor particle_index_map{
                        shambase::get_check_ref(tree.tree_morton_codes.buf_particle_index_map),
                        cgh,
                        sycl::read_only};

                    shambase::parallel_for(
                        cgh, block_count, "pack i32 object records", [=](u64 gid) {
                            u32 id_s = (u32) gid;
                            u32 id_b = particle_index_map[id_s];

                            bool bad             = false;
                            sycl::vec<i32, 3> lo = to_rel(bmin[id_b], origin, bad);
                            sycl::vec<i32, 3> up = to_rel(bmax[id_b], origin, bad);

                            orec[2 * id_s]     = {lo.x(), lo.y(), lo.z(), i32(id_b)};
                            orec[2 * id_s + 1] = {up.x(), up.y(), up.z(), 0};

                            if (bad) {
                                flag2[0] = 1;
                            }
                        });
                });

                obj_rec.complete_event_state(e2);
                out_of_range.complete_event_state(e2);
                block_min.get_buf().complete_event_state(e2);
                block_max.get_buf().complete_event_state(e2);
            }

            // the fused traversal assumes the unit offsets +x, -x, +y, -y, +z, -z and packs the
            // node ids with the direction masks
            const std::array<TgridVec, 6> unit_offsets{
                TgridVec{1, 0, 0},
                TgridVec{-1, 0, 0},
                TgridVec{0, 1, 0},
                TgridVec{0, -1, 0},
                TgridVec{0, 0, 1},
                TgridVec{0, 0, -1}};
            bool offsets_ok = true;
            for (u32 dir = 0; dir < 6; dir++) {
                offsets_ok
                    = offsets_ok && sham::equals(result.offset_check[dir], unit_offsets[dir]);
            }

            bool use_i32 = (block_count > 0) && (out_of_range.get_val_at_idx(0) == 0) && offsets_ok
                           && (tot_count <= AMRBlockFinderI32::node_id_mask);

            if (use_i32) {
                AMRBlockFinderI32 finder{
                    node_rec,
                    obj_rec,
                    block_min.get_buf(),
                    block_max.get_buf(),
                    internal_cell_count,
                    origin};

                // [i] is the number of link for block i (last value is 0)
                std::array<std::unique_ptr<sham::DeviceBuffer<u32>>, 6> link_counts;
                for (u32 dir = 0; dir < 6; dir++) {
                    link_counts[dir]
                        = std::make_unique<sham::DeviceBuffer<u32>>(block_count + 1, dev_sched);
                }

                {
                    sham::EventList deps;
                    auto ker = finder.get_read_access(deps);
                    std::array<u32 *, 6> cnt;
                    for (u32 dir = 0; dir < 6; dir++) {
                        cnt[dir] = link_counts[dir]->get_write_access(deps);
                    }

                    auto e = q.submit(deps, [&](sycl::handler &cgh) {
                        sycl::local_accessor<u32, 1> stack_local(
                            block_finder_shared_stack_depth * block_finder_group_size, cgh);

                        cgh.parallel_for(
                            sham::make_ndrange(block_finder_group_size, block_count),
                            [=](sycl::nd_item<1> item) {
                                u32 id_a = (u32) item.get_global_linear_id();
                                if (id_a >= block_count) {
                                    return;
                                }

                                u32 *sh_stack = &stack_local[item.get_local_linear_id()];
                                u32 sh_stride = (u32) item.get_local_range(0);

                                std::array<u32, 6> found{0, 0, 0, 0, 0, 0};

                                ker.for_each_neigh_6dir(
                                    id_a, sh_stack, sh_stride, [&](u32 dirs, u32 id_b) {
#pragma unroll
                                        for (u32 dir = 0; dir < 6; dir++) {
                                            found[dir] += (dirs >> dir) & 1;
                                        }
                                    });

#pragma unroll
                                for (u32 dir = 0; dir < 6; dir++) {
                                    cnt[dir][id_a] = found[dir];
                                }
                            });
                    });

                    finder.complete_event_state(e);
                    for (u32 dir = 0; dir < 6; dir++) {
                        link_counts[dir]->complete_event_state(e);
                    }
                }

                std::array<std::unique_ptr<sham::DeviceBuffer<u32>>, 6> link_offsets;
                std::array<std::unique_ptr<sham::DeviceBuffer<u32>>, 6> links;
                std::array<u32, 6> link_cnt;
                for (u32 dir = 0; dir < 6; dir++) {
                    // set the last val to 0 so that the last slot after exclusive scan is the sum
                    link_counts[dir]->set_val_at_idx(block_count, 0);

                    link_offsets[dir] = std::make_unique<sham::DeviceBuffer<u32>>(
                        shamalgs::numeric::scan_exclusive(
                            dev_sched, *link_counts[dir], block_count + 1));

                    link_cnt[dir] = link_offsets[dir]->get_val_at_idx(block_count);
                    links[dir]
                        = std::make_unique<sham::DeviceBuffer<u32>>(link_cnt[dir], dev_sched);
                }

                {
                    sham::EventList deps;
                    auto ker = finder.get_read_access(deps);
                    std::array<const u32 *, 6> offsets;
                    std::array<u32 *, 6> ids;
                    for (u32 dir = 0; dir < 6; dir++) {
                        offsets[dir] = link_offsets[dir]->get_read_access(deps);
                        ids[dir]     = links[dir]->get_write_access(deps);
                    }

                    auto e = q.submit(deps, [&](sycl::handler &cgh) {
                        sycl::local_accessor<u32, 1> stack_local(
                            block_finder_shared_stack_depth * block_finder_group_size, cgh);

                        cgh.parallel_for(
                            sham::make_ndrange(block_finder_group_size, block_count),
                            [=](sycl::nd_item<1> item) {
                                u32 id_a = (u32) item.get_global_linear_id();
                                if (id_a >= block_count) {
                                    return;
                                }

                                u32 *sh_stack = &stack_local[item.get_local_linear_id()];
                                u32 sh_stride = (u32) item.get_local_range(0);

                                std::array<u32, 6> next_link_idx;
#pragma unroll
                                for (u32 dir = 0; dir < 6; dir++) {
                                    next_link_idx[dir] = offsets[dir][id_a];
                                }

                                ker.for_each_neigh_6dir(
                                    id_a, sh_stack, sh_stride, [&](u32 dirs, u32 id_b) {
#pragma unroll
                                        for (u32 dir = 0; dir < 6; dir++) {
                                            if ((dirs >> dir) & 1) {
                                                ids[dir][next_link_idx[dir]] = id_b;
                                                next_link_idx[dir]++;
                                            }
                                        }
                                    });
                            });
                    });

                    finder.complete_event_state(e);
                    for (u32 dir = 0; dir < 6; dir++) {
                        link_offsets[dir]->complete_event_state(e);
                        links[dir]->complete_event_state(e);
                    }
                }

                for (u32 dir = 0; dir < 6; dir++) {
                    shamlog_debug_ln(
                        "AMR Block Graph",
                        "Patch",
                        id,
                        "direction",
                        dir,
                        "link cnt",
                        link_cnt[dir]);

                    result.graph_links[dir] = std::make_unique<AMRGraph>(AMRGraph{
                        .node_link_offset = std::move(*link_offsets[dir]),
                        .node_links       = std::move(*links[dir]),
                        .link_count       = link_cnt[dir],
                        .obj_cnt          = block_count});
                }
            } else {
                sycl::buffer<TgridVec> buf_block_min_sycl
                    = block_min.get_buf().copy_to_sycl_buffer();
                sycl::buffer<TgridVec> buf_block_max_sycl
                    = block_max.get_buf().copy_to_sycl_buffer();

                for (u32 dir = 0; dir < 6; dir++) {

                    TgridVec dir_offset = result.offset_check[dir];

                    AMRGraph rslt = details::compute_neigh_graph_deprecated<AMRBlockFinder>(
                        dev_sched,
                        block_count,
                        tree,
                        buf_block_min_sycl,
                        buf_block_max_sycl,
                        dir_offset);

                    shamlog_debug_ln(
                        "AMR Block Graph",
                        "Patch",
                        id,
                        "direction",
                        dir,
                        "link cnt",
                        rslt.link_count);

                    result.graph_links[dir] = std::make_unique<AMRGraph>(std::move(rslt));
                }
            }

            graph.add_obj(id, std::move(result));
        });

        edges.block_neigh_graph.graph = std::move(graph);

        // possible unittest
        /*
        one patch with :
        sz = 1 << 4
        base = 4
        model.make_base_grid((0,0,0),(sz,sz,sz),(base*multx,base*multy,base*multz))

        make a grid of 4^3 blocks, which when merge with interface make 6^3 blocks.
        In each direction one slab will have no links, hence the number of links should always be
        6^3 - 6^2 = 180 which we get here on all directions
        */
    }

    template<class Tvec, class TgridVec, class Tmorton>
    std::string FindBlockNeighOpt<Tvec, TgridVec, Tmorton>::_impl_get_tex() const {

        std::string sizes             = get_ro_edge_base(0).get_tex_symbol();
        std::string block_min         = get_ro_edge_base(1).get_tex_symbol();
        std::string block_max         = get_ro_edge_base(2).get_tex_symbol();
        std::string trees             = get_ro_edge_base(3).get_tex_symbol();
        std::string block_neigh_graph = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Find neighbour blocks

            \begin{align}
            {block_neigh_graph} = \text{FindBlockNeighOpt}({sizes}, {block_min}, {block_max}, {trees})
            \end{align}
        )tex";

        shambase::replace_all(tex, "{sizes}", sizes);
        shambase::replace_all(tex, "{block_min}", block_min);
        shambase::replace_all(tex, "{block_max}", block_max);
        shambase::replace_all(tex, "{trees}", trees);
        shambase::replace_all(tex, "{block_neigh_graph}", block_neigh_graph);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::FindBlockNeighOpt<f64_3, i64_3, u64>;
