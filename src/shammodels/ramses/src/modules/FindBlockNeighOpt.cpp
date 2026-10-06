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
#include "shammath/AABB.hpp"
#include "shammodels/ramses/modules/FindBlockNeighOpt.hpp"
#include "shammodels/ramses/modules/details/compute_neigh_graph.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/TreeTraversal.hpp"

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
                    u32 current_node_id    = id_stack[stack_cursor];
                    id_stack[stack_cursor] = _nindex;
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

            sycl::buffer<TgridVec> buf_block_min_sycl = block_min.get_buf().copy_to_sycl_buffer();
            sycl::buffer<TgridVec> buf_block_max_sycl = block_max.get_buf().copy_to_sycl_buffer();

            for (u32 dir = 0; dir < 6; dir++) {

                TgridVec dir_offset = result.offset_check[dir];

                AMRGraph rslt = details::compute_neigh_graph_deprecated<AMRBlockFinder>(
                    shamsys::instance::get_compute_scheduler_ptr(),
                    edges.sizes.indexes.get(id),
                    tree,
                    buf_block_min_sycl,
                    buf_block_max_sycl,
                    dir_offset);

                shamlog_debug_ln(
                    "AMR Block Graph", "Patch", id, "direction", dir, "link cnt", rslt.link_count);

                std::unique_ptr<AMRGraph> tmp_graph = std::make_unique<AMRGraph>(std::move(rslt));

                result.graph_links[dir] = std::move(tmp_graph);
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
