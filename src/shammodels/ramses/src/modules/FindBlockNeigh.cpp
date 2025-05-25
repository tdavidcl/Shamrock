// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FindBlockNeigh.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/FindBlockNeigh.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shammath/AABB.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace {

    /**
     * @brief Create a neighbour graph using a class that will list the ids of the found neighbourgh
     * NeighFindKernel will list the index and that function will run it twice to generate the graph
     *
     * @tparam NeighFindKernel the neigh find kernel
     * @tparam Args arguments that will be forwarded to the kernel
     * @param q the sycl queue
     * @param graph_nodes the number of graph nodes
     * @param args arguments that will be forwarded to the kernel
     * @return shammodels::basegodunov::modules::NeighGraph the neigh graph
     */
    template<class NeighFindKernel, class... Args>
    shammodels::basegodunov::modules::NeighGraph
    compute_neigh_graph(sycl::queue &q, u32 graph_nodes, Args &&...args) {

        // [i] is the number of link for block i in mpdat (last value is 0)
        sycl::buffer<u32> link_counts(graph_nodes + 1);

        // fill buffer with number of link in the block graph
        q.submit([&](sycl::handler &cgh) {
            NeighFindKernel ker(cgh, std::forward<Args>(args)...);
            sycl::accessor link_cnt{link_counts, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, graph_nodes, "count block graph link", [=](u64 gid) {
                u32 id_a              = (u32) gid;
                u32 block_found_count = 0;

                ker.for_each_other_index(id_a, [&](u32 id_b) {
                    block_found_count++;
                });

                link_cnt[id_a] = block_found_count;
            });
        });

        // set the last val to 0 so that the last slot after exclusive scan is the sum
        shamalgs::memory::set_element<u32>(q, link_counts, graph_nodes, 0);

        sycl::buffer<u32> link_cnt_offsets
            = shamalgs::numeric::exclusive_sum(q, link_counts, graph_nodes + 1);

        u32 link_cnt = shamalgs::memory::extract_element(q, link_cnt_offsets, graph_nodes);

        sycl::buffer<u32> ids_links(link_cnt);

        // find the neigh ids
        q.submit([&](sycl::handler &cgh) {
            NeighFindKernel ker(cgh, std::forward<Args>(args)...);
            sycl::accessor cnt_offsets{link_cnt_offsets, cgh, sycl::read_only};
            sycl::accessor links{ids_links, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, graph_nodes, "get ids block graph link", [=](u64 gid) {
                u32 id_a = (u32) gid;

                u32 next_link_idx = cnt_offsets[id_a];

                ker.for_each_other_index(id_a, [&](u32 id_b) {
                    links[next_link_idx] = id_b;
                    next_link_idx++;
                });
            });
        });

        using Graph = shammodels::basegodunov::modules::NeighGraph;
        return Graph(
            Graph{std::move(link_cnt_offsets), std::move(ids_links), link_cnt, graph_nodes});
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec, class Tmorton>
    class FindBlockNeigh<Tvec, TgridVec, Tmorton>::AMRBlockFinder {
        public:
        shamrock::tree::ObjectIterator<Tmorton, TgridVec> block_looper;

        sycl::accessor<TgridVec, 1, sycl::access::mode::read, sycl::target::device> acc_block_min;
        sycl::accessor<TgridVec, 1, sycl::access::mode::read, sycl::target::device> acc_block_max;

        TgridVec dir_offset;

        AMRBlockFinder(
            sycl::handler &cgh,
            const RTree &tree,
            sycl::buffer<TgridVec> &buf_block_min,
            sycl::buffer<TgridVec> &buf_block_max,
            TgridVec dir_offset)
            : block_looper(tree, cgh), acc_block_min{buf_block_min, cgh, sycl::read_only},
              acc_block_max{buf_block_max, cgh, sycl::read_only},
              dir_offset(std::move(dir_offset)) {}

        template<class IndexFunctor>
        void for_each_other_index(u32 id_a, IndexFunctor &&fct) const {

            // current block AABB
            shammath::AABB<TgridVec> block_aabb{acc_block_min[id_a], acc_block_max[id_a]};

            // The wanted AABB (the block we look for)
            shammath::AABB<TgridVec> check_aabb{
                block_aabb.lower + dir_offset, block_aabb.upper + dir_offset};

            block_looper.rtree_for(
                [&](u32 node_id, TgridVec bmin, TgridVec bmax) -> bool {
                    return shammath::AABB<TgridVec>{bmin, bmax}
                        .get_intersect(check_aabb)
                        .is_volume_not_null();
                },
                [&](u32 id_b) {
                    bool interact
                        = shammath::AABB<TgridVec>{acc_block_min[id_b], acc_block_max[id_b]}
                              .get_intersect(check_aabb)
                              .is_volume_not_null()
                          && id_b != id_a;

                    if (interact) {
                        fct(id_b);
                    }
                });
        }
    };

    template<class Tvec, class TgridVec, class Tmorton>
    void FindBlockNeigh<Tvec, TgridVec, Tmorton>::_impl_evaluate_internal() {
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

                AMRGraph rslt = compute_neigh_graph<AMRBlockFinder>(
                    q.q,
                    edges.sizes.indexes.get(id),
                    tree,
                    buf_block_min_sycl,
                    buf_block_max_sycl,
                    dir_offset);

                logger::debug_ln(
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
    std::string FindBlockNeigh<Tvec, TgridVec, Tmorton>::_impl_get_tex() {
        return "TODO";
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::FindBlockNeigh<f64_3, i64_3, u64>;
