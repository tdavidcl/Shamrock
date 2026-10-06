// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BlockNeighToCellNeighOpt.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Optimised variant of BlockNeighToCellNeigh (NeighGraphStrategy::NeighGraphOpt), builds the
 * same cell graph
 */

#include "shambase/exception.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shambackends/EventList.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/modules/BlockNeighToCellNeighOpt.hpp"
#include "shammodels/ramses/modules/details/compute_neigh_graph.hpp"
#include "shammodels/ramses/modules/details/neigh_graph_6dir.hpp"
#include "shamrock/amr/AMRCell.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace {

    /**
     * @brief Call fct on the cells sharing a face with cell id_a in the direction dir_offset
     *
     * Same as AMRLowering::ro_acces::for_each_other_index_safe: candidates are the cells of the
     * block of id_a, then of the blocks linked to it in the block graph of that direction, in
     * link order.
     */
    template<class AMRBlock, class TgridVec, class IndexFunctor>
    inline void for_each_cell_neigh_safe(
        u32 id_a,
        const shammodels::basegodunov::modules::AMRGraph::ro_access &graph_iter,
        const TgridVec *acc_block_min,
        const TgridVec *acc_block_max,
        TgridVec dir_offset,
        IndexFunctor &&fct) {

        const u32 cell_global_id = (u32) id_a;

        const u32 block_id    = cell_global_id / AMRBlock::block_size;
        const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

        // fetch current block info
        const TgridVec cblock_min = acc_block_min[block_id];
        const TgridVec cblock_max = acc_block_max[block_id];
        const TgridVec delta_cell = (cblock_max - cblock_min) / AMRBlock::Nside;

        std::array<u32, 3> lcoord_arr = AMRBlock::get_coord(cell_loc_id);
        TgridVec lcoord               = {lcoord_arr[0], lcoord_arr[1], lcoord_arr[2]};

        shammath::AABB<TgridVec> current_cell_aabb
            = {cblock_min + lcoord * delta_cell,
               cblock_min + (lcoord + TgridVec{1, 1, 1}) * delta_cell};

        const shammath::AABB<TgridVec> current_cell_aabb_shifted
            = {current_cell_aabb.lower + dir_offset, current_cell_aabb.upper + dir_offset};

        auto on_block = [&](u32 block_b) {
            TgridVec block_b_min = acc_block_min[block_b];
            TgridVec block_b_max = acc_block_max[block_b];

            const TgridVec delta_cell_b = (block_b_max - block_b_min) / AMRBlock::Nside;

            for (u32 lx = 0; lx < AMRBlock::Nside; lx++) {
                for (u32 ly = 0; ly < AMRBlock::Nside; ly++) {
                    for (u32 lz = 0; lz < AMRBlock::Nside; lz++) {

                        shammath::AABB<TgridVec> found_cell
                            = {TgridVec{block_b_min + TgridVec{lx, ly, lz} * delta_cell_b},
                               TgridVec{
                                   block_b_min + TgridVec{lx + 1, ly + 1, lz + 1} * delta_cell_b}};

                        u32 idx
                            = block_b * AMRBlock::block_size + AMRBlock::get_index({lx, ly, lz});

                        bool overlap = found_cell.get_intersect(current_cell_aabb_shifted)
                                           .is_volume_not_null()
                                       && id_a != idx;

                        if (overlap) {
                            fct(idx);
                        }
                    }
                }
            }
        };

        on_block(block_id);
        graph_iter.for_each_object_link(block_id, [&](u32 block_b) {
            on_block(block_b);
        });
    }

} // namespace

namespace shammodels::basegodunov::modules {

    // here if we want to find the stencil instead of just the common face, we can just the change
    // lowering mask to the cube offseted in the wanted direction instead of just checking if we
    // have a common surface

    // like on the above case with block Nside 2 we get 12^3 - 12^2 = 1584 link count
    // it is a really good possible test

    template<class Tvec, class TgridVec, class Tmorton>
    void BlockNeighToCellNeighOpt<Tvec, TgridVec, Tmorton>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        edges.spans_block_min.check_sizes(edges.sizes.indexes);
        edges.spans_block_max.check_sizes(edges.sizes.indexes);

        using AMRBlock = amr::AMRBlock<Tvec, TgridVec, 1>;

        if (block_nside_pow != 1) {
            shambase::throw_unimplemented("block_nside_pow != 1");
        }

        shambase::DistributedData<OrientedAMRGraph> cell_graph_links;

        edges.block_neigh_graph.graph.for_each([&](u64 id,
                                                   const OrientedAMRGraph &oriented_block_graph) {
            OrientedAMRGraph result;

            PatchDataField<TgridVec> &block_min = edges.spans_block_min.get_refs().get(id);
            PatchDataField<TgridVec> &block_max = edges.spans_block_max.get_refs().get(id);

            sham::DeviceBuffer<TgridVec> &buf_block_min = block_min.get_buf();
            sham::DeviceBuffer<TgridVec> &buf_block_max = block_max.get_buf();

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
            auto &q        = dev_sched->get_queue();

            u32 cell_count = (edges.sizes.indexes.get(id)) * AMRBlock::block_size;

            std::array<AMRGraph *, 6> block_graph;
            for (u32 dir = 0; dir < 6; dir++) {
                block_graph[dir] = &shambase::get_check_ref(oriented_block_graph.graph_links[dir]);
            }

            const std::array<TgridVec, 6> &off = result.offset_check;

            // The 6 directions are handled by the same thread one after the other, each in the
            // order of for_each_other_index_safe, so each link list is unchanged.

            // link counts of the 6 directions in one buffer (see details::neigh_6dir_count_idx)
            sham::DeviceBuffer<u32> link_counts(
                details::neigh_6dir_count_size(cell_count), dev_sched);

            if (cell_count > 0) {
                sham::EventList deps;
                auto g0              = block_graph[0]->get_read_access(deps);
                auto g1              = block_graph[1]->get_read_access(deps);
                auto g2              = block_graph[2]->get_read_access(deps);
                auto g3              = block_graph[3]->get_read_access(deps);
                auto g4              = block_graph[4]->get_read_access(deps);
                auto g5              = block_graph[5]->get_read_access(deps);
                const TgridVec *bmin = buf_block_min.get_read_access(deps);
                const TgridVec *bmax = buf_block_max.get_read_access(deps);
                u32 *cnt             = link_counts.get_write_access(deps);

                auto e = q.submit(deps, [&, off, cell_count](sycl::handler &cgh) {
                    shambase::parallel_for(
                        cgh, cell_count, "count cell graph links (6 dirs)", [=](u64 gid) {
                            u32 id_a = (u32) gid;

                            u32 c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0;

                            for_each_cell_neigh_safe<AMRBlock>(
                                id_a, g0, bmin, bmax, off[0], [&](u32) {
                                    c0++;
                                });
                            for_each_cell_neigh_safe<AMRBlock>(
                                id_a, g1, bmin, bmax, off[1], [&](u32) {
                                    c1++;
                                });
                            for_each_cell_neigh_safe<AMRBlock>(
                                id_a, g2, bmin, bmax, off[2], [&](u32) {
                                    c2++;
                                });
                            for_each_cell_neigh_safe<AMRBlock>(
                                id_a, g3, bmin, bmax, off[3], [&](u32) {
                                    c3++;
                                });
                            for_each_cell_neigh_safe<AMRBlock>(
                                id_a, g4, bmin, bmax, off[4], [&](u32) {
                                    c4++;
                                });
                            for_each_cell_neigh_safe<AMRBlock>(
                                id_a, g5, bmin, bmax, off[5], [&](u32) {
                                    c5++;
                                });

                            cnt[details::neigh_6dir_count_idx(0, id_a, cell_count)] = c0;
                            cnt[details::neigh_6dir_count_idx(1, id_a, cell_count)] = c1;
                            cnt[details::neigh_6dir_count_idx(2, id_a, cell_count)] = c2;
                            cnt[details::neigh_6dir_count_idx(3, id_a, cell_count)] = c3;
                            cnt[details::neigh_6dir_count_idx(4, id_a, cell_count)] = c4;
                            cnt[details::neigh_6dir_count_idx(5, id_a, cell_count)] = c5;
                        });
                });

                for (u32 dir = 0; dir < 6; dir++) {
                    block_graph[dir]->complete_event_state(e);
                }
                buf_block_min.complete_event_state(e);
                buf_block_max.complete_event_state(e);
                link_counts.complete_event_state(e);
            }

            details::NeighGraph6DirOffsets scanned
                = details::scan_link_counts_6dir(dev_sched, link_counts, cell_count);

            std::array<std::unique_ptr<sham::DeviceBuffer<u32>>, 6> links;
            for (u32 dir = 0; dir < 6; dir++) {
                links[dir]
                    = std::make_unique<sham::DeviceBuffer<u32>>(scanned.link_count[dir], dev_sched);
            }

            if (cell_count > 0) {
                sham::EventList deps;
                auto g0              = block_graph[0]->get_read_access(deps);
                auto g1              = block_graph[1]->get_read_access(deps);
                auto g2              = block_graph[2]->get_read_access(deps);
                auto g3              = block_graph[3]->get_read_access(deps);
                auto g4              = block_graph[4]->get_read_access(deps);
                auto g5              = block_graph[5]->get_read_access(deps);
                const TgridVec *bmin = buf_block_min.get_read_access(deps);
                const TgridVec *bmax = buf_block_max.get_read_access(deps);
                const u32 *sc        = scanned.scanned->get_read_access(deps);
                u32 *off0            = scanned.node_link_offset[0]->get_write_access(deps);
                u32 *off1            = scanned.node_link_offset[1]->get_write_access(deps);
                u32 *off2            = scanned.node_link_offset[2]->get_write_access(deps);
                u32 *off3            = scanned.node_link_offset[3]->get_write_access(deps);
                u32 *off4            = scanned.node_link_offset[4]->get_write_access(deps);
                u32 *off5            = scanned.node_link_offset[5]->get_write_access(deps);
                u32 *ids0            = links[0]->get_write_access(deps);
                u32 *ids1            = links[1]->get_write_access(deps);
                u32 *ids2            = links[2]->get_write_access(deps);
                u32 *ids3            = links[3]->get_write_access(deps);
                u32 *ids4            = links[4]->get_write_access(deps);
                u32 *ids5            = links[5]->get_write_access(deps);

                std::array<u32, 6> start = scanned.start;

                auto e = q.submit(deps, [&, off, cell_count, start](sycl::handler &cgh) {
                    shambase::parallel_for(
                        cgh, cell_count, "get ids cell graph links (6 dirs)", [=](u64 gid) {
                            u32 id_a = (u32) gid;

                            // offsets from the single scan, stored in the graphs here
                            auto link_offset = [&](u32 dir, u32 i) {
                                return details::neigh_6dir_link_offset(
                                    sc, start[dir], dir, i, cell_count);
                            };

                            u32 w0 = link_offset(0, id_a), w1 = link_offset(1, id_a);
                            u32 w2 = link_offset(2, id_a), w3 = link_offset(3, id_a);
                            u32 w4 = link_offset(4, id_a), w5 = link_offset(5, id_a);

                            off0[id_a] = w0;
                            off1[id_a] = w1;
                            off2[id_a] = w2;
                            off3[id_a] = w3;
                            off4[id_a] = w4;
                            off5[id_a] = w5;

                            if (id_a == cell_count - 1) {
                                off0[cell_count] = link_offset(0, cell_count);
                                off1[cell_count] = link_offset(1, cell_count);
                                off2[cell_count] = link_offset(2, cell_count);
                                off3[cell_count] = link_offset(3, cell_count);
                                off4[cell_count] = link_offset(4, cell_count);
                                off5[cell_count] = link_offset(5, cell_count);
                            }

                            for_each_cell_neigh_safe<AMRBlock>(
                                id_a, g0, bmin, bmax, off[0], [&](u32 idx) {
                                    ids0[w0++] = idx;
                                });
                            for_each_cell_neigh_safe<AMRBlock>(
                                id_a, g1, bmin, bmax, off[1], [&](u32 idx) {
                                    ids1[w1++] = idx;
                                });
                            for_each_cell_neigh_safe<AMRBlock>(
                                id_a, g2, bmin, bmax, off[2], [&](u32 idx) {
                                    ids2[w2++] = idx;
                                });
                            for_each_cell_neigh_safe<AMRBlock>(
                                id_a, g3, bmin, bmax, off[3], [&](u32 idx) {
                                    ids3[w3++] = idx;
                                });
                            for_each_cell_neigh_safe<AMRBlock>(
                                id_a, g4, bmin, bmax, off[4], [&](u32 idx) {
                                    ids4[w4++] = idx;
                                });
                            for_each_cell_neigh_safe<AMRBlock>(
                                id_a, g5, bmin, bmax, off[5], [&](u32 idx) {
                                    ids5[w5++] = idx;
                                });
                        });
                });

                scanned.scanned->complete_event_state(e);
                for (u32 dir = 0; dir < 6; dir++) {
                    block_graph[dir]->complete_event_state(e);
                    scanned.node_link_offset[dir]->complete_event_state(e);
                    links[dir]->complete_event_state(e);
                }
                buf_block_min.complete_event_state(e);
                buf_block_max.complete_event_state(e);
            }

            for (u32 dir = 0; dir < 6; dir++) {
                shamlog_debug_ln(
                    "AMR Cell Graph",
                    "Patch",
                    id,
                    "direction",
                    dir,
                    "link cnt",
                    scanned.link_count[dir]);

                result.graph_links[dir] = std::make_unique<AMRGraph>(AMRGraph{
                    .node_link_offset = std::move(*scanned.node_link_offset[dir]),
                    .node_links       = std::move(*links[dir]),
                    .link_count       = scanned.link_count[dir],
                    .obj_cnt          = cell_count});
            }

            cell_graph_links.add_obj(id, std::move(result));
        });

        shamlog_debug_ln("[AMR cell graph]", "compute antecedent map");
        cell_graph_links.for_each([&](u64 id, OrientedAMRGraph &oriented_block_graph) {
            auto ptr       = shamsys::instance::get_compute_scheduler_ptr();
            u32 cell_count = (edges.sizes.indexes.get(id)) * AMRBlock::block_size;
            for (u32 dir = 0; dir < 6; dir++) {
                oriented_block_graph.graph_links[dir]->compute_antecedent(ptr);
            }
        });

        edges.cell_neigh_graph.graph = std::move(cell_graph_links);
    }

    template<class Tvec, class TgridVec, class Tmorton>
    std::string BlockNeighToCellNeighOpt<Tvec, TgridVec, Tmorton>::_impl_get_tex() const {

        std::string sizes             = get_ro_edge_base(0).get_tex_symbol();
        std::string block_min         = get_ro_edge_base(1).get_tex_symbol();
        std::string block_max         = get_ro_edge_base(2).get_tex_symbol();
        std::string block_neigh_graph = get_ro_edge_base(3).get_tex_symbol();
        std::string cell_neigh_graph  = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Find neighbour blocks

            \begin{align}
            {cell_neigh_graph} &= \text{BlockNeighToCellNeighOpt}({block_neigh_graph}, {sizes}, {block_min}, {block_max})
            \end{align}
        )tex";

        shambase::replace_all(tex, "{sizes}", sizes);
        shambase::replace_all(tex, "{block_min}", block_min);
        shambase::replace_all(tex, "{block_max}", block_max);
        shambase::replace_all(tex, "{block_neigh_graph}", block_neigh_graph);
        shambase::replace_all(tex, "{cell_neigh_graph}", cell_neigh_graph);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::BlockNeighToCellNeighOpt<f64_3, i64_3, u64>;
