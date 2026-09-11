// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file compute_neigh_graph.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/details/numeric/numeric.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/EventList.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/TreeTraversal.hpp"
#include <vector>

namespace shammodels::basegodunov::modules::details {

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
    shammodels::basegodunov::modules::NeighGraph compute_neigh_graph(
        const sham::DeviceScheduler_ptr &dev_sched, u32 graph_nodes, Args &&...args) {

        auto &q = dev_sched->get_queue();

        NeighFindKernel kergen(std::forward<Args>(args)...);

        // [i] is the number of link for block i in mpdat (last value is 0)
        sham::DeviceBuffer<u32> link_counts(graph_nodes + 1, dev_sched);

        sham::EventList deps;
        auto ker          = kergen.get_read_access(deps);
        auto ptr_link_cnt = link_counts.get_write_access(deps);

        // fill buffer with number of link in the block graph
        auto e = q.submit(deps, [&](sycl::handler &cgh) {
            shambase::parallel_for(cgh, graph_nodes, "count block graph link", [=](u64 gid) {
                u32 id_a              = (u32) gid;
                u32 block_found_count = 0;

                ker.for_each_other_index(id_a, [&](u32 id_b) {
                    block_found_count++;
                });

                ptr_link_cnt[id_a] = block_found_count;
            });
        });

        link_counts.complete_event_state(e);
        kergen.complete_event_state(e);

        // set the last val to 0 so that the last slot after exclusive scan is the sum
        link_counts.set_val_at_idx(graph_nodes, 0);

        sham::DeviceBuffer<u32> link_cnt_offsets
            = shamalgs::numeric::scan_exclusive(dev_sched, link_counts, graph_nodes + 1);

        u32 link_cnt = link_cnt_offsets.get_val_at_idx(graph_nodes);

        sham::DeviceBuffer<u32> ids_links(link_cnt, dev_sched);

        sham::EventList deps2;
        auto cnt_offsets = link_cnt_offsets.get_read_access(deps2);
        auto ker2        = kergen.get_read_access(deps);
        auto links       = ids_links.get_write_access(deps2);

        // find the neigh ids
        auto e2 = q.submit(deps2, [&](sycl::handler &cgh) {
            shambase::parallel_for(cgh, graph_nodes, "get ids block graph link", [=](u64 gid) {
                u32 id_a = (u32) gid;

                u32 next_link_idx = cnt_offsets[id_a];

                ker2.for_each_other_index(id_a, [&](u32 id_b) {
                    links[next_link_idx] = id_b;
                    next_link_idx++;
                });
            });
        });

        link_cnt_offsets.complete_event_state(e2);
        ids_links.complete_event_state(e2);
        kergen.complete_event_state(e2);

        using Graph = shammodels::basegodunov::modules::NeighGraph;
        return Graph(
            Graph{
                .node_link_offset = std::move(link_cnt_offsets),
                .node_links       = std::move(ids_links),
                .link_count       = link_cnt,
                .obj_cnt          = graph_nodes});
    };

    /**
     * @brief Create a neighbour graph using a class that will list the ids of the found neighbourgh
     * NeighFindKernel will list the index and that function will run it twice to generate the graph
     *
     * Unlike compute_neigh_graph, NeighFindKernel is constructed with the sycl::handler of each
     * kernel submission (instead of once up front), so that it can hold sycl::accessor members
     * bound to a tree that is not yet USM. NeighFindKernel must therefore only take raw
     * pointers/values in its constructor (no USM buffer access of its own): if it also reads a USM
     * buffer, the caller must acquire that access (get_read_access) before calling this function,
     * pass the resulting pointer through args, pass the resulting EventList as extra_deps, and
     * complete that access (complete_event_state) after this function returns, using the events
     * appended to out_events (see FindBlockNeigh.cpp for an example).
     *
     * // TODO remove it when the tree will finally be USM
     *
     * @tparam NeighFindKernel the neigh find kernel
     * @tparam Args arguments that will be forwarded to the kernel
     * @param q the sycl queue
     * @param graph_nodes the number of graph nodes
     * @param extra_deps events that both kernel submissions must wait on, in addition to their own
     * internal dependencies (e.g. read access to USM buffers used by NeighFindKernel)
     * @param out_events the completion event of each kernel submission is appended here, so the
     * caller can complete the event state of any USM buffer access it acquired for extra_deps
     * @param args arguments that will be forwarded to the kernel
     * @return shammodels::basegodunov::modules::NeighGraph the neigh graph
     */
    template<class NeighFindKernel, class... Args>
    shammodels::basegodunov::modules::NeighGraph compute_neigh_graph_deprecated(
        const sham::DeviceScheduler_ptr &dev_sched,
        u32 graph_nodes,
        sham::EventList &extra_deps,
        std::vector<sycl::event> &out_events,
        Args &&...args) {

        auto &q = dev_sched->get_queue();

        // [i] is the number of link for block i in mpdat (last value is 0)
        sham::DeviceBuffer<u32> link_counts(graph_nodes + 1, dev_sched);

        sham::EventList deps;
        auto ptr_link_cnt = link_counts.get_write_access(deps);

        // fill buffer with number of link in the block graph
        auto e = q.submit(deps, [&](sycl::handler &cgh) {
            extra_deps.apply_dependency(cgh);
            NeighFindKernel ker(cgh, std::forward<Args>(args)...);
            shambase::parallel_for(cgh, graph_nodes, "count block graph link", [=](u64 gid) {
                u32 id_a              = (u32) gid;
                u32 block_found_count = 0;

                ker.for_each_other_index(id_a, [&](u32 id_b) {
                    block_found_count++;
                });

                ptr_link_cnt[id_a] = block_found_count;
            });
        });

        link_counts.complete_event_state(e);
        out_events.push_back(e);

        // set the last val to 0 so that the last slot after exclusive scan is the sum
        link_counts.set_val_at_idx(graph_nodes, 0);

        sham::DeviceBuffer<u32> link_cnt_offsets
            = shamalgs::numeric::scan_exclusive(dev_sched, link_counts, graph_nodes + 1);

        u32 link_cnt = link_cnt_offsets.get_val_at_idx(graph_nodes);

        sham::DeviceBuffer<u32> ids_links(link_cnt, dev_sched);

        sham::EventList deps2;
        auto cnt_offsets = link_cnt_offsets.get_read_access(deps2);
        auto links       = ids_links.get_write_access(deps2);

        // find the neigh ids
        auto e2 = q.submit(deps2, [&](sycl::handler &cgh) {
            extra_deps.apply_dependency(cgh);
            NeighFindKernel ker(cgh, std::forward<Args>(args)...);
            shambase::parallel_for(cgh, graph_nodes, "get ids block graph link", [=](u64 gid) {
                u32 id_a = (u32) gid;

                u32 next_link_idx = cnt_offsets[id_a];

                ker.for_each_other_index(id_a, [&](u32 id_b) {
                    links[next_link_idx] = id_b;
                    next_link_idx++;
                });
            });
        });

        link_cnt_offsets.complete_event_state(e2);
        ids_links.complete_event_state(e2);
        out_events.push_back(e2);

        using Graph = shammodels::basegodunov::modules::NeighGraph;
        return Graph(
            Graph{
                .node_link_offset = std::move(link_cnt_offsets),
                .node_links       = std::move(ids_links),
                .link_count       = link_cnt,
                .obj_cnt          = graph_nodes});
    };

} // namespace shammodels::basegodunov::modules::details
