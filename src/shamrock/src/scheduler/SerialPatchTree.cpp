// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SerialPatchTree.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamsys/legacy/log.hpp"

template<class vec>
void SerialPatchTree<vec>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<vec> box_transform) {

    using Node = PatchTree::Node;

    // below these sizes the OpenMP startup cost outweighs the gain (measured on the
    // SerialPatchTree benchmark: serial is faster at ~40k nodes, parallel ~3x faster at ~300k)
    constexpr u64 omp_min_lookups = 1U << 16;
    constexpr u64 omp_min_nodes   = 1U << 17;

    // Pass 1: list the nodes in breadth-first order, roots first. The 8 childs of a node are
    // appended together, so their ids are first_child_ids[i] + 0..7.
    // Done level by level since the lookups within a level are independent and can run in parallel
    std::vector<const Node *> nodes;
    std::vector<u64> first_child_ids;
    nodes.reserve(ptree.tree.size());
    first_child_ids.reserve(ptree.tree.size());

    roots_ids.reserve(ptree.roots_id.size());
    for (u64 root_id : ptree.roots_id) {
        roots_ids.push_back(nodes.size());
        nodes.push_back(&ptree.tree.at(root_id));
    }
    root_count = roots_ids.size();

    u64 level_begin = 0;
    u64 level_end   = nodes.size();
    while (level_begin < level_end) {

        // assign the ids of the childs of this level
        u64 next_id = level_end;
        first_child_ids.resize(level_end);
        for (u64 i = level_begin; i < level_end; i++) {
            bool is_leaf       = nodes[i]->tree_node.is_leaf;
            first_child_ids[i] = (is_leaf) ? u64_max : next_id;
            next_id += (is_leaf) ? 0 : 8;
        }
        nodes.resize(next_id);

        bool missing_node = false;
#pragma omp parallel for if (next_id - level_end > omp_min_lookups) reduction(|| : missing_node)
        for (u64 i = level_begin; i < level_end; i++) {
            u64 first = first_child_ids[i];
            if (first == u64_max) {
                continue;
            }
            for (u32 c = 0; c < 8; c++) {
                auto it = ptree.tree.find(nodes[i]->tree_node.childs_nid[c]);
                if (it == ptree.tree.end()) {
                    missing_node = true;
                    continue;
                }
                nodes[first + c] = &it->second;
            }
        }
        if (missing_node) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "a child node referenced in the patch tree does not exist");
        }

        level_begin = level_end;
        level_end   = next_id;
    }

    // Pass 2 (parallel): convert the nodes, this is bound by the write of the serial tree
    u64 node_count = nodes.size();
    serial_tree.resize(node_count);
    linked_patch_ids.resize(node_count);

    u32 max_level = 0;
#pragma omp parallel for if (node_count > omp_min_nodes) reduction(max : max_level)
    for (u64 i = 0; i < node_count; i++) {
        const Node &n = *nodes[i];

        PtNode pnode = n.convert(box_transform);
        u64 first    = first_child_ids[i];
        if (first != u64_max) {
            for (u32 c = 0; c < 8; c++) {
                pnode.childs_id[c] = first + c;
            }
        }

        serial_tree[i]      = pnode;
        linked_patch_ids[i] = n.linked_patchid;
        max_level           = sycl::max(max_level, n.tree_node.level);
    }
    level_count = max_level;

    shamlog_debug_ln("Serial Patch Tree", "tree cell count = ", serial_tree.size());
    shamlog_debug_ln("Serial Patch Tree", "level_count =", level_count);
}

template void SerialPatchTree<f32_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<f32_3> box_transform);
template void SerialPatchTree<f64_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<f64_3> box_transform);
template void SerialPatchTree<i64_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<i64_3> box_transform);
