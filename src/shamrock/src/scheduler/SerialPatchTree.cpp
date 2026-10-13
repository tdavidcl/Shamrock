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

using PatchTree = shamrock::scheduler::PatchTree;

template<class vec>
void get_serial_tree(
    u64 root_key,
    PatchTree &ptree,
    std::vector<shamrock::scheduler::SerialPatchNode<vec>> &result_tree,
    std::vector<u64> &result_tree_linked_patch_id,
    u64 &counter,
    u32 &max_level,
    const shamrock::patch::PatchCoordTransform<vec> box_transform) {

    // Iterative pre-order traversal (same node ordering as the recursive version):
    // each entry is a node to visit along with the slot of its parent to write its new id into
    struct TraversalItem {
        u64 key;
        u64 parent_id;
        u32 child_slot;
    };

    std::vector<TraversalItem> stack;
    stack.reserve(8 * 64);
    stack.push_back({root_key, u64_max, 0});

    while (!stack.empty()) {
        TraversalItem cur = stack.back();
        stack.pop_back();

        const PatchTree::Node &ptn = ptree.tree.at(cur.key);

        u64 id_nvec = counter;
        counter++;

        if (cur.parent_id != u64_max) {
            result_tree[cur.parent_id].childs_id[cur.child_slot] = id_nvec;
        }

        max_level = sycl::max((u32) ptn.tree_node.level, max_level);

        result_tree.push_back(ptn.convert(box_transform));
        result_tree_linked_patch_id.push_back(ptn.linked_patchid);

        if (!ptn.tree_node.is_leaf) {
            // pushed in reverse so that child 0 is visited first
            for (u32 i = 8; i-- > 0;) {
                stack.push_back({ptn.tree_node.childs_nid[i], id_nvec, i});
            }
        }
    }
}

template void get_serial_tree(
    u64 root_key,
    PatchTree &ptree,
    std::vector<shamrock::scheduler::SerialPatchNode<f32_3>> &result_tree,
    std::vector<u64> &result_tree_linked_patch_id,
    u64 &counter,
    u32 &max_level,
    const shamrock::patch::PatchCoordTransform<f32_3> box_transform);
template void get_serial_tree(
    u64 root_key,
    PatchTree &ptree,
    std::vector<shamrock::scheduler::SerialPatchNode<f64_3>> &result_tree,
    std::vector<u64> &result_tree_linked_patch_id,
    u64 &counter,
    u32 &max_level,
    const shamrock::patch::PatchCoordTransform<f64_3> box_transform);

template void get_serial_tree(
    u64 root_key,
    PatchTree &ptree,
    std::vector<shamrock::scheduler::SerialPatchNode<i64_3>> &result_tree,
    std::vector<u64> &result_tree_linked_patch_id,
    u64 &counter,
    u32 &max_level,
    const shamrock::patch::PatchCoordTransform<i64_3> box_transform);

template<>
void SerialPatchTree<f32_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<f32_3> box_transform) {

    u64 cnt     = 0;
    level_count = 0;
    for (u64 root_id : ptree.roots_id) {
        root_count++;
        roots_ids.push_back(cnt);
        shamlog_debug_ln("Serial Patch Tree", "get serial tree fp32 root id :", root_id);
        get_serial_tree<f32_3>(
            root_id, ptree, serial_tree, linked_patch_ids, cnt, level_count, box_transform);
    }

    shamlog_debug_ln("Serial Patch Tree", "tree cell count = ", serial_tree.size());
    shamlog_debug_ln("Serial Patch Tree", "level_count =", level_count);
}

template<>
void SerialPatchTree<f64_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<f64_3> box_transform) {

    u64 cnt     = 0;
    level_count = 0;
    for (u64 root_id : ptree.roots_id) {
        root_count++;
        roots_ids.push_back(cnt);
        get_serial_tree<f64_3>(
            root_id, ptree, serial_tree, linked_patch_ids, cnt, level_count, box_transform);
    }
    shamlog_debug_ln("Serial Patch Tree", "tree cell count = ", serial_tree.size());
    shamlog_debug_ln("Serial Patch Tree", "level_count =", level_count);
}

template<>
void SerialPatchTree<i64_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<i64_3> box_transform) {

    u64 cnt     = 0;
    level_count = 0;
    for (u64 root_id : ptree.roots_id) {
        root_count++;
        roots_ids.push_back(cnt);
        get_serial_tree<i64_3>(
            root_id, ptree, serial_tree, linked_patch_ids, cnt, level_count, box_transform);
    }
    shamlog_debug_ln("Serial Patch Tree", "tree cell count = ", serial_tree.size());
    shamlog_debug_ln("Serial Patch Tree", "level_count =", level_count);
}
