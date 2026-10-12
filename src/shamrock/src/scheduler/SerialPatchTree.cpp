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

    // A node still to be visited, and the slot of its parent's childs_id that will receive its
    // serial index (no parent for a root)
    struct PendingNode {
        Node *node;
        u64 parent_idx;
        u32 child_slot;
    };

    serial_tree.reserve(ptree.tree.size());
    linked_patch_ids.reserve(ptree.tree.size());

    std::vector<PendingNode> stack;

    level_count = 0;

    // Iterative depth first traversal, children visited in order: the serial index of a node is
    // its rank in the pre-order traversal of the tree
    for (u64 root_id : ptree.roots_id) {
        root_count++;
        roots_ids.push_back(serial_tree.size());

        stack.push_back({&ptree.tree.at(root_id), u64_max, 0});

        while (!stack.empty()) {
            PendingNode cur = stack.back();
            stack.pop_back();

            u64 idx = serial_tree.size();
            if (cur.parent_idx != u64_max) {
                serial_tree[cur.parent_idx].childs_id[cur.child_slot] = idx;
            }

            Node &ptn = *cur.node;

            level_count = sycl::max((u32) ptn.tree_node.level, level_count);

            serial_tree.push_back(ptn.convert(box_transform));
            linked_patch_ids.push_back(ptn.linked_patchid);

            if (!ptn.is_leaf()) {
                // the 8 lookups are independent, resolving them together lets their memory
                // accesses overlap. Pushed in reverse so that child 0 is visited first.
                for (u32 i = Node::split_count; i-- > 0;) {
                    stack.push_back({&ptree.tree.at(ptn.get_child_nid(i)), idx, i});
                }
            }
        }
    }

    shamlog_debug_ln("Serial Patch Tree", "tree cell count = ", serial_tree.size());
    shamlog_debug_ln("Serial Patch Tree", "level_count =", level_count);
}

template void SerialPatchTree<f32_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<f32_3> box_transform);
template void SerialPatchTree<f64_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<f64_3> box_transform);
template void SerialPatchTree<i64_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<i64_3> box_transform);
