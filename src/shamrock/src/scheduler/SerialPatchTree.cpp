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

#include "shambase/exception.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamsys/legacy/log.hpp"
#include <algorithm>
#include <utility>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace {

    /// Below this node count the parallel build does not pay for its extra pass
    constexpr u64 parallel_build_min_nodes = 1 << 17;

    /// Number of subtrees per thread in the parallel build, for load balancing
    constexpr u64 parallel_build_tasks_per_thread = 64;

    u32 get_thread_count() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

} // namespace

template<class Tvec>
void SerialPatchTree<Tvec>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<Tvec> box_transform) {

    serial_tree.reserve(ptree.tree.size());
    linked_patch_ids.reserve(ptree.tree.size());

    u32 thread_count = get_thread_count();
    if (thread_count > 1 && ptree.tree.size() >= parallel_build_min_nodes) {
        build_from_patch_tree_parallel(ptree, box_transform, thread_count);
    } else {
        build_from_patch_tree_serial(ptree, box_transform);
    }

    shamlog_debug_ln("Serial Patch Tree", "tree cell count = ", serial_tree.size());
    shamlog_debug_ln("Serial Patch Tree", "level_count =", level_count);
}

template<class Tvec>
void SerialPatchTree<Tvec>::build_from_patch_tree_serial(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<Tvec> box_transform) {

    using Node = PatchTree::Node;

    // A node still to be visited, and the slot of its parent's childs_id that will receive its
    // serial index (no parent for a root)
    struct PendingNode {
        Node *node;
        u64 parent_idx;
        u32 child_slot;
    };

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
}

/*
 * Same layout as the serial build, the pre-order of the tree, built in 4 steps:
 * 1. walk the top levels of the tree serially, cutting it into a pre-ordered list of items, each
 *    either a single node of the top levels or a whole subtree below the cut;
 * 2. in parallel, list the nodes of each subtree in pre-order, which gives its size;
 * 3. the offset of each item in the serial tree is the exclusive scan of the item sizes;
 * 4. in parallel, write each item at its offset, then link the top nodes to their children.
 */
template<class Tvec>
void SerialPatchTree<Tvec>::build_from_patch_tree_parallel(
    PatchTree &ptree,
    const shamrock::patch::PatchCoordTransform<Tvec> box_transform,
    u32 thread_count) {

    using Node = PatchTree::Node;

    // cut at the first depth with enough subtrees to balance the threads
    u32 cut_depth = 0;
    {
        std::vector<Node *> level_nodes;
        for (u64 root_id : ptree.roots_id) {
            level_nodes.push_back(&ptree.tree.at(root_id));
        }
        while (level_nodes.size() < parallel_build_tasks_per_thread * thread_count) {
            std::vector<Node *> next_level;
            for (Node *n : level_nodes) {
                if (!n->is_leaf()) {
                    for (u32 i = 0; i < Node::split_count; i++) {
                        next_level.push_back(&ptree.tree.at(n->get_child_nid(i)));
                    }
                }
            }
            if (next_level.empty()) {
                break;
            }
            level_nodes = std::move(next_level);
            cut_depth++;
        }
    }

    // 1. top levels, in pre-order
    struct Item {
        Node *node;
        u64 parent_item; ///< u64_max for a root
        u32 child_slot;
        bool is_subtree; ///< whole subtree below the cut, or single node above it
    };
    std::vector<Item> items;
    std::vector<u64> root_items;
    {
        struct PendingItem {
            Node *node;
            u64 parent_item;
            u32 child_slot;
            u32 depth;
        };
        std::vector<PendingItem> stack;

        for (u64 root_id : ptree.roots_id) {
            root_items.push_back(items.size());
            stack.push_back({&ptree.tree.at(root_id), u64_max, 0, 0});

            while (!stack.empty()) {
                PendingItem cur = stack.back();
                stack.pop_back();

                u64 item_id     = items.size();
                bool is_subtree = cur.depth == cut_depth && !cur.node->is_leaf();
                items.push_back({cur.node, cur.parent_item, cur.child_slot, is_subtree});

                if (!is_subtree && !cur.node->is_leaf()) {
                    for (u32 i = Node::split_count; i-- > 0;) {
                        stack.push_back(
                            {&ptree.tree.at(cur.node->get_child_nid(i)),
                             item_id,
                             i,
                             cur.depth + 1});
                    }
                }
            }
        }
    }

    // 2. pre-order node list of each subtree
    std::vector<std::vector<Node *>> subtree_nodes(items.size());
    std::vector<u64> offsets(items.size());
    u32 max_level      = 0;
    bool missing_child = false;

#pragma omp parallel for schedule(dynamic) reduction(max : max_level)
    for (u64 item_id = 0; item_id < items.size(); item_id++) {
        const Item &item = items[item_id];

        if (!item.is_subtree) {
            offsets[item_id] = 1;
            max_level        = std::max(max_level, (u32) item.node->tree_node.level);
            continue;
        }

        std::vector<Node *> &nodes = subtree_nodes[item_id];
        std::vector<Node *> stack{item.node};
        while (!stack.empty()) {
            Node *cur = stack.back();
            stack.pop_back();

            nodes.push_back(cur);
            max_level = std::max(max_level, (u32) cur->tree_node.level);

            if (!cur->is_leaf()) {
                for (u32 i = Node::split_count; i-- > 0;) {
                    // no exception may leave the parallel region, report it after
                    auto it = ptree.tree.find(cur->get_child_nid(i));
                    if (it == ptree.tree.end()) {
#pragma omp atomic write
                        missing_child = true;
                        continue;
                    }
                    stack.push_back(&it->second);
                }
            }
        }
        offsets[item_id] = nodes.size();
    }

    if (missing_child) {
        throw shambase::make_except_with_loc<std::runtime_error>(
            "a child node is missing in the patch tree");
    }

    level_count = max_level;

    // 3. offsets
    u64 node_count = 0;
    for (u64 &offset : offsets) {
        u64 size = offset;
        offset   = node_count;
        node_count += size;
    }

    for (u64 root_item : root_items) {
        root_count++;
        roots_ids.push_back(offsets[root_item]);
    }

    // 4. write, without initializing first so that each page is first touched by the thread
    // writing it
    serial_tree.resize(node_count);
    linked_patch_ids.resize(node_count);

#pragma omp parallel for schedule(dynamic)
    for (u64 item_id = 0; item_id < items.size(); item_id++) {
        const Item &item = items[item_id];
        u64 offset       = offsets[item_id];

        if (!item.is_subtree) {
            ::new (&serial_tree[offset]) PtNode(item.node->convert(box_transform));
            linked_patch_ids[offset] = item.node->linked_patchid;
            continue;
        }

        // the pre-order list is replayed with a stack of the childs_id slots still to fill
        std::vector<std::pair<u64, u32>> pending_slots;

        const std::vector<Node *> &nodes = subtree_nodes[item_id];
        for (u64 i = 0; i < nodes.size(); i++) {
            u64 idx   = offset + i;
            Node &ptn = *nodes[i];

            ::new (&serial_tree[idx]) PtNode(ptn.convert(box_transform));
            linked_patch_ids[idx] = ptn.linked_patchid;

            if (i > 0) {
                auto [parent_idx, child_slot] = pending_slots.back();
                pending_slots.pop_back();
                serial_tree[parent_idx].childs_id[child_slot] = idx;
            }

            if (!ptn.is_leaf()) {
                for (u32 c = Node::split_count; c-- > 0;) {
                    pending_slots.push_back({idx, c});
                }
            }
        }
    }

    for (u64 item_id = 0; item_id < items.size(); item_id++) {
        const Item &item = items[item_id];
        if (item.parent_item != u64_max) {
            serial_tree[offsets[item.parent_item]].childs_id[item.child_slot] = offsets[item_id];
        }
    }
}

template void SerialPatchTree<f32_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<f32_3> box_transform);
template void SerialPatchTree<f64_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<f64_3> box_transform);
template void SerialPatchTree<i64_3>::build_from_patch_tree(
    PatchTree &ptree, const shamrock::patch::PatchCoordTransform<i64_3> box_transform);
