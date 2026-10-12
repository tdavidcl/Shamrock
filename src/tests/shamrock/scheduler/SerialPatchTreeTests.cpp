// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/scheduler/HilbertLoadBalance.hpp"
#include "shamrock/scheduler/PatchTree.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamtest/shamtest.hpp"
#include <cstring>
#include <random>
#include <vector>

using namespace shamrock::scheduler;
using namespace shamrock::patch;

namespace {

    constexpr u64 max_coord = HilbertLoadBalance<u64>::max_box_sz;

    /// Result of the reference serial tree build
    template<class Tvec>
    struct RefSerialTree {
        std::vector<SerialPatchNode<Tvec>> nodes;
        std::vector<u64> linked_patch_ids;
        std::vector<u64> roots_ids;
        u32 level_count = 0;
    };

    /// Reference: the recursive serial tree build SerialPatchTree used before it was made
    /// iterative
    template<class Tvec>
    void ref_get_serial_tree(
        u64 root_key,
        PatchTree &ptree,
        RefSerialTree<Tvec> &res,
        u64 &counter,
        const PatchCoordTransform<Tvec> box_transform) {

        PatchTree::Node ptn = ptree.tree[root_key];

        SerialPatchNode<Tvec> n = ptn.convert(box_transform);

        res.level_count = sycl::max((u32) ptn.tree_node.level, res.level_count);

        u64 id_nvec = counter;
        res.nodes.push_back(n);
        res.linked_patch_ids.push_back(ptn.linked_patchid);
        counter++;

        if (!ptn.is_leaf()) {
            for (u32 i = 0; i < 8; i++) {
                res.nodes.at(id_nvec).childs_id[i] = counter;
                ref_get_serial_tree<Tvec>(ptn.get_child_nid(i), ptree, res, counter, box_transform);
            }
        }
    }

    template<class Tvec>
    RefSerialTree<Tvec> ref_build(PatchTree &ptree, const PatchCoordTransform<Tvec> box_transform) {
        RefSerialTree<Tvec> res;
        u64 cnt = 0;
        for (u64 root_id : ptree.roots_id) {
            res.roots_ids.push_back(cnt);
            ref_get_serial_tree<Tvec>(root_id, ptree, res, cnt, box_transform);
        }
        return res;
    }

    template<class Tvec>
    bool bit_equal(const Tvec &a, const Tvec &b) {
        using T = typename sham::VectorProperties<Tvec>::component_type;
        for (u32 i = 0; i < 3; i++) {
            T ca = a[i];
            T cb = b[i];
            if (std::memcmp(&ca, &cb, sizeof(T)) != 0) {
                return false;
            }
        }
        return true;
    }

    template<class Tvec>
    bool same_nodes(
        const std::vector<SerialPatchNode<Tvec>> &a, const std::vector<SerialPatchNode<Tvec>> &b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (u64 i = 0; i < a.size(); i++) {
            if (!bit_equal(a[i].box_min, b[i].box_min) || !bit_equal(a[i].box_max, b[i].box_max)
                || a[i].childs_id != b[i].childs_id) {
                return false;
            }
        }
        return true;
    }

    template<class Tvec>
    void check_against_ref(PatchTree &ptree, const PatchCoordTransform<Tvec> box_transform) {
        RefSerialTree<Tvec> ref = ref_build(ptree, box_transform);
        SerialPatchTree<Tvec> sptree(ptree, box_transform);

        REQUIRE_EQUAL(sptree.get_element_count(), ref.nodes.size());
        REQUIRE_EQUAL(ref.nodes.size(), ptree.tree.size());
        REQUIRE_NAMED("serial tree nodes", same_nodes(sptree.get_nodes(), ref.nodes));
        REQUIRE_NAMED("linked patch ids", sptree.get_linked_patch_ids() == ref.linked_patch_ids);
        REQUIRE_NAMED("roots ids", sptree.get_roots_ids() == ref.roots_ids);
        REQUIRE_EQUAL(sptree.root_count, ref.roots_ids.size());
        REQUIRE_EQUAL(sptree.get_level_count(), ref.level_count);
    }

    void check_all_types(PatchTree &ptree) {
        check_against_ref<f64_3>(
            ptree,
            PatchCoordTransform<f64_3>(
                {{0, 0, 0}, {max_coord + 1, max_coord + 1, max_coord + 1}},
                {{-0.6, -0.6, -0.6}, {0.6, 0.6, 0.6}}));
        check_against_ref<f32_3>(
            ptree,
            PatchCoordTransform<f32_3>(
                {{0, 0, 0}, {max_coord + 1, max_coord + 1, max_coord + 1}},
                {{-1.f, -2.f, -3.f}, {1.f, 2.f, 3.f}}));

        i64 half = max_coord + 1;
        check_against_ref<i64_3>(
            ptree,
            PatchCoordTransform<i64_3>(
                {{0, 0, 0}, {max_coord + 1, max_coord + 1, max_coord + 1}},
                {{-half, -half, -half}, {half, half, half}}));
    }

    /// Give every leaf a patch id and every internal node u64_max
    void link_leafs(PatchTree &ptree) {
        u64 pid = 0;
        for (auto &[key, node] : ptree.tree) {
            node.linked_patchid = (node.tree_node.is_leaf) ? pid++ : u64_max;
        }
    }

} // namespace

NEW_TEST(Unittest, "shamrock/scheduler/SerialPatchTree:uniform", 1) {
    for (u32 level = 1; level <= 4; level++) {
        PatchTree ptree;
        ptree.insert_root_node(0, PatchCoord<3>({0, 0, 0}, {max_coord, max_coord, max_coord}));
        for (u32 l = 0; l < level; l++) {
            std::vector<u64> leafs(ptree.leaf_key.begin(), ptree.leaf_key.end());
            for (u64 id : leafs) {
                ptree.split_node(id);
            }
        }
        link_leafs(ptree);
        check_all_types(ptree);
    }
}

NEW_TEST(Unittest, "shamrock/scheduler/SerialPatchTree:irregular", 1) {
    // Several roots on a 4x2x3 grid (as PatchScheduler::make_patch_base_grid), then random
    // partial splits and some merges
    std::array<u32, 3> root_count = {4, 2, 3};
    u64 sz_root                   = (max_coord + 1) / 4;

    PatchTree ptree;
    u32 pid = 0;
    for (u32 x = 0; x < root_count[0]; x++) {
        for (u32 y = 0; y < root_count[1]; y++) {
            for (u32 z = 0; z < root_count[2]; z++) {
                ptree.insert_root_node(
                    pid++,
                    PatchCoord<3>(
                        {sz_root * x, sz_root * y, sz_root * z},
                        {sz_root * (x + 1) - 1, sz_root * (y + 1) - 1, sz_root * (z + 1) - 1}));
            }
        }
    }

    std::mt19937 eng(0x5e7a1);
    for (u32 round = 0; round < 4; round++) {
        std::vector<u64> leafs(ptree.leaf_key.begin(), ptree.leaf_key.end());
        std::sort(leafs.begin(), leafs.end());
        for (u64 id : leafs) {
            if (std::uniform_real_distribution<f64>(0, 1)(eng) < 0.4) {
                ptree.split_node(id);
            }
        }
    }

    std::vector<u64> mergeable(
        ptree.parent_of_only_leaf_key.begin(), ptree.parent_of_only_leaf_key.end());
    std::sort(mergeable.begin(), mergeable.end());
    for (u64 i = 0; i < mergeable.size(); i += 3) {
        ptree.merge_node_dm1(mergeable[i]);
    }

    link_leafs(ptree);

    REQUIRE(ptree.roots_id.size() == 24);
    check_all_types(ptree);
}
