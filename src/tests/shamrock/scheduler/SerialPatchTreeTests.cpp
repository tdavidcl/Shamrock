// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shamrock/scheduler/HilbertLoadBalance.hpp"
#include "shamrock/scheduler/PatchTree.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <vector>

using namespace shamrock::scheduler;
using namespace shamrock::patch;

namespace {

    constexpr u64 max_coord = HilbertLoadBalance<u64>::max_box_sz;

    /// Give every leaf a patch id and every internal node u64_max
    void link_leafs(PatchTree &ptree) {
        u64 pid = 0;
        for (auto &[key, node] : ptree.tree) {
            node.linked_patchid = (node.tree_node.is_leaf) ? pid++ : u64_max;
        }
    }

    /// Uniform octree: a single root split `level` times
    PatchTree make_uniform_tree(u32 level) {
        PatchTree ptree;
        ptree.insert_root_node(0, PatchCoord<3>({0, 0, 0}, {max_coord, max_coord, max_coord}));
        for (u32 l = 0; l < level; l++) {
            std::vector<u64> leafs(ptree.leaf_key.begin(), ptree.leaf_key.end());
            for (u64 id : leafs) {
                ptree.split_node(id);
            }
        }
        link_leafs(ptree);
        return ptree;
    }

    f64 median(std::vector<f64> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }

} // namespace

/// Median time of the SerialPatchTree<f64_3> build and of attach_buf, as done every step by the
/// solvers, on uniform octrees up to 2M patches
NEW_TEST(Benchmark, "shamrock/scheduler/SerialPatchTree:benchmark", 1) {
    constexpr u32 rep_count = 7;

    PatchCoordTransform<f64_3> transform(
        {{0, 0, 0}, {max_coord + 1, max_coord + 1, max_coord + 1}},
        {{-0.6, -0.6, -0.6}, {0.6, 0.6, 0.6}});

    std::vector<f64> patch_counts, node_counts, t_build, t_attach_buf;

    for (u32 level = 1; level <= 7; level++) {
        PatchTree ptree = make_uniform_tree(level);

        std::vector<f64> rep_build, rep_attach_buf;
        for (u32 rep = 0; rep < rep_count; rep++) {
            shambase::Timer timer_build;
            timer_build.start();
            SerialPatchTree<f64_3> sptree(ptree, transform);
            timer_build.stop();

            shambase::Timer timer_attach_buf;
            timer_attach_buf.start();
            sptree.attach_buf();
            timer_attach_buf.stop();

            rep_build.push_back(timer_build.elapsed_sec());
            rep_attach_buf.push_back(timer_attach_buf.elapsed_sec());
        }

        patch_counts.push_back(ptree.leaf_key.size());
        node_counts.push_back(ptree.tree.size());
        t_build.push_back(median(rep_build));
        t_attach_buf.push_back(median(rep_attach_buf));

        logger::info_ln(
            "ShamrockTest",
            sham::format(
                "SerialPatchTree build : patches = {:8} nodes = {:8} build = {:9.3f} ms "
                "attach_buf = {:.3f} ms",
                ptree.leaf_key.size(),
                ptree.tree.size(),
                t_build.back() * 1e3,
                t_attach_buf.back() * 1e3));
    }

    auto &dataset = shamtest::test_data().new_dataset("SerialPatchTree build");
    dataset.add_data("Npatch", patch_counts);
    dataset.add_data("Nnode", node_counts);
    dataset.add_data("t_build", t_build);
    dataset.add_data("t_attach_buf", t_attach_buf);
}
