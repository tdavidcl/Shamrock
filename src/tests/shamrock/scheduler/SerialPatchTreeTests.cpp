// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/Timer.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace {

    using Patch     = shamrock::patch::Patch;
    using PatchTree = shamrock::scheduler::PatchTree;

    /// A patch tree with its leaves as patches (shuffled ids) and the corresponding coord transform
    template<class Tvec>
    struct TestPatchTree {
        PatchTree ptree;
        std::vector<Patch> leaves;
        shamrock::patch::PatchCoordTransform<Tvec> transform;
        shammath::CoordRange<Tvec> domain;
    };

    /**
     * @brief Make a patch tree with `root_grid` roots, all split `base_level` times, then
     * `extra_splits` splits of random leaves (so some branches are deeper)
     */
    template<class Tvec>
    TestPatchTree<Tvec> make_test_tree(
        std::mt19937 &eng,
        std::array<u32, 3> root_grid,
        u32 base_level,
        u32 extra_splits,
        shammath::CoordRange<Tvec> domain) {

        // same layout as PatchScheduler::make_patch_base_grid
        u32 max_lin = std::max({root_grid[0], root_grid[1], root_grid[2]});
        u64 sz_root = PatchScheduler::max_axis_patch_coord_length / sham::roundup_pow2_clz(max_lin);

        PatchTree ptree;
        u32 tmp_id = 0;
        for (u32 x = 0; x < root_grid[0]; x++) {
            for (u32 y = 0; y < root_grid[1]; y++) {
                for (u32 z = 0; z < root_grid[2]; z++) {
                    shamrock::patch::PatchCoord<3> coord;
                    coord.coord_min = {sz_root * x, sz_root * y, sz_root * z};
                    coord.coord_max
                        = {sz_root * (x + 1) - 1, sz_root * (y + 1) - 1, sz_root * (z + 1) - 1};
                    ptree.insert_root_node(tmp_id++, coord);
                }
            }
        }

        shamrock::patch::PatchCoord<3> bounds;
        bounds.coord_min = {0, 0, 0};
        bounds.coord_max
            = {sz_root * root_grid[0] - 1, sz_root * root_grid[1] - 1, sz_root * root_grid[2] - 1};

        for (u32 lev = 0; lev < base_level; lev++) {
            std::vector<u64> to_split(ptree.leaf_key.begin(), ptree.leaf_key.end());
            for (u64 id : to_split) {
                ptree.split_node(id);
            }
        }

        // split random leaves (possibly ones that were just created)
        std::vector<u64> leaves(ptree.leaf_key.begin(), ptree.leaf_key.end());
        std::sort(leaves.begin(), leaves.end());
        for (u32 i = 0; i < extra_splits; i++) {
            u64 pick = std::uniform_int_distribution<u64>(0, leaves.size() - 1)(eng);
            u64 id   = leaves[pick];
            ptree.split_node(id);
            leaves[pick] = leaves.back();
            leaves.pop_back();
            for (u64 c : ptree.tree[id].tree_node.childs_nid) {
                leaves.push_back(c);
            }
        }

        // shuffled patch ids, like the global list ordered by owner rank rather than by id
        std::vector<u64> leaf_nodes(ptree.leaf_key.begin(), ptree.leaf_key.end());
        std::sort(leaf_nodes.begin(), leaf_nodes.end());
        std::vector<u64> ids(leaf_nodes.size());
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), eng);

        for (auto &[nid, node] : ptree.tree) {
            node.linked_patchid = u64_max;
        }

        std::vector<Patch> patches;
        for (u64 i = 0; i < leaf_nodes.size(); i++) {
            auto &node          = ptree.tree[leaf_nodes[i]];
            node.linked_patchid = ids[i];

            Patch p{};
            p.id_patch = ids[i];
            p.override_from_coord(node.patch_coord);
            patches.push_back(p);
        }
        std::shuffle(patches.begin(), patches.end(), eng);

        return {
            std::move(ptree),
            std::move(patches),
            shamrock::patch::PatchCoordTransform<Tvec>{bounds.get_patch_range(), domain},
            domain};
    }

    template<class Tvec>
    Tvec rand_in_box(std::mt19937 &eng, Tvec lo, Tvec hi) {
        using Tscal = shambase::VecComponent<Tvec>;
        std::uniform_real_distribution<Tscal> dx(lo.x(), hi.x());
        std::uniform_real_distribution<Tscal> dy(lo.y(), hi.y());
        std::uniform_real_distribution<Tscal> dz(lo.z(), hi.z());
        return Tvec{dx(eng), dy(eng), dz(eng)};
    }

    /// Positions mostly in `box`, some on its faces, some anywhere (including outside `domain`)
    template<class Tvec>
    std::vector<Tvec> gen_positions(
        std::mt19937 &eng,
        u32 n,
        shammath::CoordRange<Tvec> box,
        shammath::CoordRange<Tvec> domain) {
        using Tscal = shambase::VecComponent<Tvec>;

        Tvec ext = (domain.upper - domain.lower) * Tscal(0.1);
        std::vector<Tvec> ret;
        std::uniform_int_distribution<u32> kind(0, 9);
        std::uniform_int_distribution<u32> axis(0, 2);
        for (u32 i = 0; i < n; i++) {
            u32 k = kind(eng);
            if (k < 6) {
                ret.push_back(rand_in_box(eng, box.lower, box.upper));
            } else if (k < 8) {
                // exactly on a lower (in) or upper (out) face of the box
                Tvec r = rand_in_box(eng, box.lower, box.upper);
                u32 a  = axis(eng);
                r[a]   = (k == 6) ? box.lower[a] : box.upper[a];
                ret.push_back(r);
            } else {
                ret.push_back(rand_in_box(eng, domain.lower - ext, domain.upper + ext));
            }
        }
        return ret;
    }

    template<class Tvec>
    bool vec_bit_equal(Tvec a, Tvec b) {
        return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
    }

    template<class T>
    std::vector<T> buf_to_vec(sycl::buffer<T> &buf, u32 len) {
        std::vector<T> ret(len);
        sycl::host_accessor acc{buf, sycl::read_only};
        for (u32 i = 0; i < len; i++) {
            ret[i] = acc[i];
        }
        return ret;
    }

    template<class Tvec>
    void test_compute_patch_owner(
        std::array<u32, 3> root_grid, u32 base_level, u32 extra_splits, u32 seed) {

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        std::mt19937 eng(seed);

        shammath::CoordRange<Tvec> domain{Tvec{-1.1, -0.7, 0.3}, Tvec{2.3, 1.9, 1.7}};

        auto [ptree, leaves, transform, dom]
            = make_test_tree<Tvec>(eng, root_grid, base_level, extra_splits, domain);

        SerialPatchTree<Tvec> sptree(ptree, transform);
        sptree.attach_buf();

        using PtNode = typename SerialPatchTree<Tvec>::PtNode;

        // the box of each patch must be bit-identical to its leaf in the serial tree
        std::unordered_map<u64, PtNode> leaf_nodes;
        sptree.host_for_each_leafs(
            [](u64, PtNode) {
                return true;
            },
            [&](u64 patch_id, PtNode n) {
                leaf_nodes[patch_id] = n;
            });

        REQUIRE_EQUAL(leaf_nodes.size(), leaves.size());

        u32 box_mismatch = 0;
        for (Patch &p : leaves) {
            auto box  = sptree.get_patch_box(p);
            PtNode &n = leaf_nodes.at(p.id_patch);
            if (!vec_bit_equal(box.lower, n.box_min) || !vec_bit_equal(box.upper, n.box_max)) {
                box_mismatch++;
            }
        }
        REQUIRE_EQUAL(box_mismatch, 0);

        // the two passes version must give exactly the result of the full tree search
        u32 n_test_patches = std::min<u32>(leaves.size(), 48);
        u32 mismatch       = 0;
        u32 bad_count      = 0;
        u32 not_kept       = 0;
        u32 total_moved    = 0;
        u32 total_invalid  = 0;

        for (u32 ip = 0; ip < n_test_patches; ip++) {
            Patch cur_p = leaves[ip];
            auto box    = sptree.get_patch_box(cur_p);

            std::vector<Tvec> pos = gen_positions(eng, 2000, box, domain);
            u32 len               = pos.size();

            sham::DeviceBuffer<Tvec> pos_buf(len, dev_sched);
            pos_buf.copy_from_stdvec(pos);

            sycl::buffer<u64> ref_buf = sptree.compute_patch_owner(dev_sched, pos_buf, len);
            auto [new_buf, moved_count]
                = sptree.compute_patch_owner(dev_sched, pos_buf, len, cur_p);

            std::vector<u64> ref = buf_to_vec(ref_buf, len);
            std::vector<u64> res = buf_to_vec(new_buf, len);

            u32 expected_moved = 0;
            for (u32 i = 0; i < len; i++) {
                bool in = Patch::is_in_patch_converted(pos[i], box.lower, box.upper);
                if (!in) {
                    expected_moved++;
                }
                if (in && res[i] != cur_p.id_patch) {
                    not_kept++;
                }
                if (ref[i] != res[i]) {
                    mismatch++;
                }
                if (res[i] == u64_max) {
                    total_invalid++;
                }
            }
            if (expected_moved != moved_count) {
                bad_count++;
            }
            total_moved += moved_count;
        }

        REQUIRE_EQUAL(mismatch, 0);
        REQUIRE_EQUAL(bad_count, 0);
        REQUIRE_EQUAL(not_kept, 0);

        // make sure the test covers both objects changing patch and objects outside the domain
        REQUIRE(total_moved > 0);
        REQUIRE(total_invalid > 0);

        sptree.detach_buf();
    }

} // namespace

NEW_TEST(Unittest, "shamrock/scheduler/SerialPatchTree::compute_patch_owner(current patch)", 1) {
    test_compute_patch_owner<f64_3>({1, 1, 1}, 2, 30, 0x111);
    test_compute_patch_owner<f32_3>({1, 1, 1}, 2, 30, 0x222);
    test_compute_patch_owner<f64_3>({2, 1, 3}, 1, 40, 0x333);
    test_compute_patch_owner<f32_3>({2, 1, 3}, 1, 40, 0x444);
}

NEW_TEST(Unittest, "shamrock/scheduler/SerialPatchTree::compute_patch_owner(no move, no tree)", 1) {

    // when no object leaves its patch the tree buffers must not be accessed: here they are not
    // even allocated, so accessing them would throw

    using Tvec     = f64_3;
    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    std::mt19937 eng(0x555);

    shammath::CoordRange<Tvec> domain{Tvec{-1, -1, -1}, Tvec{1, 1, 1}};
    auto [ptree, leaves, transform, dom] = make_test_tree<Tvec>(eng, {1, 1, 1}, 2, 10, domain);

    SerialPatchTree<Tvec> sptree(ptree, transform);

    Patch cur_p = leaves[0];
    auto box    = sptree.get_patch_box(cur_p);

    u32 len = 10000;
    std::vector<Tvec> pos;
    for (u32 i = 0; i < len; i++) {
        Tvec r = rand_in_box(eng, box.lower, box.upper);
        // uniform_real_distribution may return the upper bound due to rounding
        pos.push_back(Patch::is_in_patch_converted(r, box.lower, box.upper) ? r : box.lower);
    }
    pos[0] = box.lower; // the lower corner is in the box

    sham::DeviceBuffer<Tvec> pos_buf(len, dev_sched);
    pos_buf.copy_from_stdvec(pos);

    auto [new_buf, moved_count] = sptree.compute_patch_owner(dev_sched, pos_buf, len, cur_p);

    REQUIRE_EQUAL(moved_count, 0);

    std::vector<u64> res = buf_to_vec(new_buf, len);
    u32 not_kept         = 0;
    for (u32 i = 0; i < len; i++) {
        if (res[i] != cur_p.id_patch) {
            not_kept++;
        }
    }
    REQUIRE_EQUAL(not_kept, 0);
}

namespace {

    /**
     * @brief Time the old (full tree search) and new (current patch first) compute_patch_owner
     *
     * The serial tree buffers are recreated before each call as the solver does every step, so
     * that the cost of moving the tree to the device is included when the tree is accessed.
     */
    void bench_compute_patch_owner(
        std::string name,
        u32 base_level,
        u32 extra_splits,
        u32 npart,
        f64 moved_frac,
        u32 nrepeat) {

        using Tvec     = f64_3;
        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        auto &q        = dev_sched->get_queue();
        std::mt19937 eng(0x666);

        shammath::CoordRange<Tvec> domain{Tvec{-1, -1, -1}, Tvec{1, 1, 1}};
        auto [ptree, leaves, transform, dom]
            = make_test_tree<Tvec>(eng, {1, 1, 1}, base_level, extra_splits, domain);

        SerialPatchTree<Tvec> sptree(ptree, transform);
        u64 tree_bytes = u64(sptree.get_element_count())
                         * (sizeof(typename SerialPatchTree<Tvec>::PtNode) + sizeof(u64));

        // current patch : the largest one, others particles moved slightly outside of it
        Patch cur_p = leaves[0];
        for (Patch &p : leaves) {
            if (p.coord_max[0] - p.coord_min[0] > cur_p.coord_max[0] - cur_p.coord_min[0]) {
                cur_p = p;
            }
        }
        auto box = sptree.get_patch_box(cur_p);
        Tvec ext = (box.upper - box.lower) * 0.05;

        std::vector<Tvec> pos(npart);
        std::uniform_real_distribution<f64> u(0, 1);
        for (u32 i = 0; i < npart; i++) {
            if (u(eng) < moved_frac) {
                // just outside of the lower x face, still in the domain
                Tvec r = rand_in_box(eng, box.lower, box.upper);
                r.x()  = box.lower.x() - ext.x() * u(eng) - 1e-9;
                if (r.x() < domain.lower.x()) {
                    r.x() = box.upper.x() + ext.x() * u(eng);
                }
                pos[i] = r;
            } else {
                pos[i] = rand_in_box(eng, box.lower, box.upper);
            }
        }

        sham::DeviceBuffer<Tvec> pos_buf(npart, dev_sched);
        pos_buf.copy_from_stdvec(pos);

        auto run = [&](bool new_version) -> std::pair<f64, u32> {
            f64 tot   = 0;
            u32 moved = 0;
            for (u32 r = 0; r < nrepeat + 1; r++) {
                sptree.attach_buf();

                q.q.wait();
                shambase::Timer t;
                t.start();
                if (new_version) {
                    auto [buf, cnt] = sptree.compute_patch_owner(dev_sched, pos_buf, npart, cur_p);
                    moved           = cnt;
                    q.q.wait();
                } else {
                    sycl::buffer<u64> buf = sptree.compute_patch_owner(dev_sched, pos_buf, npart);
                    q.q.wait();
                }
                t.stop();

                sptree.detach_buf();

                if (r > 0) { // first run is warmup
                    tot += t.elapsed_sec();
                }
            }
            return {tot / nrepeat, moved};
        };

        auto [t_old, m_old] = run(false);
        auto [t_new, moved] = run(true);

        logger::raw_ln(
            shambase::format(
                "{:<28} patches={:>7} tree_nodes={:>7} tree_MB={:>7.2f} N={:>8} moved={:>6} "
                "old={:>9.3f} ms new={:>9.3f} ms speedup={:>7.1f}",
                name,
                leaves.size(),
                sptree.get_element_count(),
                f64(tree_bytes) / 1e6,
                npart,
                moved,
                t_old * 1e3,
                t_new * 1e3,
                t_old / t_new));

        auto &dat = shamtest::test_data().new_dataset(name);
        dat.add_data("t_old", std::vector<f64>{t_old});
        dat.add_data("t_new", std::vector<f64>{t_new});
        dat.add_data("moved", std::vector<f64>{f64(moved)});
        dat.add_data("patches", std::vector<f64>{f64(leaves.size())});
    }

} // namespace

NEW_TEST(Benchmark, "shamrock/scheduler/SerialPatchTree::compute_patch_owner:benchmark", 1) {

    u32 npart = 1u << 22;

    // 8^2 = 64 patches
    bench_compute_patch_owner("small tree, no move", 2, 0, npart, 0, 10);
    bench_compute_patch_owner("small tree, 0.1% move", 2, 0, npart, 1e-3, 10);

    // 8^6 + 77000 * 7 ~ 800k patches
    bench_compute_patch_owner("800k patches, no move", 6, 77000, npart, 0, 10);
    bench_compute_patch_owner("800k patches, 0.1% move", 6, 77000, npart, 1e-3, 10);
}
