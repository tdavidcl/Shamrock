// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/ReattributeDataUtility.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamtest/shamtest.hpp"
#include <random>
#include <vector>

namespace {

    using Tvec = f64_3;

    /// deterministic position of object `id` at step `step` in the unit box
    Tvec get_pos(u32 id, u32 step) {
        std::mt19937_64 eng(u64(id) * 0x9E3779B97F4A7C15ULL + step);
        std::uniform_real_distribution<f64> distf(0, 1);
        f64 x = distf(eng);
        f64 y = distf(eng);
        f64 z = distf(eng);
        return {x, y, z};
    }

    /// set the position of every object according to get_pos(id, step)
    void set_positions(PatchScheduler &sched, u32 step) {
        using namespace shamrock::patch;
        sched.for_each_patch_data([&](u64 id_patch, Patch p, PatchDataLayer &pdat) {
            if (pdat.is_empty()) {
                return;
            }
            std::vector<u32> ids = pdat.get_field<u32>(1).get_buf().copy_to_stdvec();
            std::vector<Tvec> pos(ids.size());
            for (u32 i = 0; i < ids.size(); i++) {
                pos[i] = get_pos(ids[i], step);
            }
            pdat.get_field<Tvec>(0).get_buf().copy_from_stdvec(pos);
        });
    }

    void reattribute(PatchScheduler &sched) {
        SerialPatchTree<Tvec> sptree(
            sched.patch_tree, sched.get_sim_box().get_patch_transform<Tvec>());
        sptree.attach_buf();
        shamrock::ReattributeDataUtility reatrib(sched);
        reatrib.reatribute_patch_objects(sptree, "xyz");
    }

    /// check that every object is in its patch, that no object was lost or duplicated, and that
    /// the fields of every object are still consistent with each other
    void check_state(PatchScheduler &sched, u32 npart, u32 step) {
        using namespace shamrock::patch;

        std::vector<u32> id_count(npart, 0);
        u32 total         = 0;
        bool all_in       = true;
        bool all_coherent = true;

        sched.for_each_patch_data([&](u64 id_patch, Patch p, PatchDataLayer &pdat) {
            if (pdat.is_empty()) {
                return;
            }
            auto [bmin, bmax] = sched.get_sim_box().patch_coord_to_domain<Tvec>(p);

            std::vector<Tvec> pos = pdat.get_field<Tvec>(0).get_buf().copy_to_stdvec();
            std::vector<u32> ids  = pdat.get_field<u32>(1).get_buf().copy_to_stdvec();

            for (u32 i = 0; i < ids.size(); i++) {
                all_in       = all_in && Patch::is_in_patch_converted(pos[i], bmin, bmax);
                all_coherent = all_coherent && sham::equals(pos[i], get_pos(ids[i], step));
                if (ids[i] < npart) {
                    id_count[ids[i]]++;
                }
            }
            total += ids.size();
        });

        bool all_once = true;
        for (u32 c : id_count) {
            all_once = all_once && (c == 1);
        }

        REQUIRE_EQUAL(total, npart);
        REQUIRE_NAMED("every object is in its patch", all_in);
        REQUIRE_NAMED("fields stay consistent", all_coherent);
        REQUIRE_NAMED("every object is present once", all_once);
    }

    std::vector<u32> get_patch_counts(PatchScheduler &sched) {
        using namespace shamrock::patch;
        std::vector<u32> ret;
        sched.for_each_patch_data([&](u64 id_patch, Patch p, PatchDataLayer &pdat) {
            ret.push_back(pdat.get_obj_cnt());
        });
        return ret;
    }

} // namespace

NEW_TEST(Unittest, "shamrock/scheduler/ReattributeDataUtility", 1) {

    using namespace shamrock::patch;

    auto layout_ptr = std::make_shared<PatchDataLayerLayout>();
    layout_ptr->add_field<Tvec>("xyz", 1);
    layout_ptr->add_field<u32>("id", 1);

    PatchScheduler sched(layout_ptr, 1e9, 1);
    sched.make_patch_base_grid<3>({{2, 2, 2}});
    sched.set_coord_domain_bound<Tvec>({0, 0, 0}, {1, 1, 1});

    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_load_value([&](Patch p) {
        return sched.patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });
    sched.scheduler_step(false, false);

    REQUIRE_EQUAL(sched.patch_list.global.size(), 8_u64);

    // insert every object in the first patch, the positions are random so most of them have to
    // move to one of the 7 other patches
    const u32 npart = 20000;
    {
        PatchDataLayer pdat_ins(layout_ptr);
        pdat_ins.resize(npart);

        std::vector<u32> ids(npart);
        std::vector<Tvec> pos(npart);
        for (u32 i = 0; i < npart; i++) {
            ids[i] = i;
            pos[i] = get_pos(i, 0);
        }
        pdat_ins.get_field<Tvec>(0).get_buf().copy_from_stdvec(pos);
        pdat_ins.get_field<u32>(1).get_buf().copy_from_stdvec(ids);

        bool inserted = false;
        sched.for_each_patch_data([&](u64 id_patch, Patch p, PatchDataLayer &pdat) {
            if (!inserted) {
                pdat.insert_elements(pdat_ins);
                inserted = true;
            }
        });
        REQUIRE(inserted);
    }

    // one source patch, many destinations
    reattribute(sched);
    check_state(sched, npart, 0);

    // nothing moves: the fast path must leave the patches untouched
    std::vector<u32> counts_before = get_patch_counts(sched);
    reattribute(sched);
    check_state(sched, npart, 0);
    REQUIRE_NAMED("patch counts unchanged", get_patch_counts(sched) == counts_before);

    // every patch sends to every other patch
    set_positions(sched, 1);
    reattribute(sched);
    check_state(sched, npart, 1);
}
