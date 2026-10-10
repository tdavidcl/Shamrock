// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>

NEW_TEST(Unittest, "shamrock/scheduler/PatchScheduler::scheduler_step:global_idx_map", 1) {

    using namespace shamrock::patch;

    auto pdl_ptr = std::make_shared<PatchDataLayerLayout>();
    pdl_ptr->add_field<f64_3>("xyz", 1);

    // a leaf with load > 1000 splits, a parent of only leaves with load < 100 merges
    PatchScheduler sched(pdl_ptr, 1000, 100);
    sched.get_sim_box().set_bounding_box<f64_3>({f64_3{0, 0, 0}, f64_3{1, 1, 1}});
    sched.add_root_patch();
    sched.owned_patch_id = sched.patch_list.build_local();

    auto set_load = [&](u64 load) {
        sched.update_local_load_value([&](Patch) {
            return load;
        });
    };

    auto step_and_check = [&](bool do_split_merge, bool do_load_balancing, u64 expected_cnt) {
        sched.scheduler_step(do_split_merge, do_load_balancing);
        REQUIRE(sched.patch_list.is_global_idx_map_valid());
        u64 valid_cnt = 0;
        for (const Patch &p : sched.patch_list.global) {
            valid_cnt += (p.is_err_mode()) ? 0 : 1;
        }
        REQUIRE_EQUAL(valid_cnt, expected_cnt);
    };

    // split 1 -> 8 -> 64
    set_load(2000);
    step_and_check(true, true, 8);
    set_load(2000);
    step_and_check(false, false, 8);
    set_load(2000);
    step_and_check(true, true, 64);
    set_load(500);
    step_and_check(false, false, 64);

    // neither split nor merge
    set_load(500);
    step_and_check(true, true, 64);
    step_and_check(false, false, 64);

    // merge 64 -> 8 -> 1, the merged patches stay in error mode in the global list until the
    // next global sync
    set_load(1);
    step_and_check(true, true, 8);
    set_load(1);
    step_and_check(false, false, 8);
    REQUIRE_EQUAL(sched.patch_list.global.size(), 8);
    set_load(1);
    step_and_check(true, true, 1);
    set_load(1);
    step_and_check(false, false, 1);
    REQUIRE_EQUAL(sched.patch_list.global.size(), 1);
}
