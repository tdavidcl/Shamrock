// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/ramses/modules/ShearingPeriodicUtils.hpp"
#include "shamtest/shamtest.hpp"

using namespace shammodels::basegodunov::modules;

using TgridVec = i64_3;
using AABB     = shammath::AABB<TgridVec>;

NEW_TEST(Unittest, "shammodels/ramses/modules/ShearingPeriodicUtils/side_shift", 1) {
    // ceil of the offset, residual in [0, 1)
    ShearSideShift p = get_shear_side_shift(5.25, 1);
    REQUIRE_EQUAL(p.int_shift, 6);
    REQUIRE_FLOAT_EQUAL(p.phi, 0.75, 1e-15);

    ShearSideShift m = get_shear_side_shift(5.25, -1);
    REQUIRE_EQUAL(m.int_shift, -5);
    REQUIRE_FLOAT_EQUAL(m.phi, 0.25, 1e-15);

    // integer offsets give an exact alignment
    ShearSideShift e = get_shear_side_shift(3.0, 1);
    REQUIRE_EQUAL(e.int_shift, 3);
    REQUIRE_EQUAL(e.phi, 0.0);

    // the offset is wrapped around zero
    REQUIRE_FLOAT_EQUAL(get_shear_offset_grid(-1.0, 0.75, 1.0 / 64, 64), 16.0, 1e-12);
    REQUIRE_FLOAT_EQUAL(get_shear_offset_grid(-1.0, 0.25, 1.0 / 64, 64), -16.0, 1e-12);
}

NEW_TEST(Unittest, "shammodels/ramses/modules/ShearingPeriodicUtils/overlap_uniform", 1) {
    // target [0, 4) along y (idir = 1), unit extent in x and z, footprint shifted by phi = 0.6
    AABB t{{0, 0, 0}, {1, 4, 1}};
    f64 phi = 0.6;

    auto below = compute_shear_overlap<f64>(t, AABB{{0, -4, 0}, {1, 0, 1}}, 1, phi);
    auto same  = compute_shear_overlap<f64>(t, AABB{{0, 0, 0}, {1, 4, 1}}, 1, phi);
    auto above = compute_shear_overlap<f64>(t, AABB{{0, 4, 0}, {1, 8, 1}}, 1, phi);
    auto far   = compute_shear_overlap<f64>(t, AABB{{0, 8, 0}, {1, 12, 1}}, 1, phi);

    REQUIRE(!below.positive);
    REQUIRE(!far.positive);
    REQUIRE(same.positive);
    REQUIRE(above.positive);

    // overlaps 4 - phi and phi, exactly tiling the footprint
    REQUIRE_EQUAL(same.vol_int, 4);
    REQUIRE_EQUAL(same.vol_coef, -1);
    REQUIRE_EQUAL(above.vol_int, 0);
    REQUIRE_EQUAL(above.vol_coef, 1);
    REQUIRE_EQUAL(same.vol_int + above.vol_int, 4);
    REQUIRE_EQUAL(same.vol_coef + above.vol_coef, 0);

    // overlap centroids minus the source centres: [0.6, 4) -> 2.3 - 2, [4, 4.6) -> 4.3 - 6
    REQUIRE_FLOAT_EQUAL(same.dr, 0.3, 1e-14);
    REQUIRE_FLOAT_EQUAL(above.dr, -1.7, 1e-14);
}

NEW_TEST(Unittest, "shammodels/ramses/modules/ShearingPeriodicUtils/overlap_aligned", 1) {
    // phi = 0: the footprint is exactly one source cell, the next one does not overlap
    AABB t{{0, 0, 0}, {1, 4, 1}};

    auto same  = compute_shear_overlap<f64>(t, AABB{{0, 0, 0}, {1, 4, 1}}, 1, 0.0);
    auto above = compute_shear_overlap<f64>(t, AABB{{0, 4, 0}, {1, 8, 1}}, 1, 0.0);

    REQUIRE(same.positive);
    REQUIRE_EQUAL(same.vol_int, 4);
    REQUIRE_EQUAL(same.vol_coef, 0);
    REQUIRE_EQUAL(same.dr, 0.0);
    REQUIRE(!above.positive);
}

NEW_TEST(Unittest, "shammodels/ramses/modules/ShearingPeriodicUtils/overlap_levels", 1) {
    // coarse target (4 units) over finer sources (2 units), phi = 0.5
    AABB t{{0, 0, 0}, {4, 4, 4}};
    f64 phi = 0.5;

    i64 sum_int  = 0;
    i64 sum_coef = 0;
    f64 sum_vol  = 0;
    // 2 x 2 sources in x/z, 3 along y covering [0, 6)
    for (i64 x = 0; x < 4; x += 2) {
        for (i64 z = 0; z < 4; z += 2) {
            for (i64 y = 0; y < 6; y += 2) {
                auto ov = compute_shear_overlap<f64>(t, AABB{{x, y, z}, {x + 2, y + 2, z + 2}}, 1, phi);
                REQUIRE(ov.positive);
                sum_int += ov.vol_int;
                sum_coef += ov.vol_coef;
                sum_vol += f64(ov.vol_int) + phi * f64(ov.vol_coef);
            }
        }
    }

    // exact coverage of the 4^3 target, written as an integer and a phi coefficient
    REQUIRE_EQUAL(sum_int, 64);
    REQUIRE_EQUAL(sum_coef, 0);
    REQUIRE_FLOAT_EQUAL(sum_vol, 64.0, 1e-12);

    // a source sharing only a face along x does not overlap
    auto face = compute_shear_overlap<f64>(t, AABB{{4, 0, 0}, {6, 2, 2}}, 1, phi);
    REQUIRE(!face.positive);
}
