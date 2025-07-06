// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammath/AABB.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shammath/AABB/is_volume_not_null", test_aabb_is_volume_not_null, 1) {

    using namespace shammath;

    // Test AABB volume checks
    AABB<f32_3> aabb1{f32_3{0, 0, 0}, f32_3{1, 1, 1}};
    REQUIRE(aabb1.is_volume_not_null()); // Has non-zero volume

    AABB<f32_3> aabb2{f32_3{0, 0, 0}, f32_3{0, 0, 0}};
    REQUIRE(!aabb2.is_volume_not_null()); // Zero volume
}

TestStart(Unittest, "shammath/AABB/is_surface", test_aabb_is_surface, 1) {

    using namespace shammath;

    // Test surface detection
    AABB<f32_3> surface_aabb{f32_3{0, 0, 0}, f32_3{1, 0, 1}};
    REQUIRE(surface_aabb.is_surface()); // Only y dimension is zero

    AABB<f32_3> volume_aabb{f32_3{0, 0, 0}, f32_3{1, 1, 1}};
    REQUIRE(!volume_aabb.is_surface()); // All dimensions non-zero
}

TestStart(Unittest, "shammath/AABB/is_surface_or_volume", test_aabb_is_surface_or_volume, 1) {

    using namespace shammath;

    // Test surface or volume detection
    AABB<f32_3> surface_aabb{f32_3{0, 0, 0}, f32_3{1, 0, 1}};
    REQUIRE(surface_aabb.is_surface_or_volume()); // Surface

    AABB<f32_3> volume_aabb{f32_3{0, 0, 0}, f32_3{1, 1, 1}};
    REQUIRE(volume_aabb.is_surface_or_volume()); // Volume

    AABB<f32_3> line_aabb{f32_3{0, 0, 0}, f32_3{1, 0, 0}};
    REQUIRE(!line_aabb.is_surface_or_volume()); // Line (2 zero dimensions)
}

TestStart(Unittest, "shammath/AABB/clamp_coord", test_aabb_clamp_coord, 1) {

    using namespace shammath;

    AABB<f32_3> aabb{f32_3{0, 0, 0}, f32_3{1, 1, 1}};

    f32_3 inside_coord{0.5, 0.5, 0.5};
    f32_3 clamped_inside = aabb.clamp_coord(inside_coord);
    REQUIRE(sham::equals(clamped_inside, inside_coord)); // No change

    f32_3 outside_coord{2.0, -1.0, 0.5};
    f32_3 clamped_outside = aabb.clamp_coord(outside_coord);
    f32_3 expected_clamped{1.0, 0.0, 0.5};
    REQUIRE(sham::equals(clamped_outside, expected_clamped)); // Clamped to bounds
} 