// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammath/matrix.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shammath/matrix::mat_inv_33", test_inv_33, 1) {

    shammath::mat<f32, 3, 3> mat{
        //clang-format off
        0,
        -3,
        -2,
        1,
        -4,
        -2,
        -3,
        4,
        1
        //clang-format on
    };

    shammath::mat<f32, 3, 3> expected_inverse{
        //clang-format off
        4,
        -5,
        -2,
        5,
        -6,
        -2,
        -8,
        9,
        3
        //clang-format on
    };

    shammath::mat<f32, 3, 3> result;
    shammath::mat_inv_33(mat.get_mdspan(), result.get_mdspan());

    REQUIRE(result == expected_inverse);
}

TestStart(Unittest, "shammath/matrix::mat_prod", test_mat_prod, 1) {

    shammath::mat<f32, 3, 3> mat{
        //clang-format off
        0,
        -3,
        -2,
        1,
        -4,
        -2,
        -3,
        4,
        1
        //clang-format on
    };

    shammath::mat<f32, 3, 3> inverse{
        //clang-format off
        4,
        -5,
        -2,
        5,
        -6,
        -2,
        -8,
        9,
        3
        //clang-format on
    };

    shammath::mat<f32, 3, 3> id = shammath::mat_identity<f32, 3>();

    shammath::mat<f32, 3, 3> result;
    shammath::mat_prod(mat.get_mdspan(), inverse.get_mdspan(), result.get_mdspan());

    REQUIRE_EQUAL(result.data, id.data);
}
