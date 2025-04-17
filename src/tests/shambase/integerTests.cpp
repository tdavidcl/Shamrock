// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/integer.hpp"
#include "shamtest/shamtest.hpp"
#include <cstdio>

TestStart(Unittest, "shambase/integer::round_up_multiple", integerTests_round_up_multiple, 1) {
    using namespace shambase;

    auto test_func = [&](int value, int multiple) {
        int result = round_up_multiple(value, multiple);

        printf("round_up_multiple(%d, %d) = %d\n", value, multiple, result);

        if (value % multiple == 0) {
            REQUIRE_EQUAL(result, value);
        } else {
            REQUIRE(result > value);
            REQUIRE(result % multiple == 0);
        }
    };

    for (int i = 0; i < 1000; i++) {
        test_func(i, 1);
        test_func(i, 2);
        test_func(i, 3);
        test_func(i, 4);
        test_func(i, 5);
        test_func(i, 6);
        test_func(i, 7);
        test_func(i, 8);
        test_func(i, 9);
        test_func(i, 10);
        test_func(i, 11);
        test_func(i, 12);
        test_func(i, 13);
        test_func(i, 14);
        test_func(i, 15);
        test_func(i, 16);
    }
}
