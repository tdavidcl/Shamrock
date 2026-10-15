// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/shamtest.hpp"
#include <ankerl/unordered_dense.h>
#include <string>

NEW_TEST(Unittest, "external/unordered_dense::map", 1) {

    ankerl::unordered_dense::map<std::string, int> map;

    map["a"] = 1;
    map["b"] = 2;
    map.emplace("c", 3);

    REQUIRE_EQUAL(map.size(), 3);
    REQUIRE_EQUAL(map.at("b"), 2);
    REQUIRE(map.contains("c"));
    REQUIRE(!map.contains("d"));

    map.erase("a");
    REQUIRE_EQUAL(map.size(), 2);
    REQUIRE(!map.contains("a"));
}

NEW_TEST(Unittest, "external/unordered_dense::set", 1) {

    ankerl::unordered_dense::set<int> set{1, 2, 3, 2, 1};

    REQUIRE_EQUAL(set.size(), 3);
    REQUIRE(set.contains(2));
    REQUIRE(!set.contains(4));
}
