// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/aliases_int.hpp"
#include "shamtest/shamtest.hpp"
#include <ankerl/unordered_dense.h>
#include <algorithm>
#include <string>
#include <vector>

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

namespace {

    template<class Map>
    std::vector<typename Map::key_type> keys_in_iteration_order(const Map &map) {
        std::vector<typename Map::key_type> ret;
        for (const auto &[k, v] : map) {
            ret.push_back(k);
        }
        return ret;
    }

} // namespace

// Shamrock runs must be reproducible down to the byte, so iterating over a map must give the same
// order on every run. unordered_dense stores its values in a dense vector: an insertion appends to
// it and an erase moves the last value into the hole, so the order depends only on the sequence of
// operations, never on hash values or on the bucket layout.
NEW_TEST(Unittest, "external/unordered_dense::deterministic_iteration_order", 1) {

    using Keys = std::vector<std::string>;

    ankerl::unordered_dense::map<std::string, int> map;

    map["alpha"]   = 1;
    map["bravo"]   = 2;
    map["charlie"] = 3;
    map["delta"]   = 4;
    map["echo"]    = 5;
    REQUIRE_EQUAL_NAMED(
        "after insertions",
        keys_in_iteration_order(map),
        (Keys{"alpha", "bravo", "charlie", "delta", "echo"}));

    map.emplace("foxtrot", 6);
    REQUIRE_EQUAL_NAMED(
        "after another insertion",
        keys_in_iteration_order(map),
        (Keys{"alpha", "bravo", "charlie", "delta", "echo", "foxtrot"}));

    // touching existing keys must not move them
    map.emplace("bravo", 20);
    map["charlie"] = 30;
    REQUIRE_EQUAL_NAMED(
        "after updating existing keys",
        keys_in_iteration_order(map),
        (Keys{"alpha", "bravo", "charlie", "delta", "echo", "foxtrot"}));

    map.erase("bravo");
    REQUIRE_EQUAL_NAMED(
        "after erasing bravo",
        keys_in_iteration_order(map),
        (Keys{"alpha", "foxtrot", "charlie", "delta", "echo"}));

    map.erase("alpha");
    REQUIRE_EQUAL_NAMED(
        "after erasing alpha",
        keys_in_iteration_order(map),
        (Keys{"echo", "foxtrot", "charlie", "delta"}));

    map.erase("delta");
    REQUIRE_EQUAL_NAMED(
        "after erasing delta (last)",
        keys_in_iteration_order(map),
        (Keys{"echo", "foxtrot", "charlie"}));

    map["golf"]  = 7;
    map["bravo"] = 8;
    REQUIRE_EQUAL_NAMED(
        "after reinsertions",
        keys_in_iteration_order(map),
        (Keys{"echo", "foxtrot", "charlie", "golf", "bravo"}));

    std::vector<int> values;
    for (const auto &[k, v] : map) {
        values.push_back(v);
    }
    REQUIRE_EQUAL_NAMED("values", values, (std::vector<int>{5, 6, 30, 7, 8}));
}

// Same as above with enough elements to go through several rehashes, checked against a model of
// the dense vector and against a hardcoded digest of the final iteration order.
NEW_TEST(Unittest, "external/unordered_dense::deterministic_iteration_order_rehash", 1) {

    ankerl::unordered_dense::map<u64, u64> map;
    std::vector<u64> model;

    auto key_of = [](u64 i) -> u64 {
        return (i * 2654435761_u64) % 4294967296_u64;
    };

    for (u64 i = 0; i < 1000; i++) {
        map[key_of(i)] = i;
        model.push_back(key_of(i));
    }
    REQUIRE_EQUAL_NAMED("after insertions", keys_in_iteration_order(map), model);

    for (u64 i = 0; i < 1000; i += 3) {
        map.erase(key_of(i));

        auto it = std::find(model.begin(), model.end(), key_of(i));
        *it     = model.back();
        model.pop_back();
    }
    REQUIRE_EQUAL_NAMED("after deletions", keys_in_iteration_order(map), model);

    for (u64 i = 1000; i < 1500; i++) {
        map[key_of(i)] = i;
        model.push_back(key_of(i));
    }
    REQUIRE_EQUAL_NAMED("after reinsertions", keys_in_iteration_order(map), model);

    // FNV-1a over the keys in iteration order
    u64 digest = 14695981039346656037_u64;
    for (const auto &[k, v] : map) {
        digest = (digest ^ k) * 1099511628211_u64;
    }
    REQUIRE_EQUAL_NAMED("iteration order digest", digest, 16084171427108075294_u64);
}
