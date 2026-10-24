// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file to_shared_tests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/memory.hpp"
#include "shamtest/shamtest.hpp"
#include <type_traits>
#include <memory>
#include <utility>

namespace {

    struct Foo {
        int x;
        explicit Foo(int x) : x(x) {}
        Foo(const Foo &)            = delete;
        Foo &operator=(const Foo &) = delete;
        Foo(Foo &&)                 = default;
        Foo &operator=(Foo &&)      = default;
    };

    struct Node : std::enable_shared_from_this<Node> {
        int x;
        explicit Node(int x) : x(x) {}
    };

} // namespace

NEW_TEST(Unittest, "shambase/to_shared", 1) {

    {
        auto p = shambase::to_shared(Foo{42});
        static_assert(std::is_same_v<decltype(p), std::shared_ptr<Foo>>);
        REQUIRE(bool(p));
        REQUIRE_EQUAL(p->x, 42);
        REQUIRE_EQUAL(p.use_count(), 1);
    }

    {
        Foo named{7};
        auto p = shambase::to_shared(std::move(named));
        REQUIRE(bool(p));
        REQUIRE_EQUAL(p->x, 7);
    }

    {
        auto p = shambase::to_shared(Node{3});
        REQUIRE(bool(p));
        REQUIRE_EQUAL(p->x, 3);
        REQUIRE(p->shared_from_this() == p);
    }
}
