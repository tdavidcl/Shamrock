// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "sham/format/format.hpp"
#include "sham/format/human_readable.hpp"
#include "shamtest/shamtest.hpp"
#include <string_view>

namespace {
    void throwing_format() {
        std::string fmt = "{"; // runtime format string
        int value       = 42;

        auto s = shambase::vformat(std::string_view{fmt}, fmt::make_format_args(value));

        // just to trap the result and avoid optimizations
        std::cout << s << '\n';
    }
} // namespace

TestStart(Unittest, "shamformat/format", test_exception_throw, 1) {
    REQUIRE_EXCEPTION_THROW(throwing_format(), fmt::format_error);
}

TestStart(Unittest, "shamformat/human_readable", test_to_human_readable, 1) {
    using sham::human_readable_t;
    using sham::to_human_readable;

    // Zero: no prefix
    {
        auto hr = to_human_readable(0.0);
        REQUIRE_FLOAT_EQUAL(hr.value, 0.0, 1e-15);
        REQUIRE_EQUAL(hr.prefix, "");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1.0, 1e-15);
    }

    // Too small (below nano): clamped to nano
    {
        auto hr = to_human_readable(1e-10);
        REQUIRE_FLOAT_EQUAL(hr.value, 0.1, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "n");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1e-9, 1e-15);
    }

    // Too large (above yotta): clamped to yotta
    {
        auto hr = to_human_readable(1e26);
        REQUIRE_FLOAT_EQUAL(hr.value, 100.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "Y");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1e24, 1e20);
    }

    // Exactly at boundaries
    {
        auto hr = to_human_readable(1e-9);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "n");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1e-9, 1e-15);
    }

    {
        auto hr = to_human_readable(1e24);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "Y");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1e24, 1e20);
    }

    // Common SI prefixes
    {
        auto hr = to_human_readable(1e3);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "k");
    }

    {
        auto hr = to_human_readable(1e6);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "M");
    }

    {
        auto hr = to_human_readable(1e9);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "G");
    }

    {
        auto hr = to_human_readable(1e12);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "T");
    }

    // No prefix needed (value in [1, 1000))
    {
        auto hr = to_human_readable(500.0);
        REQUIRE_FLOAT_EQUAL(hr.value, 500.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1.0, 1e-15);
    }

    // Negative values
    {
        auto hr = to_human_readable(-1e3);
        REQUIRE_FLOAT_EQUAL(hr.value, -1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "k");
    }

    // Boundary: just below kilo (999.999)
    {
        auto hr = to_human_readable(999.0);
        REQUIRE_FLOAT_EQUAL(hr.value, 999.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "");
    }

    // Between prefixes: 2500 -> 2.5 k
    {
        auto hr = to_human_readable(2500.0);
        REQUIRE_FLOAT_EQUAL(hr.value, 2.5, 1e-3);
        REQUIRE_EQUAL(hr.prefix, "k");
    }

    // Peta/E zetta: large non-standard values
    {
        auto hr = to_human_readable(5e15);
        REQUIRE_FLOAT_EQUAL(hr.value, 5.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "P");
    }
}
