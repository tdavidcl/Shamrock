// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file human_readable.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Core formatting functions: `format`, `vformat`, and `format_printf`
 *
 * This is the primary entry point for the shamformat library. It re-exports
 * aliases and exception declarations from the other headers and provides the
 * public formatting API wrapped with exception handling.
 */

#include <array>
#include <cmath>

namespace sham {

    namespace details {

        consteval std::array<std::pair<const char *, double>, 12> make_si_pairs() {
            return {
                {{"n", 1e-9},
                 {"u", 1e-6},
                 {"m", 1e-3},
                 {"", 1.0},
                 {"k", 1e3},
                 {"M", 1e6},
                 {"G", 1e9},
                 {"T", 1e12},
                 {"P", 1e15},
                 {"E", 1e18},
                 {"Z", 1e21},
                 {"Y", 1e24}}};
        }

    } // namespace details

    struct human_readable_t {
        double value;
        const char *prefix;
        double ratio;
    };

    inline human_readable_t to_human_readable(double value) {
        static constexpr auto si = details::make_si_pairs();

        double ax = std::fabs(value);

        // zero: no prefix
        if (ax == 0.0) {
            return {.value = 0.0, .prefix = "", .ratio = 1.0};
        }

        for (int i = static_cast<int>(si.size()) - 1; i >= 0; --i) {
            if (ax >= si[i].second) {
                return {
                    .value = value / si[i].second, .prefix = si[i].first, .ratio = si[i].second};
            }
        }

        // too small, clamp to nano
        const auto &smallest = si.front();
        if (ax < si.front().second) {
            return {
                .value  = value / smallest.second,
                .prefix = smallest.first,
                .ratio  = smallest.second};
        }

        // too large, clamp to yotta
        const auto &largest = si.back();
        if (ax >= si.back().second) {
            return {
                .value = value / largest.second, .prefix = largest.first, .ratio = largest.second};
        }

        return {.value = value, .prefix = "", .ratio = 1.0}; // unreachable in correct table
    }

} // namespace sham
