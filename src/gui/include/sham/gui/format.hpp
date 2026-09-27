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
 * @file format.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Number formatting helpers matching the Python app's format specs.
 *
 */

#include <cstdio>
#include <string>

namespace sham::gui {

    // ============================================================================
    //  Formatting helpers (match Python's format specs)
    // ============================================================================
    inline std::string fmt(const char *f, double v) {
        char buf[64];
        std::snprintf(buf, sizeof buf, f, v);
        return buf;
    }
    inline std::string fmt_thousands(long long v) {
        std::string s = std::to_string(v), out;
        int n         = int(s.size());
        for (int i = 0; i < n; ++i) {
            out += s[i];
            if ((n - i - 1) % 3 == 0 && i != n - 1)
                out += ',';
        }
        return out;
    }
    inline std::string strip_zeros(std::string s) {
        while (!s.empty() && s.back() == '0')
            s.pop_back();
        if (!s.empty() && s.back() == '.')
            s.pop_back();
        return s;
    }

} // namespace sham::gui
