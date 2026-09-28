// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file string.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/string.hpp"

namespace shambase {

    std::string details::format_array_line_prefix(u32 i) {
        if (i == 0) {
            return sham::format("{:8} : ", i);
        } else {
            return sham::format("\n{:8} : ", i);
        }
    }

    std::string readable_sizeof(double size) {
        auto res = sham::to_human_readable<false>(size);
        return sham::format("{:.2f} {}B", res.value, res.prefix);
    }

    std::string shorten_string(std::string str, u32 len) {
        if (len > str.size()) {
            throw make_except_with_loc<std::invalid_argument>(
                "the string is too short to be shortened"
                "\n args : "
                + sham::format("{} : {} \n {} : {}", "str", str, "len", len));
        }
        return str.substr(0, str.size() - len);
    }

} // namespace shambase
