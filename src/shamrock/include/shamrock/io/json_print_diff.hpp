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
 * @file json_print_diff.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "nlohmann/json_fwd.hpp"
#include <string>

namespace shamrock {

    /// Shown the line-oriented diff between two JSON objects
    std::string json_diff_str(const nlohmann::json &j1, const nlohmann::json &j2);
} // namespace shamrock
