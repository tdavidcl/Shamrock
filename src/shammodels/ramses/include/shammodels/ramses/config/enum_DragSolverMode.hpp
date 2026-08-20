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
 * @file enum_DragSolverMode.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Drag solver mode enum + json serialization/deserialization
 *
 */

#include "shambase/exception.hpp"
#include "nlohmann/json_fwd.hpp"

namespace shammodels::basegodunov {

    enum DragSolverMode {
        NoDrag = 0,
        IRK1   = 1, // Implicit RK1
        IRK2   = 2, // Implicit RK2
        EXPO   = 3  // Matrix exponential
    };

    void to_json(nlohmann::json &j, const DragSolverMode &e);
    void from_json(const nlohmann::json &j, DragSolverMode &e);

} // namespace shammodels::basegodunov
