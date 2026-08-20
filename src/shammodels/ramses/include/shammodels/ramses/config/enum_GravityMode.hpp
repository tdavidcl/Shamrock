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
 * @file enum_GravityMode.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Gravity mode enum + json serialization/deserialization
 */

#include "shambase/exception.hpp"
#include "nlohmann/json_fwd.hpp"

namespace shammodels::basegodunov {

    enum GravityMode {
        NoGravity = 0,
        CG        = 1, // conjuguate gradient
        PCG       = 2, // preconditioned conjuguate gradient
        BICGSTAB  = 3, // bicgstab
        MULTIGRID = 4  // multigrid
    };

    void to_json(nlohmann::json &j, const GravityMode &e);
    void from_json(const nlohmann::json &j, GravityMode &e);

} // namespace shammodels::basegodunov
