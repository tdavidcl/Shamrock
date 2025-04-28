// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SedovTaylor.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
namespace shamphys {

    /**
     * @brief Represents a Sedov-Taylor solution, a self-similar solution to the hydrodynamic
     * equations describing a blast wave.
     *
     * This class provides a way to calculate the values of density, velocity, and pressure at a
     * given time and position using the Sedov-Taylor solution.
     *
     * gamma = 5./3.
     * t = 0.1
     * \int u_inj = 1
     *
     */
    class SedovTaylor {
        public:
        inline SedovTaylor() {}

        struct field_val {
            f64 rho, vx, P;
        };

        field_val get_value(f64 x);
    };

} // namespace shamphys
