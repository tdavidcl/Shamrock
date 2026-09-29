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
 * @file dust_variables.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Evolved dust variables X_j of the monofluid TVA solver and their map to/from the dust
 * fraction \f$\epsilon_j\f$.
 *
 */

#include "shambase/exception.hpp"
#include "shambackends/sycl.hpp"
#include <string>

namespace shammodels::sph {

    /**
     * @brief Variable evolved by the monofluid TVA dust solver for each dust species j
     *
     * - SqrtRhoEps : \f$S_j = \sqrt{\rho \epsilon_j}\f$ (Hutchison, Price & Laibe 2018, eq. 51)
     * - Eps : \f$\epsilon_j\f$ with the arithmetic pair coefficient (Price & Laibe 2015 direct
     *   form)
     * - SqrtEpsOverOneMinusEps : \f$s_j = \sqrt{\epsilon_j / (1-\epsilon_j)}\f$ (per-species
     *   generalisation of Ballabio et al. 2018)
     */
    enum class DustVariable { SqrtRhoEps, Eps, SqrtEpsOverOneMinusEps };

    /// Dust fraction \f$\epsilon_j\f$ from the evolved variable X_j and the total density rho
    template<class Tscal>
    inline Tscal dust_var_to_eps(DustVariable v, Tscal X, Tscal rho) {
        switch (v) {
        case DustVariable::SqrtRhoEps            : return X * X / rho;
        case DustVariable::Eps                   : return X;
        case DustVariable::SqrtEpsOverOneMinusEps: return X * X / (1 + X * X);
        }
        return X;
    }

    /// Evolved variable X_j from the dust fraction \f$\epsilon_j\f$ and the total density rho
    template<class Tscal>
    inline Tscal dust_var_from_eps(DustVariable v, Tscal eps, Tscal rho) {
        switch (v) {
        case DustVariable::SqrtRhoEps            : return sycl::sqrt(rho * eps);
        case DustVariable::Eps                   : return eps;
        case DustVariable::SqrtEpsOverOneMinusEps: return sycl::sqrt(eps / (1 - eps));
        }
        return eps;
    }

    /// Name of the dust variable used in configs (json / python)
    inline std::string dust_variable_to_string(DustVariable v) {
        switch (v) {
        case DustVariable::SqrtRhoEps            : return "sqrt_rho_eps";
        case DustVariable::Eps                   : return "eps";
        case DustVariable::SqrtEpsOverOneMinusEps: return "sqrt_eps_over_1m_eps";
        }
        return "unknown";
    }

    /// Inverse of dust_variable_to_string
    inline DustVariable dust_variable_from_string(const std::string &s) {
        if (s == "sqrt_rho_eps") {
            return DustVariable::SqrtRhoEps;
        }
        if (s == "eps") {
            return DustVariable::Eps;
        }
        if (s == "sqrt_eps_over_1m_eps") {
            return DustVariable::SqrtEpsOverOneMinusEps;
        }
        shambase::throw_with_loc<std::invalid_argument>(
            "unknown dust variable \"" + s
            + "\", expected one of: sqrt_rho_eps, eps, sqrt_eps_over_1m_eps");
        return DustVariable::SqrtRhoEps;
    }

    /// Name of the patch field holding the evolved dust variable
    inline std::string dust_variable_field_name(DustVariable v) {
        switch (v) {
        case DustVariable::SqrtRhoEps            : return "s_j";
        case DustVariable::Eps                   : return "eps_j";
        case DustVariable::SqrtEpsOverOneMinusEps: return "sb_j";
        }
        return "s_j";
    }

    /// Name of the patch field holding the time derivative of the evolved dust variable
    inline std::string dust_variable_deriv_field_name(DustVariable v) {
        return "d" + dust_variable_field_name(v) + "_dt";
    }

} // namespace shammodels::sph
