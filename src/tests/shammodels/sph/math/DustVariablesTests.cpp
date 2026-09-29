// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/sph/math/dust_variables.hpp"
#include "shamtest/shamtest.hpp"
#include <vector>

NEW_TEST(Unittest, "shammodels/sph/math/dust_variables", 1) {
    using Tscal = f64;
    using namespace shammodels::sph;

    const std::vector<DustVariable> vars
        = {DustVariable::SqrtRhoEps, DustVariable::Eps, DustVariable::SqrtEpsOverOneMinusEps};

    Tscal tol = 1e-12;

    // eps -> X -> eps round trip
    for (DustVariable v : vars) {
        for (Tscal rho : {0.1, 1.0, 7.5}) {
            for (Tscal eps : {0.0, 1e-6, 0.01, 0.3, 0.5, 0.9, 0.99}) {
                Tscal X    = dust_var_from_eps(v, eps, rho);
                Tscal back = dust_var_to_eps(v, X, rho);
                REQUIRE_FLOAT_EQUAL_NAMED(
                    sham::format("{} rho={} eps={}", dust_variable_to_string(v), rho, eps),
                    back,
                    eps,
                    tol);
            }
        }
    }

    // explicit values of the maps
    REQUIRE_FLOAT_EQUAL(dust_var_to_eps(DustVariable::SqrtRhoEps, 0.5, 2.0), 0.125, tol);
    REQUIRE_FLOAT_EQUAL(dust_var_to_eps(DustVariable::Eps, 0.25, 2.0), 0.25, tol);
    REQUIRE_FLOAT_EQUAL(dust_var_to_eps(DustVariable::SqrtEpsOverOneMinusEps, 1.0, 2.0), 0.5, tol);

    // s_j = sqrt(eps_j/(1-eps_j)) maps any real s_j to 0 <= eps_j < 1
    for (Tscal s : {-1e3, -2.0, -0.1, 0.0, 0.1, 2.0, 1e3}) {
        Tscal eps = dust_var_to_eps(DustVariable::SqrtEpsOverOneMinusEps, s, 1.0);
        REQUIRE_NAMED(sham::format("0 <= eps(s={}) < 1", s), eps >= 0 && eps < 1);
    }

    // config names round trip
    for (DustVariable v : vars) {
        REQUIRE(dust_variable_from_string(dust_variable_to_string(v)) == v);
    }
    REQUIRE_EXCEPTION_THROW(dust_variable_from_string("not_a_variable"), std::invalid_argument);

    // field names : the default keeps the historical s_j name, the others are distinct
    REQUIRE(dust_variable_field_name(DustVariable::SqrtRhoEps) == "s_j");
    REQUIRE(dust_variable_deriv_field_name(DustVariable::SqrtRhoEps) == "ds_j_dt");
    REQUIRE(dust_variable_field_name(DustVariable::Eps) == "eps_j");
    REQUIRE(dust_variable_deriv_field_name(DustVariable::Eps) == "deps_j_dt");
    REQUIRE(dust_variable_field_name(DustVariable::SqrtEpsOverOneMinusEps) == "sb_j");
    REQUIRE(dust_variable_deriv_field_name(DustVariable::SqrtEpsOverOneMinusEps) == "dsb_j_dt");
}
