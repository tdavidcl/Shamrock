// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/aliases_float.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/riemann.hpp"
#include "shammath/riemann_dust.hpp"
#include "shamtest/shamtest.hpp"

NEW_TEST(Unittest, "shammath/flux_symmetry", 1) {

    using Tcons = shammath::ConsState<f64_3>;
    using Tprim = shammath::PrimState<f64_3>;

    constexpr f64 gamma = 1.6666;

    // Riemann solvers now take primitive states directly (see riemann_hll.hpp,
    // riemann_hllc.hpp, riemann_rusanov.hpp), so the reference states below are converted
    // from conservative once, up front.
    Tcons cons1  = {.rho = 1._f64, .rhoe = 1.2_f64, .rhovel = f64_3{1, 0, 0}};
    Tcons cons2  = {.rho = 1.5_f64, .rhoe = 1._f64, .rhovel = f64_3{2, 0, 0}};
    Tprim state1 = shammath::cons_to_prim(cons1, gamma);
    Tprim state2 = shammath::cons_to_prim(cons2, gamma);

    {
        Tcons f1 = shammath::rusanov_flux_x(state1, state2, gamma);
        Tcons f2 = shammath::rusanov_flux_mx(state2, state1, gamma);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rho, -f2.rho, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhovel, -f2.rhovel, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhoe, -f2.rhoe, sham::equals);
    }

    {
        Tcons f1 = shammath::rusanov_flux_y(state1, state2, gamma);
        Tcons f2 = shammath::rusanov_flux_my(state2, state1, gamma);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rho, -f2.rho, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhovel, -f2.rhovel, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhoe, -f2.rhoe, sham::equals);
    }

    {
        Tcons f1 = shammath::rusanov_flux_z(state1, state2, gamma);
        Tcons f2 = shammath::rusanov_flux_mz(state2, state1, gamma);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rho, -f2.rho, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhovel, -f2.rhovel, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhoe, -f2.rhoe, sham::equals);
    }

    auto to_prim = [&](Tcons c) {
        return shammath::cons_to_prim(c, gamma);
    };

    Tprim state_xp = to_prim({.rho = 1.1_f64, .rhoe = 0.8_f64, .rhovel = f64_3{1.1, 0, 0}});
    Tprim state_yp = to_prim({.rho = 1._f64, .rhoe = 1._f64, .rhovel = f64_3{1, 0, 0}});
    Tprim state_zp = to_prim({.rho = 1._f64, .rhoe = 1._f64, .rhovel = f64_3{1, 0, 0}});
    Tprim state_i  = to_prim({.rho = 1._f64, .rhoe = 1._f64, .rhovel = f64_3{1, 0, 0}});
    Tprim state_xm = to_prim({.rho = 0.7_f64, .rhoe = 1.2_f64, .rhovel = f64_3{1.1, 0, 0}});
    Tprim state_ym = to_prim({.rho = 1._f64, .rhoe = 1._f64, .rhovel = f64_3{1, 0, 0}});
    Tprim state_zm = to_prim({.rho = 1._f64, .rhoe = 1._f64, .rhovel = f64_3{1, 0, 0}});
    {
        Tcons fx = shammath::rusanov_flux_x(state_i, state_xp, gamma);
        shamlog_debug_ln("Riemann Solver", fx.rho, fx.rhovel, fx.rhoe);
        Tcons fy = shammath::rusanov_flux_y(state_i, state_yp, gamma);
        shamlog_debug_ln("Riemann Solver", fy.rho, fy.rhovel, fy.rhoe);
        Tcons fz = shammath::rusanov_flux_z(state_i, state_zp, gamma);
        shamlog_debug_ln("Riemann Solver", fz.rho, fz.rhovel, fz.rhoe);
        Tcons fmx = shammath::rusanov_flux_mx(state_i, state_xm, gamma);
        shamlog_debug_ln("Riemann Solver", fmx.rho, fmx.rhovel, fmx.rhoe);
        Tcons fmy = shammath::rusanov_flux_my(state_i, state_ym, gamma);
        shamlog_debug_ln("Riemann Solver", fmy.rho, fmy.rhovel, fmy.rhoe);
        Tcons fmz = shammath::rusanov_flux_mz(state_i, state_zm, gamma);
        shamlog_debug_ln("Riemann Solver", fmz.rho, fmz.rhovel, fmz.rhoe);
        Tcons sum = fx + fy + fz + fmx + fmy + fmz;
        shamlog_debug_ln("Riemann Solver", "sum=", sum.rho, sum.rhovel, sum.rhoe);
        REQUIRE(sum.rhovel[1] == 0);
        REQUIRE(sum.rhovel[2] == 0);
    }
}

NEW_TEST(Unittest, "shammath/flux_n_matches_directional", 1) {

    using Tvec  = f64_3;
    using Tcons = shammath::ConsState<Tvec>;
    using Tprim = shammath::PrimState<Tvec>;

    using DTcons = shammath::DustConsState<Tvec>;
    using DTprim = shammath::DustPrimState<Tvec>;

    constexpr f64 gamma = 1.6666;

    // Every _n(..., n) call should reproduce the corresponding permutation-based
    // _flux_<direction>(...) call when n is one of the six axis-aligned unit vectors.
    // Compared with a tolerance rather than exact equality: the two code paths group
    // floating point operations differently, so compiler-dependent choices (e.g. FMA
    // contraction) can make them differ by a ULP or two.
    constexpr f64 eps = 1e-15;

    auto to_prim = [&](Tcons c) {
        return shammath::cons_to_prim(c, gamma);
    };

    Tprim pL = to_prim({.rho = 1.2_f64, .rhoe = 1.1_f64, .rhovel = f64_3{0.3, -0.2, 0.5}});
    Tprim pR = to_prim({.rho = 0.9_f64, .rhoe = 1.4_f64, .rhovel = f64_3{-0.1, 0.4, -0.3}});

    DTprim dL{.rho = 1.1_f64, .vel = f64_3{0.2, -0.3, 0.1}};
    DTprim dR{.rho = 0.8_f64, .vel = f64_3{-0.4, 0.1, 0.2}};

    auto require_cons_equal = [&](Tcons lhs, Tcons rhs) {
        REQUIRE_FLOAT_EQUAL(lhs.rho, rhs.rho, eps);
        REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED("", lhs.rhovel, rhs.rhovel, eps, sycl::length);
        REQUIRE_FLOAT_EQUAL(lhs.rhoe, rhs.rhoe, eps);
    };

    auto require_dust_cons_equal = [&](DTcons lhs, DTcons rhs) {
        REQUIRE_FLOAT_EQUAL(lhs.rho, rhs.rho, eps);
        REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED("", lhs.rhovel, rhs.rhovel, eps, sycl::length);
    };

    auto check_gas_solver
        = [&](auto solver_n, Tcons fx, Tcons fy, Tcons fz, Tcons fmx, Tcons fmy, Tcons fmz) {
              require_cons_equal(solver_n(pL, pR, gamma, Tvec{1, 0, 0}), fx);
              require_cons_equal(solver_n(pL, pR, gamma, Tvec{0, 1, 0}), fy);
              require_cons_equal(solver_n(pL, pR, gamma, Tvec{0, 0, 1}), fz);
              require_cons_equal(solver_n(pL, pR, gamma, Tvec{-1, 0, 0}), fmx);
              require_cons_equal(solver_n(pL, pR, gamma, Tvec{0, -1, 0}), fmy);
              require_cons_equal(solver_n(pL, pR, gamma, Tvec{0, 0, -1}), fmz);
          };

    auto check_dust_solver
        = [&](auto solver_n, DTcons fx, DTcons fy, DTcons fz, DTcons fmx, DTcons fmy, DTcons fmz) {
              require_dust_cons_equal(solver_n(dL, dR, Tvec{1, 0, 0}), fx);
              require_dust_cons_equal(solver_n(dL, dR, Tvec{0, 1, 0}), fy);
              require_dust_cons_equal(solver_n(dL, dR, Tvec{0, 0, 1}), fz);
              require_dust_cons_equal(solver_n(dL, dR, Tvec{-1, 0, 0}), fmx);
              require_dust_cons_equal(solver_n(dL, dR, Tvec{0, -1, 0}), fmy);
              require_dust_cons_equal(solver_n(dL, dR, Tvec{0, 0, -1}), fmz);
          };

    check_gas_solver(
        [](Tprim a, Tprim b, f64 g, Tvec n) {
            return shammath::rusanov_flux_n(a, b, g, n);
        },
        shammath::rusanov_flux_x(pL, pR, gamma),
        shammath::rusanov_flux_y(pL, pR, gamma),
        shammath::rusanov_flux_z(pL, pR, gamma),
        shammath::rusanov_flux_mx(pL, pR, gamma),
        shammath::rusanov_flux_my(pL, pR, gamma),
        shammath::rusanov_flux_mz(pL, pR, gamma));

    check_gas_solver(
        [](Tprim a, Tprim b, f64 g, Tvec n) {
            return shammath::hll_flux_n(a, b, g, n);
        },
        shammath::hll_flux_x(pL, pR, gamma),
        shammath::hll_flux_y(pL, pR, gamma),
        shammath::hll_flux_z(pL, pR, gamma),
        shammath::hll_flux_mx(pL, pR, gamma),
        shammath::hll_flux_my(pL, pR, gamma),
        shammath::hll_flux_mz(pL, pR, gamma));

    check_gas_solver(
        [](Tprim a, Tprim b, f64 g, Tvec n) {
            return shammath::hllc_adiab_toro_flux_n(a, b, g, n);
        },
        shammath::hllc_adiab_toro_flux_x(pL, pR, gamma),
        shammath::hllc_adiab_toro_flux_y(pL, pR, gamma),
        shammath::hllc_adiab_toro_flux_z(pL, pR, gamma),
        shammath::hllc_adiab_toro_flux_mx(pL, pR, gamma),
        shammath::hllc_adiab_toro_flux_my(pL, pR, gamma),
        shammath::hllc_adiab_toro_flux_mz(pL, pR, gamma));

    check_gas_solver(
        [](Tprim a, Tprim b, f64 g, Tvec n) {
            return shammath::hllc_davis_flux_n(a, b, g, n);
        },
        shammath::hllc_davis_flux_x(pL, pR, gamma),
        shammath::hllc_davis_flux_y(pL, pR, gamma),
        shammath::hllc_davis_flux_z(pL, pR, gamma),
        shammath::hllc_davis_flux_mx(pL, pR, gamma),
        shammath::hllc_davis_flux_my(pL, pR, gamma),
        shammath::hllc_davis_flux_mz(pL, pR, gamma));

    check_dust_solver(
        [](DTprim a, DTprim b, Tvec n) {
            return shammath::d_hll_flux_n(a, b, n);
        },
        shammath::d_hll_flux_x(dL, dR),
        shammath::d_hll_flux_y(dL, dR),
        shammath::d_hll_flux_z(dL, dR),
        shammath::d_hll_flux_mx(dL, dR),
        shammath::d_hll_flux_my(dL, dR),
        shammath::d_hll_flux_mz(dL, dR));

    check_dust_solver(
        [](DTprim a, DTprim b, Tvec n) {
            return shammath::huang_bai_flux_n(a, b, n);
        },
        shammath::huang_bai_flux_x(dL, dR),
        shammath::huang_bai_flux_y(dL, dR),
        shammath::huang_bai_flux_z(dL, dR),
        shammath::huang_bai_flux_mx(dL, dR),
        shammath::huang_bai_flux_my(dL, dR),
        shammath::huang_bai_flux_mz(dL, dR));
}
