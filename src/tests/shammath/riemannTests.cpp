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
#include <array>
#include <utility>
#include <vector>

NEW_TEST(Unittest, "shammath/flux_symmetry", 1) {

    using Tcons = shammath::ConsState<f64_3>;
    using Tprim = shammath::PrimState<f64_3>;

    constexpr f64 gamma = 1.6666;

    shammath::FluidStateAdiabatic<f64_3> adiab_fluid{.m_gamma = gamma};

    // Riemann solvers now take primitive states directly (see riemann_hll.hpp,
    // riemann_hllc.hpp, riemann_rusanov.hpp), so the reference states below are converted
    // from conservative once, up front.
    Tcons cons1  = {.rho = 1._f64, .rhoe = 1.2_f64, .rhovel = f64_3{1, 0, 0}};
    Tcons cons2  = {.rho = 1.5_f64, .rhoe = 1._f64, .rhovel = f64_3{2, 0, 0}};
    Tprim state1 = shammath::cons_to_prim(cons1, gamma);
    Tprim state2 = shammath::cons_to_prim(cons2, gamma);

    {
        Tcons f1 = shammath::rusanov_flux(adiab_fluid, state1, state2, f64_3{1, 0, 0});
        Tcons f2 = shammath::rusanov_flux(adiab_fluid, state2, state1, f64_3{-1, 0, 0});
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rho, -f2.rho, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhovel, -f2.rhovel, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhoe, -f2.rhoe, sham::equals);
    }

    {
        Tcons f1 = shammath::rusanov_flux(adiab_fluid, state1, state2, f64_3{0, 1, 0});
        Tcons f2 = shammath::rusanov_flux(adiab_fluid, state2, state1, f64_3{0, -1, 0});
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rho, -f2.rho, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhovel, -f2.rhovel, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(f1.rhoe, -f2.rhoe, sham::equals);
    }

    {
        Tcons f1 = shammath::rusanov_flux(adiab_fluid, state1, state2, f64_3{0, 0, 1});
        Tcons f2 = shammath::rusanov_flux(adiab_fluid, state2, state1, f64_3{0, 0, -1});
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
        Tcons fx = shammath::rusanov_flux(adiab_fluid, state_i, state_xp, f64_3{1, 0, 0});
        shamlog_debug_ln("Riemann Solver", fx.rho, fx.rhovel, fx.rhoe);
        Tcons fy = shammath::rusanov_flux(adiab_fluid, state_i, state_yp, f64_3{0, 1, 0});
        shamlog_debug_ln("Riemann Solver", fy.rho, fy.rhovel, fy.rhoe);
        Tcons fz = shammath::rusanov_flux(adiab_fluid, state_i, state_zp, f64_3{0, 0, 1});
        shamlog_debug_ln("Riemann Solver", fz.rho, fz.rhovel, fz.rhoe);
        Tcons fmx = shammath::rusanov_flux(adiab_fluid, state_i, state_xm, f64_3{-1, 0, 0});
        shamlog_debug_ln("Riemann Solver", fmx.rho, fmx.rhovel, fmx.rhoe);
        Tcons fmy = shammath::rusanov_flux(adiab_fluid, state_i, state_ym, f64_3{0, -1, 0});
        shamlog_debug_ln("Riemann Solver", fmy.rho, fmy.rhovel, fmy.rhoe);
        Tcons fmz = shammath::rusanov_flux(adiab_fluid, state_i, state_zm, f64_3{0, 0, -1});
        shamlog_debug_ln("Riemann Solver", fmz.rho, fmz.rhovel, fmz.rhoe);
        Tcons sum = fx + fy + fz + fmx + fmy + fmz;
        shamlog_debug_ln("Riemann Solver", "sum=", sum.rho, sum.rhovel, sum.rhoe);
        REQUIRE(sum.rhovel[1] == 0);
        REQUIRE(sum.rhovel[2] == 0);
    }
}

namespace {

    // Local stand-ins for the gas riemann_common.hpp _x/_y/_z/_mx/_my/_mz axis dispatch
    // (riemann_rusanov.hpp etc.), rebuilt here from a plain n-taking solver so this test can
    // keep validating axis-rotation vs. direct n-projection after those per-axis overloads
    // are removed from the solvers themselves. flux_func is expected to have the same
    // signature as the solvers: (primL, primR, gamma, n).
    template<class Tprim, class Func>
    inline auto _x_dispatch(
        Func &&flux_func, Tprim primL, Tprim primR, typename Tprim::Tscal gamma) {
        return flux_func(primL, primR, gamma, typename Tprim::Tvec{1, 0, 0});
    }

    template<class Tprim, class Func>
    inline auto _y_dispatch(
        Func &&flux_func, Tprim primL, Tprim primR, typename Tprim::Tscal gamma) {
        return shammath::x_to_y(_x_dispatch(
            std::forward<Func>(flux_func),
            shammath::prim_y_to_x(primL),
            shammath::prim_y_to_x(primR),
            gamma));
    }

    template<class Tprim, class Func>
    inline auto _z_dispatch(
        Func &&flux_func, Tprim primL, Tprim primR, typename Tprim::Tscal gamma) {
        return shammath::x_to_z(_x_dispatch(
            std::forward<Func>(flux_func),
            shammath::prim_z_to_x(primL),
            shammath::prim_z_to_x(primR),
            gamma));
    }

    template<class Tprim, class Func>
    inline auto _mx_dispatch(
        Func &&flux_func, Tprim primL, Tprim primR, typename Tprim::Tscal gamma) {
        return shammath::invert_axis(_x_dispatch(
            std::forward<Func>(flux_func),
            shammath::prim_invert_axis(primL),
            shammath::prim_invert_axis(primR),
            gamma));
    }

    template<class Tprim, class Func>
    inline auto _my_dispatch(
        Func &&flux_func, Tprim primL, Tprim primR, typename Tprim::Tscal gamma) {
        return shammath::invert_axis(_y_dispatch(
            std::forward<Func>(flux_func),
            shammath::prim_invert_axis(primL),
            shammath::prim_invert_axis(primR),
            gamma));
    }

    template<class Tprim, class Func>
    inline auto _mz_dispatch(
        Func &&flux_func, Tprim primL, Tprim primR, typename Tprim::Tscal gamma) {
        return shammath::invert_axis(_z_dispatch(
            std::forward<Func>(flux_func),
            shammath::prim_invert_axis(primL),
            shammath::prim_invert_axis(primR),
            gamma));
    }

    // Same idea for the dust solvers, which take no gamma and use the d_-prefixed
    // rotation helpers (d_prim_y_to_x, d_x_to_y, d_invert_axis, ...).
    template<class Tprim, class Func>
    inline auto d_x_dispatch(Func &&flux_func, Tprim primL, Tprim primR) {
        return flux_func(primL, primR, typename Tprim::Tvec{1, 0, 0});
    }

    template<class Tprim, class Func>
    inline auto d_y_dispatch(Func &&flux_func, Tprim primL, Tprim primR) {
        return shammath::d_x_to_y(d_x_dispatch(
            std::forward<Func>(flux_func),
            shammath::d_prim_y_to_x(primL),
            shammath::d_prim_y_to_x(primR)));
    }

    template<class Tprim, class Func>
    inline auto d_z_dispatch(Func &&flux_func, Tprim primL, Tprim primR) {
        return shammath::d_x_to_z(d_x_dispatch(
            std::forward<Func>(flux_func),
            shammath::d_prim_z_to_x(primL),
            shammath::d_prim_z_to_x(primR)));
    }

    template<class Tprim, class Func>
    inline auto d_mx_dispatch(Func &&flux_func, Tprim primL, Tprim primR) {
        return shammath::d_invert_axis(d_x_dispatch(
            std::forward<Func>(flux_func),
            shammath::d_prim_invert_axis(primL),
            shammath::d_prim_invert_axis(primR)));
    }

    template<class Tprim, class Func>
    inline auto d_my_dispatch(Func &&flux_func, Tprim primL, Tprim primR) {
        return shammath::d_invert_axis(d_y_dispatch(
            std::forward<Func>(flux_func),
            shammath::d_prim_invert_axis(primL),
            shammath::d_prim_invert_axis(primR)));
    }

    template<class Tprim, class Func>
    inline auto d_mz_dispatch(Func &&flux_func, Tprim primL, Tprim primR) {
        return shammath::d_invert_axis(d_z_dispatch(
            std::forward<Func>(flux_func),
            shammath::d_prim_invert_axis(primL),
            shammath::d_prim_invert_axis(primR)));
    }

} // namespace

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

    auto check_gas_solver = [&](auto solver_n) {
        require_cons_equal(
            solver_n(pL, pR, gamma, Tvec{1, 0, 0}), _x_dispatch(solver_n, pL, pR, gamma));
        require_cons_equal(
            solver_n(pL, pR, gamma, Tvec{0, 1, 0}), _y_dispatch(solver_n, pL, pR, gamma));
        require_cons_equal(
            solver_n(pL, pR, gamma, Tvec{0, 0, 1}), _z_dispatch(solver_n, pL, pR, gamma));
        require_cons_equal(
            solver_n(pL, pR, gamma, Tvec{-1, 0, 0}), _mx_dispatch(solver_n, pL, pR, gamma));
        require_cons_equal(
            solver_n(pL, pR, gamma, Tvec{0, -1, 0}), _my_dispatch(solver_n, pL, pR, gamma));
        require_cons_equal(
            solver_n(pL, pR, gamma, Tvec{0, 0, -1}), _mz_dispatch(solver_n, pL, pR, gamma));
    };

    auto check_dust_solver = [&](auto solver_n) {
        require_dust_cons_equal(solver_n(dL, dR, Tvec{1, 0, 0}), d_x_dispatch(solver_n, dL, dR));
        require_dust_cons_equal(solver_n(dL, dR, Tvec{0, 1, 0}), d_y_dispatch(solver_n, dL, dR));
        require_dust_cons_equal(solver_n(dL, dR, Tvec{0, 0, 1}), d_z_dispatch(solver_n, dL, dR));
        require_dust_cons_equal(solver_n(dL, dR, Tvec{-1, 0, 0}), d_mx_dispatch(solver_n, dL, dR));
        require_dust_cons_equal(solver_n(dL, dR, Tvec{0, -1, 0}), d_my_dispatch(solver_n, dL, dR));
        require_dust_cons_equal(solver_n(dL, dR, Tvec{0, 0, -1}), d_mz_dispatch(solver_n, dL, dR));
    };

    check_gas_solver([](Tprim a, Tprim b, f64 g, Tvec n) {
        shammath::FluidStateAdiabatic<Tvec> fspec{.m_gamma = g};
        return shammath::rusanov_flux(fspec, a, b, n);
    });

    check_gas_solver([](Tprim a, Tprim b, f64 g, Tvec n) {
        shammath::FluidStateAdiabatic<Tvec> fspec{.m_gamma = g};
        return shammath::hll_flux(fspec, a, b, n);
    });

    check_gas_solver([](Tprim a, Tprim b, f64 g, Tvec n) {
        shammath::FluidStateAdiabatic<Tvec> fspec{.m_gamma = g};
        return shammath::hllc_adiab_toro_flux(fspec, a, b, n);
    });

    check_gas_solver([](Tprim a, Tprim b, f64 g, Tvec n) {
        shammath::FluidStateAdiabatic<Tvec> fspec{.m_gamma = g};
        return shammath::hllc_davis_flux(fspec, a, b, n);
    });

    check_dust_solver([](DTprim a, DTprim b, Tvec n) {
        shammath::FluidStateDust<Tvec> fspec{};
        return shammath::d_hll_flux(fspec, a, b, n);
    });

    check_dust_solver([](DTprim a, DTprim b, Tvec n) {
        shammath::FluidStateDust<Tvec> fspec{};
        return shammath::huang_bai_flux(fspec, a, b, n);
    });
}

NEW_TEST(Unittest, "shammath/flux_mirror_symmetry", 1) {

    // A face flux is computed from both of its sides (xp/xm, ...) by the godunov solver, so
    // conservation requires F(L, R, n) == -F(R, L, -n) for every pair of states, including
    // hypersonic ones (converging flows broke this for hllc_adiab_toro_flux, whose pressure
    // based wave speeds could give s_l > s_r, or S* outside of [s_l, s_r]).

    using Tvec  = f64_3;
    using Tcons = shammath::ConsState<Tvec>;
    using Tprim = shammath::PrimState<Tvec>;

    constexpr f64 gamma = 1.4;
    constexpr f64 cs2   = 0.0364; // isothermal sound speed^2 of the decaying turbulence test

    shammath::FluidStateAdiabatic<Tvec> fspec{.m_gamma = gamma};

    auto mirror_error = [&](auto solver, Tprim l, Tprim r, Tvec n) {
        Tcons f1  = solver(l, r, n);
        Tcons f2  = solver(r, l, -n);
        Tcons d   = f1 + f2;
        f64 scale = sham::max(
            1., sham::max(sham::abs(f1.rho), sycl::length(f1.rhovel)) + sham::abs(f1.rhoe));
        return sham::max(sham::max(sham::abs(d.rho), sycl::length(d.rhovel)), sham::abs(d.rhoe))
               / scale;
    };

    auto rusanov = [&](Tprim l, Tprim r, Tvec n) {
        return shammath::rusanov_flux(fspec, l, r, n);
    };
    auto hll = [&](Tprim l, Tprim r, Tvec n) {
        return shammath::hll_flux(fspec, l, r, n);
    };
    auto hllc_toro = [&](Tprim l, Tprim r, Tvec n) {
        return shammath::hllc_adiab_toro_flux(fspec, l, r, n);
    };
    auto hllc_davis = [&](Tprim l, Tprim r, Tvec n) {
        return shammath::hllc_davis_flux(fspec, l, r, n);
    };

    constexpr f64 eps = 1e-12;

    // unit normal of an axis, and the state with its x velocity component moved along that
    // axis (so that the converging/diverging velocity is the normal one)
    auto axis_normal = [](u32 axis) {
        return Tvec{f64(axis == 0), f64(axis == 1), f64(axis == 2)};
    };
    auto along_axis = [](Tprim p, u32 axis) {
        Tvec v = p.vel;
        if (axis == 1) {
            p.vel = Tvec{v.z(), v.x(), v.y()};
        } else if (axis == 2) {
            p.vel = Tvec{v.y(), v.z(), v.x()};
        }
        return p;
    };

    // Regressions, faces where the plain equation (10.59) PVRS wave speeds break the symmetry
    const f64 mach2_vel = 2. * sycl::sqrt(gamma);
    const std::array<std::pair<Tprim, Tprim>, 3> regression_faces{{
        // a face of the Mach 8 decaying turbulence run (s_l > s_r, mismatch of the mass flux
        // ~ 9 between the two sides)
        {Tprim{.rho = 0.0765, .press = cs2 * 0.0765, .vel = Tvec{5.9, 0.3, -0.2}},
         Tprim{.rho = 3.904, .press = cs2 * 3.904, .vel = Tvec{-2.2, -0.1, 0.4}}},
        // symmetric Mach 2 collision (s_l > 0 > s_r: each side sees itself upwind and only
        // lets mass out, so no shock forms and the mass vanishes at the face)
        {Tprim{.rho = 1., .press = 1., .vel = Tvec{mach2_vel, 0., 0.}},
         Tprim{.rho = 1., .press = 1., .vel = Tvec{-mach2_vel, 0., 0.}}},
        // ordered s_l < s_r but S* outside of [s_l, s_r] (mass flux +2.78 vs -2.76)
        {Tprim{.rho = 1.049, .press = 0.021, .vel = Tvec{2.65, 0., 0.}},
         Tprim{.rho = 5.153, .press = 3.842, .vel = Tvec{-0.86, 0., 0.}}},
    }};
    for (const auto &[l, r] : regression_faces) {
        for (u32 axis = 0; axis < 3; axis++) {
            Tvec n   = axis_normal(axis);
            Tprim la = along_axis(l, axis), ra = along_axis(r, axis);
            REQUIRE_FLOAT_EQUAL(mirror_error(hllc_toro, la, ra, n), 0., eps);
            REQUIRE_FLOAT_EQUAL(mirror_error(hllc_davis, la, ra, n), 0., eps);
            REQUIRE_FLOAT_EQUAL(mirror_error(hll, la, ra, n), 0., eps);
            REQUIRE_FLOAT_EQUAL(mirror_error(rusanov, la, ra, n), 0., eps);
        }
    }

    // Sweep: subsonic to hypersonic converging/diverging pairs, large density contrasts,
    // isothermal and hot pressures
    const f64 rhos[]  = {1e-3, 0.0765, 0.3, 1., 3.904, 50.};
    const f64 vels[]  = {-8., -2.2, -0.5, 0., 0.3, 1., 5.9, 8.};
    const f64 temps[] = {cs2, 1.};

    std::vector<Tprim> states;
    u32 k = 0;
    for (f64 rho : rhos) {
        for (f64 v : vels) {
            for (f64 T : temps) {
                f64 vt = 0.1 * f64(k % 7) - 0.3; // tangential velocity
                states.push_back(Tprim{.rho = rho, .press = T * rho, .vel = Tvec{v, vt, -vt}});
                k++;
            }
        }
    }

    f64 err_rusanov = 0, err_hll = 0, err_toro = 0, err_davis = 0;
    for (const Tprim &l : states) {
        for (const Tprim &r : states) {
            for (u32 axis = 0; axis < 3; axis++) {
                Tvec n   = axis_normal(axis);
                Tprim la = along_axis(l, axis), ra = along_axis(r, axis);
                err_rusanov = sham::max(err_rusanov, mirror_error(rusanov, la, ra, n));
                err_hll     = sham::max(err_hll, mirror_error(hll, la, ra, n));
                err_toro    = sham::max(err_toro, mirror_error(hllc_toro, la, ra, n));
                err_davis   = sham::max(err_davis, mirror_error(hllc_davis, la, ra, n));
            }
        }
    }

    REQUIRE_FLOAT_EQUAL_NAMED("rusanov mirror symmetry", err_rusanov, 0., eps);
    REQUIRE_FLOAT_EQUAL_NAMED("hll mirror symmetry", err_hll, 0., eps);
    REQUIRE_FLOAT_EQUAL_NAMED("hllc toro mirror symmetry", err_toro, 0., eps);
    REQUIRE_FLOAT_EQUAL_NAMED("hllc davis mirror symmetry", err_davis, 0., eps);
}

NEW_TEST(Unittest, "shammath/fluid_state_barotropic", 1) {

    using Tvec  = f64_3;
    using Tcons = shammath::ConsState<Tvec>;
    using Tprim = shammath::PrimState<Tvec>;

    constexpr f64 rho_c = 1e-2;
    constexpr f64 cs0   = 0.3;
    constexpr f64 gamma = 5. / 3.;

    shammath::FluidStateBarotropic<Tvec> fspec{rho_c, cs0, gamma};

    REQUIRE_FLOAT_EQUAL(fspec.rho_crit(), rho_c, 1e-15);
    REQUIRE_FLOAT_EQUAL(fspec.cs0(), cs0, 1e-15);
    REQUIRE_FLOAT_EQUAL(fspec.gamma_adiab(), gamma, 1e-15);

    auto prim_at = [&](f64 rho, Tvec vel = Tvec{}) {
        return Tprim{.rho = rho, .press = fspec.pressure(rho), .vel = vel};
    };

    // isothermal far below rho_c, polytropic far above it, twice isothermal at rho_c
    {
        const f64 rho_lo = 1e-12 * rho_c;
        const f64 rho_hi = 1e12 * rho_c;
        REQUIRE_FLOAT_EQUAL(fspec.pressure(rho_lo) / (cs0 * cs0 * rho_lo), 1., 1e-6);
        REQUIRE_FLOAT_EQUAL(fspec.pressure(rho_c) / (cs0 * cs0 * rho_c), 2., 1e-14);
        REQUIRE_FLOAT_EQUAL(
            fspec.pressure(rho_hi) / (cs0 * cs0 * rho_c * sycl::pow(rho_hi / rho_c, gamma)),
            1.,
            1e-6);

        REQUIRE_FLOAT_EQUAL(fspec.sound_speed(prim_at(rho_lo)), cs0, 1e-6);
        REQUIRE_FLOAT_EQUAL(fspec.gamma(prim_at(rho_lo)), 1., 1e-6);
        REQUIRE_FLOAT_EQUAL(fspec.gamma(prim_at(rho_c)), 0.5 * (1 + gamma), 1e-14);
        REQUIRE_FLOAT_EQUAL(fspec.gamma(prim_at(rho_hi)), gamma, 1e-6);
    }

    // sound speed is sqrt(dP/drho) and gamma(prim) is rho cs^2 / P, on both sides of rho_c
    for (f64 rho : {1e-3 * rho_c, 0.5 * rho_c, rho_c, 3. * rho_c, 1e3 * rho_c}) {
        const f64 h       = 1e-6 * rho;
        const f64 dp_drho = (fspec.pressure(rho + h) - fspec.pressure(rho - h)) / (2. * h);
        const Tprim p     = prim_at(rho);
        const f64 cs      = fspec.sound_speed(p);
        REQUIRE_FLOAT_EQUAL(cs * cs / dp_drho, 1., 1e-8);
        REQUIRE_FLOAT_EQUAL(fspec.gamma(p), rho * cs * cs / p.press, 1e-12);
    }

    // cons_to_prim ignores rhoe and recovers the barotropic pressure
    {
        const Tprim p = prim_at(4. * rho_c, Tvec{0.3, -0.2, 0.5});
        Tcons c       = fspec.prim_to_cons(p);
        c.rhoe *= 7.; // arbitrary energy, must not change the pressure
        const Tprim p2 = fspec.cons_to_prim(c);
        REQUIRE_FLOAT_EQUAL(p2.rho, p.rho, 1e-15);
        REQUIRE_FLOAT_EQUAL(p2.press, p.press, 1e-15);
        REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED("", p2.vel, p.vel, 1e-15, sycl::length);
    }

    // every gas solver accepts the spec, and is consistent and conservative with it:
    // F(L, L, n) is the physical flux and F(L, R, n) == -F(R, L, -n)
    const Tprim pL = prim_at(0.5 * rho_c, Tvec{0.3, -0.2, 0.5});
    const Tprim pR = prim_at(20. * rho_c, Tvec{-0.1, 0.4, -0.3});
    const Tvec n   = sycl::normalize(Tvec{1., 2., -0.5});

    constexpr f64 eps       = 1e-13;
    auto require_cons_equal = [&](Tcons lhs, Tcons rhs) {
        REQUIRE_FLOAT_EQUAL(lhs.rho, rhs.rho, eps);
        REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED("", lhs.rhovel, rhs.rhovel, eps, sycl::length);
        REQUIRE_FLOAT_EQUAL(lhs.rhoe, rhs.rhoe, eps);
    };

    auto check_gas_solver = [&](auto solver_n) {
        require_cons_equal(solver_n(fspec, pL, pL, n), fspec.flux(pL, n));
        require_cons_equal(solver_n(fspec, pR, pR, n), fspec.flux(pR, n));
        require_cons_equal(solver_n(fspec, pL, pR, n), -1. * solver_n(fspec, pR, pL, -n));
    };

    check_gas_solver([](auto f, Tprim a, Tprim b, Tvec n) {
        return shammath::rusanov_flux(f, a, b, n);
    });
    check_gas_solver([](auto f, Tprim a, Tprim b, Tvec n) {
        return shammath::hll_flux(f, a, b, n);
    });
    check_gas_solver([](auto f, Tprim a, Tprim b, Tvec n) {
        return shammath::hllc_adiab_toro_flux(f, a, b, n);
    });
    check_gas_solver([](auto f, Tprim a, Tprim b, Tvec n) {
        return shammath::hllc_davis_flux(f, a, b, n);
    });
}
