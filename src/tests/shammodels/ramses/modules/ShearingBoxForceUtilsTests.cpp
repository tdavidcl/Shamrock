// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ShearingBoxForceUtilsTests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Unit tests of the time integration of the Ramses shearing box forces
 */

#include "shambase/constants.hpp"
#include "shammodels/ramses/modules/ShearingBoxForceUtils.hpp"
#include "shamtest/shamtest.hpp"
#include <cmath>

using namespace shammodels::basegodunov::modules;

NEW_TEST(Unittest, "shammodels/ramses/modules/ShearingBoxForceUtils/equilibrium", 1) {
    // without fluxes, the equilibrium shear flow v_y = -q Omega x' - eta (v_x = 0) is a fixed
    // point of the update, for any time step
    const f64 Omega = 1.3, q = 1.5, eta = 0.05;

    for (f64 dt : {1e-3, 0.1, 2.0}) {
        for (f64 xp : {-0.7, 0., 0.3}) {
            for (f64 rho : {0.5, 2.0}) {
                f64 g_x  = 2 * q * Omega * Omega * xp + 2 * Omega * eta;
                f64_3 p0 = {0., rho * (-q * Omega * xp - eta), rho * 0.25};

                f64_3 p1 = shearing_box_momentum_update<f64_3>(p0, rho, g_x, Omega, dt);

                REQUIRE_FLOAT_EQUAL(p1.x(), 0., 1e-14);
                REQUIRE_FLOAT_EQUAL(p1.y(), p0.y(), 1e-14);
                REQUIRE_EQUAL(p1.z(), p0.z());

                f64 de = shearing_box_energy_update<f64_3>(p0, p1, rho);
                REQUIRE_FLOAT_EQUAL(de, 0., 1e-14);
            }
        }
    }
}

NEW_TEST(Unittest, "shammodels/ramses/modules/ShearingBoxForceUtils/coriolis_rotation", 1) {
    // without tidal term (q = 0) nor fluxes, the momentum rotates at 2 Omega (clockwise in the
    // (x, y) plane): the norm is kept to round-off and the phase is second order accurate
    const f64 Omega = 1.0;
    const f64 rho   = 1.7;
    const f64 T     = shambase::constants::pi<f64> / Omega; // period of the rotation at 2 Omega

    f64 prev_err = 0;
    for (u32 nstep : {64_u32, 128_u32, 256_u32}) {
        f64 dt  = T / nstep;
        f64_3 p = {rho * 0.3, rho * -0.1, rho * 0.5};
        f64 n0  = sycl::length(p);

        for (u32 i = 0; i < nstep; i++) {
            p = shearing_box_momentum_update<f64_3>(p, rho, 0., Omega, dt);
        }

        // norm kept (the Coriolis force does no work)
        REQUIRE_FLOAT_EQUAL(sycl::length(p), n0, 1e-13);
        REQUIRE_EQUAL(p.z(), rho * 0.5);

        // back to the initial state after one period, with a second order phase error
        f64 err = sycl::length(p - f64_3{rho * 0.3, rho * -0.1, rho * 0.5});
        REQUIRE(err < 1e-2);
        if (prev_err > 0) {
            f64 order = std::log2(prev_err / err);
            REQUIRE_FLOAT_EQUAL(order, 2., 0.05);
        }
        prev_err = err;
    }

    // rotation sense over a small step: a_x = 2 Omega v_y, a_y = -2 Omega v_x
    f64 dt   = 1e-4;
    f64_3 p0 = {0., rho, 0.};
    f64_3 p1 = shearing_box_momentum_update<f64_3>(p0, rho, 0., Omega, dt);
    REQUIRE_FLOAT_EQUAL(p1.x(), 2 * Omega * rho * dt, 1e-10);

    p0 = {rho, 0., 0.};
    p1 = shearing_box_momentum_update<f64_3>(p0, rho, 0., Omega, dt);
    REQUIRE_FLOAT_EQUAL(p1.y(), -2 * Omega * rho * dt, 1e-10);
}

NEW_TEST(Unittest, "shammodels/ramses/modules/ShearingBoxForceUtils/tidal_work", 1) {
    // the energy update is the kinetic energy change of the kick. Without fluxes and at constant
    // density it is exactly the trapezoidal work of the radial force, dt/2 g_x (p^n_x + p^{n+1}_x)
    // (the Coriolis term, antisymmetric, does no work in the implicit midpoint average)
    const f64 Omega = 1.0, q = 1.5, rho = 1.2, xp = 0.4;
    const f64 g_x = 2 * q * Omega * Omega * xp;

    f64_3 p0 = {rho * 0.2, rho * 0.1, 0.};
    for (f64 dt : {1e-2, 0.1, 1.0}) {
        f64_3 p1 = shearing_box_momentum_update<f64_3>(p0, rho, g_x, Omega, dt);
        f64 de   = shearing_box_energy_update<f64_3>(p0, p1, rho);

        REQUIRE_FLOAT_EQUAL(de, (sycl::dot(p1, p1) - sycl::dot(p0, p0)) / (2 * rho), 1e-14);

        f64 work = dt / 2 * g_x * (p0.x() + p1.x());
        REQUIRE_FLOAT_EQUAL(de, work, 1e-14);
    }
}

NEW_TEST(Unittest, "shammodels/ramses/modules/ShearingBoxForceUtils/trapezoidal_system", 1) {
    // check the closed form against the residual of the trapezoidal system
    const f64 Omega = 0.8, dt = 0.3, g_x = 0.7, rho_np1 = 1.3;
    f64_3 p_star = {0.25, -0.35, 0.05};

    f64_3 p1 = shearing_box_momentum_update<f64_3>(p_star, rho_np1, g_x, Omega, dt);

    auto S = [&](f64 rho, f64_3 p) {
        return f64_3{rho * g_x + 2 * Omega * p.y(), -2 * Omega * p.x(), 0.};
    };
    f64_3 residual = p1 - (p_star + dt / 2 * (S(rho_np1, p_star) + S(rho_np1, p1)));

    REQUIRE_FLOAT_EQUAL(sycl::length(residual), 0., 1e-15);
}

NEW_TEST(Unittest, "shammodels/ramses/modules/ShearingBoxForceUtils/epicycle_map", 1) {
    // Velocity perturbation (u, w) = (v_x, v_y + q Omega x') of a uniform flow: the explicit
    // fluxes add the advection of the background shear, w += q Omega dt u, then the forces
    // are applied. The step map must have a unit determinant (no damping of the epicycles) and
    // oscillate at kappa^2 = 2 (2 - q) Omega^2.
    const f64 Omega = 1.0, q = 1.5, rho = 1.0;
    const f64 kappa = sycl::sqrt(2 * (2 - q)) * Omega;

    f64 prev_err = 0;
    for (u32 nstep : {200_u32, 400_u32, 800_u32}) {
        const f64 T  = 2 * shambase::constants::pi<f64> / kappa;
        const f64 dt = T / nstep;

        f64 u = 0.01, w = 0.; // v_x and v_y deviation from the shear flow
        f64 a0 = u;
        for (u32 i = 0; i < nstep; i++) {
            // the forces only see v_y = w - q Omega x', take x' = 0 (the map is the same for
            // any x' since the tidal term balances the shear)
            f64_3 p_star = {rho * u, rho * (w + q * Omega * dt * u), 0.};
            f64_3 p1     = shearing_box_momentum_update<f64_3>(p_star, rho, 0., Omega, dt);
            u            = p1.x() / rho;
            w            = p1.y() / rho;
        }

        // amplitude of the epicycle invariant u^2 + (2 Omega / kappa)^2 w^2 after one period
        f64 amp = sycl::sqrt(u * u + (4 * Omega * Omega / (kappa * kappa)) * w * w);
        REQUIRE_FLOAT_EQUAL(amp, a0, 1e-3 * a0);

        // back to the initial state after one period (frequency error)
        f64 err = sycl::sqrt((u - a0) * (u - a0) + w * w) / a0;
        REQUIRE(err < 2e-3);
        if (prev_err > 0) {
            f64 order = std::log2(prev_err / err);
            REQUIRE(order > 1.9); // second order
        }
        prev_err = err;
    }
}
