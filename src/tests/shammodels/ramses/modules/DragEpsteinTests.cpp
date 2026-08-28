// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DragEpsteinTests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Tests of the drag rate (inverse stopping time) nodes of the Ramses solver
 *
 */

#include "shammath/riemann.hpp"
#include "shammodels/ramses/modules/NodeSetDustAlphas.hpp"
#include "shamphys/Dust.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

namespace {

    using Tvec  = f64_3;
    using Tscal = f64;

    /// Reference drag rate, computed on the host from the primitive gas state
    Tscal ref_alpha(
        Tscal rho_grain, Tscal s_grain, Tscal rho, Tscal press, Tscal gamma, Tscal f = 1.0) {
        Tscal cs = sycl::sqrt(gamma * press / rho);
        return 1.0 / shamphys::epstein_stopping_time(rho_grain, s_grain, rho, cs, gamma, f);
    }

    /// Build the conservative energy of a cell from its primitive state
    Tscal rhoe_of(Tscal rho, Tvec vel, Tscal press, Tscal gamma) {
        return press / (gamma - 1.0) + 0.5 * rho * sycl::dot(vel, vel);
    }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/modules/DragEpstein", 1) {
    using namespace shamrock;
    using namespace shammodels::basegodunov::modules;

    const u32 ndust   = 2;
    const u32 N       = 3; // cells
    const Tscal gamma = 1.4;

    // ---- physical setup: three cells with different gas states -------------------------------
    std::vector<Tscal> rho   = {1.0, 4.0, 0.25};
    std::vector<Tvec> vel    = {{1.0, 0.0, 0.0}, {0.0, -2.0, 1.0}, {0.5, 0.5, 0.5}};
    std::vector<Tscal> press = {1.0, 3.0, 0.5};

    std::vector<Tscal> grains_sizes     = {1e-2, 5e-2};
    std::vector<Tscal> grains_densities = {2.0, 3.0};

    // ---- edges -------------------------------------------------------------------------------
    auto counts = std::make_shared<solvergraph::Indexes<u32>>("", "");
    counts->indexes.add_obj(0, u32{N});

    auto f_rho    = std::make_shared<solvergraph::Field<Tscal>>(1, "rho", "\\rho");
    auto f_rhov   = std::make_shared<solvergraph::Field<Tvec>>(1, "rhov", "(\\rho v)");
    auto f_rhoe   = std::make_shared<solvergraph::Field<Tscal>>(1, "rhoe", "(\\rho e)");
    auto f_rho_d  = std::make_shared<solvergraph::Field<Tscal>>(ndust, "rho_d", "\\rho_d");
    auto f_rhov_d = std::make_shared<solvergraph::Field<Tvec>>(ndust, "rhov_d", "(\\rho_d v)");
    auto f_alphas = std::make_shared<solvergraph::Field<Tscal>>(ndust, "alphas", "\\alpha");

    for (auto &f : {f_rho, f_rhoe}) {
        f->ensure_sizes(counts->indexes);
    }
    f_rhov->ensure_sizes(counts->indexes);
    f_rho_d->ensure_sizes(counts->indexes);
    f_rhov_d->ensure_sizes(counts->indexes);
    f_alphas->ensure_sizes(counts->indexes);

    std::vector<Tscal> rhoe_v;
    std::vector<Tvec> rhov_v;
    for (u32 i = 0; i < N; i++) {
        rhov_v.push_back(rho[i] * vel[i]);
        rhoe_v.push_back(rhoe_of(rho[i], vel[i], press[i], gamma));
    }

    f_rho->get_buf(0).copy_from_stdvec(rho);
    f_rhov->get_buf(0).copy_from_stdvec(rhov_v);
    f_rhoe->get_buf(0).copy_from_stdvec(rhoe_v);

    // dust states, only read by the supersonic correction
    std::vector<Tscal> rho_d_v(N * ndust, 1.0);
    std::vector<Tvec> rhov_d_v(N * ndust, Tvec{0., 0., 0.});
    f_rho_d->get_buf(0).copy_from_stdvec(rho_d_v);
    f_rhov_d->get_buf(0).copy_from_stdvec(rhov_d_v);

    // ---- subsonic (f = 1) --------------------------------------------------------------------
    {
        NodeSetDustAlphasEpstein<Tvec> node(1, ndust, gamma, false, grains_sizes, grains_densities);
        node.set_edges(counts, f_rho, f_rhov, f_rhoe, f_rho_d, f_rhov_d, f_alphas);
        node.evaluate();

        std::vector<Tscal> got = f_alphas->get_buf(0).copy_to_stdvec();

        REQUIRE_EQUAL(got.size(), std::size_t(N * ndust));

        for (u32 i = 0; i < N; i++) {
            for (u32 j = 0; j < ndust; j++) {
                Tscal expected
                    = ref_alpha(grains_densities[j], grains_sizes[j], rho[i], press[i], gamma);
                REQUIRE_FLOAT_EQUAL(got[i * ndust + j], expected, 1e-12);
            }
        }
    }

    // ---- supersonic correction ---------------------------------------------------------------
    {
        // give the dust a velocity so that delta v is non zero
        std::vector<Tvec> vel_d = {{0.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {-1.0, 2.0, 0.0}};
        for (u32 i = 0; i < N; i++) {
            for (u32 j = 0; j < ndust; j++) {
                rhov_d_v[i * ndust + j] = rho_d_v[i * ndust + j] * vel_d[i];
            }
        }
        f_rhov_d->get_buf(0).copy_from_stdvec(rhov_d_v);

        NodeSetDustAlphasEpstein<Tvec> node(1, ndust, gamma, true, grains_sizes, grains_densities);
        node.set_edges(counts, f_rho, f_rhov, f_rhoe, f_rho_d, f_rhov_d, f_alphas);
        node.evaluate();

        std::vector<Tscal> got = f_alphas->get_buf(0).copy_to_stdvec();

        for (u32 i = 0; i < N; i++) {
            Tscal cs = sycl::sqrt(gamma * press[i] / rho[i]);
            Tscal dv = sycl::length(vel_d[i] - vel[i]);
            Tscal f  = shamphys::epstein_supersonic_correction(dv, cs);

            for (u32 j = 0; j < ndust; j++) {
                Tscal expected
                    = ref_alpha(grains_densities[j], grains_sizes[j], rho[i], press[i], gamma, f);
                REQUIRE_FLOAT_EQUAL(got[i * ndust + j], expected, 1e-12);
            }
        }
    }

    // ---- pressureless cell: no drag rather than a division by zero ---------------------------
    {
        std::vector<Tscal> rhoe_cold = rhoe_v;
        // cell 1 has no internal energy left, hence cs = 0
        rhoe_cold[1] = 0.5 * rho[1] * sycl::dot(vel[1], vel[1]);
        f_rhoe->get_buf(0).copy_from_stdvec(rhoe_cold);

        NodeSetDustAlphasEpstein<Tvec> node(1, ndust, gamma, false, grains_sizes, grains_densities);
        node.set_edges(counts, f_rho, f_rhov, f_rhoe, f_rho_d, f_rhov_d, f_alphas);
        node.evaluate();

        std::vector<Tscal> got = f_alphas->get_buf(0).copy_to_stdvec();

        for (u32 j = 0; j < ndust; j++) {
            REQUIRE_EQUAL(got[1 * ndust + j], Tscal(0));
        }
    }
}

NEW_TEST(Unittest, "shammodels/ramses/modules/DragAlphasConstant", 1) {
    using namespace shamrock;
    using namespace shammodels::basegodunov::modules;

    const u32 ndust = 3;
    const u32 N     = 2;

    auto counts = std::make_shared<solvergraph::Indexes<u32>>("", "");
    counts->indexes.add_obj(0, u32{N});

    auto f_alphas = std::make_shared<solvergraph::Field<Tscal>>(ndust, "alphas", "\\alpha");
    f_alphas->ensure_sizes(counts->indexes);

    std::vector<Tscal> alphas = {1.0, 10.0, 100.0};

    NodeSetDustAlphasConstant<Tscal> node(1, ndust, alphas);
    node.set_edges(counts, f_alphas);
    node.evaluate();

    // the same per species values are broadcast to every cell
    std::vector<Tscal> expected = {1.0, 10.0, 100.0, 1.0, 10.0, 100.0};
    REQUIRE_EQUAL(f_alphas->get_buf(0).copy_to_stdvec(), expected);
}
