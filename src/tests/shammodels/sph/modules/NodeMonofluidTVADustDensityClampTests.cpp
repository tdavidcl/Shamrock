// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/sph/math/dust_variables.hpp"
#include "shammodels/sph/modules/NodeMonofluidTVADustDensityClamp.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

NEW_TEST(Unittest, "shammodels/sph/modules/NodeMonofluidTVADustDensityClamp", 1) {
    using Tvec  = f64_3;
    using Tscal = f64;
    using namespace shamrock;
    using namespace shammodels::sph::modules;

    u32 ndust = 2;
    u32 N     = 3;

    // rho(h) = pmass * (hfactd / h)^3 = 1.0 * (1.2 / 1.0)^3 = 1.728 for every particle
    // clamp_frac is the max dust-to-gas ratio eps_max = 0.99 (dimensionless, not scaled by rho_a)
    Tscal pmass  = 1.0;
    Tscal hfactd = 1.2;
    Tscal tol    = 1e-9;

    auto part_counts = std::make_shared<solvergraph::Indexes<u32>>("", "");
    part_counts->indexes.add_obj(0, u32{N});

    auto gpart_mass  = solvergraph::IDataEdge<Tscal>::make_shared("", "");
    gpart_mass->data = pmass;

    auto hfactd_edge  = solvergraph::IDataEdge<Tscal>::make_shared("", "");
    hfactd_edge->data = hfactd;

    auto clamp_frac_edge  = solvergraph::IDataEdge<Tscal>::make_shared("", "");
    clamp_frac_edge->data = 0.99;

    auto hpart = std::make_shared<solvergraph::Field<Tscal>>(1, "hpart", "h");
    auto s_j   = std::make_shared<solvergraph::Field<Tscal>>(ndust, "s_j", "s_j");

    hpart->ensure_sizes(part_counts->indexes);
    s_j->ensure_sizes(part_counts->indexes);

    hpart->get_buf(0).copy_from_stdvec({1.0, 1.0, 1.0});

    // particle 0 : well below the threshold, left untouched
    // particle 1 : species 0 alone exceeds the threshold (species 1 is 0), individual clamp only
    // particle 2 : each species has eps = 0.6 individually (< eps_max = 0.99), but their sum
    //              (1.2) does -> rescaled in pass 2
    s_j->get_buf(0).copy_from_stdvec({0.1, 0.2, 2.0, 0.0, 1.0182337649086284, 1.0182337649086284});

    NodeMonofluidTVADustDensityClamp<Tvec> node(ndust);
    node.set_edges(part_counts, gpart_mass, hfactd_edge, clamp_frac_edge, hpart, s_j);
    node.evaluate();

    std::vector<Tscal> expected
        = {0.1, 0.2, 1.3079449529701164, 0.0, 0.9248567456638892, 0.9248567456638892};

    std::vector<Tscal> got = s_j->get_buf(0).copy_to_stdvec();
    for (u32 i = 0; i < expected.size(); i++) {
        REQUIRE_FLOAT_EQUAL_NAMED(sham::format("s_j[{}]", i), got[i], expected[i], tol);
    }
}

NEW_TEST(Unittest, "shammodels/sph/modules/NodeMonofluidTVADustDensityClamp_dust_variables", 1) {
    using Tvec  = f64_3;
    using Tscal = f64;
    using namespace shamrock;
    using namespace shammodels::sph;
    using namespace shammodels::sph::modules;

    u32 ndust = 2;
    u32 N     = 3;

    // rho(h) = pmass * (hfactd / h)^3 = 1.728 for every particle
    Tscal pmass  = 1.0;
    Tscal hfactd = 1.2;
    Tscal rho    = 1.728;
    Tscal tol    = 1e-9;

    // same dust fractions as the test above, expressed directly as eps_j
    // particle 0 : below the threshold, untouched
    // particle 1 : species 0 alone is set just below 1 (the B variable cannot represent
    //              eps >= 1), individual clamp only
    // particle 2 : 0.6 + 0.6 > 0.99, rescaled in pass 2
    std::vector<Tscal> eps_in       = {0.1 * 0.1 / rho, 0.2 * 0.2 / rho, 0.999, 0.0, 0.6, 0.6};
    std::vector<Tscal> eps_expected = {0.1 * 0.1 / rho, 0.2 * 0.2 / rho, 0.99, 0.0, 0.495, 0.495};

    for (DustVariable v :
         {DustVariable::SqrtRhoEps, DustVariable::Eps, DustVariable::SqrtEpsOverOneMinusEps}) {

        auto part_counts = std::make_shared<solvergraph::Indexes<u32>>("", "");
        part_counts->indexes.add_obj(0, u32{N});

        auto gpart_mass  = solvergraph::IDataEdge<Tscal>::make_shared("", "");
        gpart_mass->data = pmass;

        auto hfactd_edge  = solvergraph::IDataEdge<Tscal>::make_shared("", "");
        hfactd_edge->data = hfactd;

        auto clamp_frac_edge  = solvergraph::IDataEdge<Tscal>::make_shared("", "");
        clamp_frac_edge->data = 0.99;

        auto hpart = std::make_shared<solvergraph::Field<Tscal>>(1, "hpart", "h");
        auto X_j   = std::make_shared<solvergraph::Field<Tscal>>(ndust, "X_j", "X_j");

        hpart->ensure_sizes(part_counts->indexes);
        X_j->ensure_sizes(part_counts->indexes);

        hpart->get_buf(0).copy_from_stdvec({1.0, 1.0, 1.0});

        std::vector<Tscal> X_in(eps_in.size());
        for (u32 i = 0; i < eps_in.size(); i++) {
            X_in[i] = dust_var_from_eps(v, eps_in[i], rho);
        }
        X_j->get_buf(0).copy_from_stdvec(X_in);

        NodeMonofluidTVADustDensityClamp<Tvec> node(ndust, v);
        node.set_edges(part_counts, gpart_mass, hfactd_edge, clamp_frac_edge, hpart, X_j);
        node.evaluate();

        std::vector<Tscal> got = X_j->get_buf(0).copy_to_stdvec();
        for (u32 i = 0; i < eps_expected.size(); i++) {
            REQUIRE_FLOAT_EQUAL_NAMED(
                sham::format("{} eps_j[{}]", dust_variable_to_string(v), i),
                dust_var_to_eps(v, got[i], rho),
                eps_expected[i],
                tol);
        }
    }
}
