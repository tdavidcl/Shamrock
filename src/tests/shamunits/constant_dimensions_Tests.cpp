// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/shamtest.hpp"
#include "shamunits/Constants.hpp"
#include "shamunits/UnitSystem.hpp"

NEW_TEST(Unittest, "shamunits/constant-dimensions", 1) {
    using namespace shamunits;

    // time in Myr, length in au, mass in solar masses
    UnitSystem<f64> si{};
    UnitSystem<f64> astro{
        si.get<mega, units::years>(),
        si.get<units::astronomical_unit>(),
        si.get<units::kilogram>() * Constants<f64>(si).sol_mass(),
    };

    using Si = Constants<f64>::Si;
    Constants<f64> astro_ctes{astro};

    // value of one J.s and one kg.m-3 in the astro system, computed by hand
    f64 joule_second = astro.get<units::Joule>() * astro.get<units::second>();
    f64 kg_per_m3    = astro.get<units::kilogram>() * astro.get<units::metre, -3>();

    REQUIRE_FLOAT_EQUAL(astro_ctes.h() / Si::h / joule_second, 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(astro_ctes.hbar() / Si::hbar / joule_second, 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(astro_ctes.guiness_density() / Si::guiness_density / kg_per_m3, 1., 1e-12);
}
