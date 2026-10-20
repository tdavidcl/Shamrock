// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file unit_algebra_Tests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Compile-time checks of the Unit / Dimension algebra (shamunits/Unit.hpp)
 *
 */

#include "shamtest/shamtest.hpp"
#include "shamunits/Unit.hpp"

namespace {
    using namespace shamunits;

    // Local stand-ins: the real tables (shamunits::units) arrive in a later PR
    constexpr Unit s  = base_unit({.second = 1});
    constexpr Unit m  = base_unit({.metre = 1});
    constexpr Unit kg = base_unit({.kilogram = 1});

    constexpr Unit newton = kg * m / upow(s, 2);
    constexpr Unit joule  = newton * m;
    constexpr Unit watt   = joule / s;
    constexpr Unit au     = 149597870700. * m;

    /// Relative comparison: factors built two different ways may differ by a few ulps
    constexpr bool close(double a, double b, double rel) {
        double d  = a - b;
        double ab = (b < 0) ? -b : b;
        return ((d < 0) ? -d : d) <= rel * ab;
    }

    // Dimension algebra
    static_assert(newton.dim == Dimension{.second = -2, .metre = 1, .kilogram = 1});
    static_assert(joule.dim == Dimension{.second = -2, .metre = 2, .kilogram = 1});
    static_assert(watt.dim == (joule / s).dim);
    static_assert((1. / s).dim == Dimension{.second = -1});
    static_assert(
        dim_pow(Dimension{.metre = 1, .ampere = -2}, 3) == Dimension{.metre = 3, .ampere = -6});

    // Factors
    static_assert(base_unit({.kelvin = 1}).si_factor == 1.);
    static_assert(scale(1e6).dim == Dimension{});
    static_assert((scale(1e3) * m).si_factor == 1e3);
    static_assert((m * 1e3) == (1e3 * m));
    static_assert((joule / 1e7).si_factor == 1e-7);
    static_assert((2. / s).si_factor == 2.);

    // upow: zero, positive, negative; exact for power-of-two factors
    static_assert(upow(au, 0) == Unit{});
    static_assert(upow(au, 1) == au);
    static_assert(upow(au, 2).si_factor == 149597870700. * 149597870700.);
    static_assert(
        upow(Unit{.dim = {.metre = 1}, .si_factor = 2.}, -2)
        == Unit{.dim = {.metre = -2}, .si_factor = .25});
    static_assert(upow(Unit{.dim = {.metre = 1}, .si_factor = 2.}, 10).si_factor == 1024.);

    // upow composes like a power, up to rounding for non-power-of-two factors
    static_assert((upow(au, 2) * upow(au, 3)).dim == upow(au, 5).dim);
    static_assert(close((upow(au, 2) * upow(au, 3)).si_factor, upow(au, 5).si_factor, 1e-15));
    static_assert(close((upow(au, -2) * upow(au, 3)).si_factor, au.si_factor, 1e-15));

    // A composition a single scalar power cannot express: m.s-2 * kg = N
    static_assert((m / upow(s, 2) * kg) == newton);

    // Unit is a structural type: by-value template parameter, prvalue argument, and its fields
    // usable as template arguments. This is the property UnitSystem::get<u>() relies on; it
    // compiles on clang 15 (only a raw template<double> is rejected before clang 18).
    template<Unit u>
    struct UnitNttp {
        static constexpr Unit value          = u;
        static constexpr int metre_exponent  = u.dim.metre;
        static constexpr double factor_value = u.si_factor;
    };
    template<int n>
    struct IntNttp {
        static constexpr int value = n;
    };
    static_assert(UnitNttp<au>::value == au);
    static_assert(UnitNttp<upow(au, 2)>::metre_exponent == 2);
    static_assert(UnitNttp<scale(1e6) * s>::factor_value == 1e6);
    static_assert(IntNttp<UnitNttp<joule>::value.dim.second>::value == -2);

} // namespace

NEW_TEST(Unittest, "shamunits/unit_algebra", 1) {
    // the static_asserts above are the test; this keeps it visible in the test list
    REQUIRE_EQUAL(joule.dim.metre, 2);
}
