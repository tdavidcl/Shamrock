// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file codegen_probe.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Codegen probe for the shamunits redesign
 *
 * Standalone C++20 program implementing enough of the planned design to read
 * the emitted assembly for each get() call shape. Not built by CMake;
 * compile by hand or paste into godbolt. See doc/redesign/README.md.
 */

// =====================================================================
// shamunits redesign -- codegen probe (standalone, C++20, godbolt-ready)
//
//   Compile with:  -std=c++20 -O2      (also try -O3, and clang vs gcc)
//
// Implements just enough of the planned design for test_func() below to
// work, so the assembly can be inspected. Look at these symbols:
//
//   test_func(UnitSystem<double>)   <- the realistic case
//   test_func_local()               <- units named in the function body
//   test_func_const()               <- same, but si is constexpr
//   test_func_runtime(...)          <- value-argument overload, for contrast
//   probe_hertz(...)                <- single-dimension unit, simplest case
//
// WHAT TO LOOK FOR in test_func
//   The ctor computes 1/x for all seven base units, but au_sq has dim
//   {metre: 2}, so only `m` is ever read. Six of those seven divisions are
//   dead, as are the two get<> calls feeding unit_time and unit_mass. What
//   should survive is roughly:
//       load si.m ; mul by au ; divide 1.0/that ; square ; mul by au*au
//   No reassociation is legal without -ffast-math, so one division stays.
//   If you see seven divsd, the elimination is not happening and that is
//   worth knowing before committing to this design.
//
// STILL TO ADD for a complete answer: the current addget implementation
// side by side, so the NTTP form can be checked byte-identical for the
// units actually used in ComputeEos.cpp and shamphys/*.
// =====================================================================

#include <iostream>

namespace shamunits {

    // -----------------------------------------------------------------
    // Unit.hpp
    // -----------------------------------------------------------------

    /// Exponents of the seven SI base dimensions
    struct Dimension {
        int second{}, metre{}, kilogram{}, ampere{}, kelvin{}, mole{}, candela{};
    };

    constexpr Dimension operator*(Dimension a, Dimension b) {
        return {
            a.second + b.second,
            a.metre + b.metre,
            a.kilogram + b.kilogram,
            a.ampere + b.ampere,
            a.kelvin + b.kelvin,
            a.mole + b.mole,
            a.candela + b.candela};
    }
    constexpr Dimension operator/(Dimension a, Dimension b) {
        return {
            a.second - b.second,
            a.metre - b.metre,
            a.kilogram - b.kilogram,
            a.ampere - b.ampere,
            a.kelvin - b.kelvin,
            a.mole - b.mole,
            a.candela - b.candela};
    }
    constexpr Dimension dim_pow(Dimension d, int n) {
        return {
            d.second * n,
            d.metre * n,
            d.kilogram * n,
            d.ampere * n,
            d.kelvin * n,
            d.mole * n,
            d.candela * n};
    }
    constexpr bool operator==(Dimension a, Dimension b) {
        return a.second == b.second && a.metre == b.metre && a.kilogram == b.kilogram
               && a.ampere == b.ampere && a.kelvin == b.kelvin && a.mole == b.mole
               && a.candela == b.candela;
    }

    /// A unit: its dimension, and its magnitude expressed in SI base units
    struct Unit {
        Dimension dim{};
        double si_factor = 1;
    };

    constexpr Unit operator*(Unit a, Unit b) { return {a.dim * b.dim, a.si_factor * b.si_factor}; }
    constexpr Unit operator/(Unit a, Unit b) { return {a.dim / b.dim, a.si_factor / b.si_factor}; }
    constexpr Unit operator*(double k, Unit u) { return {u.dim, k * u.si_factor}; }
    constexpr Unit operator*(Unit u, double k) { return {u.dim, u.si_factor * k}; }
    constexpr Unit operator/(Unit u, double k) { return {u.dim, u.si_factor / k}; }

    /// u^n -- the only place in the library that can divide, and it folds at
    /// compile time whenever the argument is a constant expression.
    constexpr Unit upow(Unit u, int n) {
        double f = 1.0, b = u.si_factor;
        int k = (n < 0) ? -n : n;
        while (k) {
            if (k & 1)
                f *= b;
            b *= b;
            k >>= 1;
        }
        return {dim_pow(u.dim, n), (n < 0) ? 1.0 / f : f};
    }

    constexpr Unit base_unit(Dimension d) { return {d, 1.0}; }
    constexpr Unit scale(double k) { return {Dimension{}, k}; }

    // -----------------------------------------------------------------
    // details/utils.hpp -- pow_constexpr_fast_inv is verbatim from the repo
    // -----------------------------------------------------------------
    namespace details {

        template<int power, class T>
        inline constexpr T pow_constexpr_fast_inv(T a, T a_inv) noexcept {
            if constexpr (power < 0) {
                return pow_constexpr_fast_inv<-power>(a_inv, a);
            } else if constexpr (power == 0) {
                return T{1};
            } else if constexpr (power % 2 == 0) {
                T tmp = pow_constexpr_fast_inv<power / 2>(a, a_inv);
                return tmp * tmp;
            } else {
                T tmp = pow_constexpr_fast_inv<(power - 1) / 2>(a, a_inv);
                return tmp * tmp * a;
            }
        }

        /// Runtime twin: same shape, caller supplies the reciprocal so a
        /// negative exponent never needs a division.
        template<class T>
        constexpr T ipow(T a, int n, T a_inv) noexcept {
            if (n == 0)
                return T{1};
            if (n < 0) {
                a = a_inv;
                n = -n;
            }
            T res = T{1};
            while (n) {
                if (n & 1)
                    res *= a;
                a *= a;
                n >>= 1;
            }
            return res;
        }

    } // namespace details

    // -----------------------------------------------------------------
    // UnitSystem.hpp
    // -----------------------------------------------------------------
    template<class T>
    struct UnitSystem {
        T s, m, kg, A, K, mol, cd;
        T s_inv, m_inv, kg_inv, A_inv, K_inv, mol_inv, cd_inv;

        constexpr explicit UnitSystem(
            T unit_time        = 1,
            T unit_length      = 1,
            T unit_mass        = 1,
            T unit_current     = 1,
            T unit_temperature = 1,
            T unit_qte         = 1,
            T unit_lumint      = 1)
            : s(1 / unit_time), m(1 / unit_length), kg(1 / unit_mass), A(1 / unit_current),
              K(1 / unit_temperature), mol(1 / unit_qte), cd(1 / unit_lumint), s_inv(unit_time),
              m_inv(unit_length), kg_inv(unit_mass), A_inv(unit_current), K_inv(unit_temperature),
              mol_inv(unit_qte), cd_inv(unit_lumint) {}

        /// Unit known at compile time. Every exponent is a constant expression,
        /// so pow_constexpr_fast_inv<0> expands to T{1} and vanishes.
        template<const Unit &u, class Tret = T>
        constexpr Tret get() const noexcept {
            using namespace details;
            return Tret(u.si_factor) * pow_constexpr_fast_inv<u.dim.second>(Tret(s), Tret(s_inv))
                   * pow_constexpr_fast_inv<u.dim.metre>(Tret(m), Tret(m_inv))
                   * pow_constexpr_fast_inv<u.dim.kilogram>(Tret(kg), Tret(kg_inv))
                   * pow_constexpr_fast_inv<u.dim.ampere>(Tret(A), Tret(A_inv))
                   * pow_constexpr_fast_inv<u.dim.kelvin>(Tret(K), Tret(K_inv))
                   * pow_constexpr_fast_inv<u.dim.mole>(Tret(mol), Tret(mol_inv))
                   * pow_constexpr_fast_inv<u.dim.candela>(Tret(cd), Tret(cd_inv));
        }

        /// Unit as an ordinary argument (string-resolved units).
        template<class Tret = T>
        constexpr Tret get(Unit u) const noexcept {
            using namespace details;
            return Tret(u.si_factor) * ipow(Tret(s), u.dim.second, Tret(s_inv))
                   * ipow(Tret(m), u.dim.metre, Tret(m_inv))
                   * ipow(Tret(kg), u.dim.kilogram, Tret(kg_inv))
                   * ipow(Tret(A), u.dim.ampere, Tret(A_inv))
                   * ipow(Tret(K), u.dim.kelvin, Tret(K_inv))
                   * ipow(Tret(mol), u.dim.mole, Tret(mol_inv))
                   * ipow(Tret(cd), u.dim.candela, Tret(cd_inv));
        }

        /// Convenience / Python path: exactly get(upow(u, power)).
        template<class Tret = T>
        constexpr Tret get(Unit u, int power) const noexcept {
            return get<Tret>(upow(u, power));
        }
    };

    // -----------------------------------------------------------------
    // unit_table.hpp / prefix_table.hpp / constant_table.hpp (excerpt)
    // -----------------------------------------------------------------
    namespace units {
        inline constexpr Unit second   = base_unit(Dimension{.second = 1});
        inline constexpr Unit metre    = base_unit(Dimension{.metre = 1});
        inline constexpr Unit kilogram = base_unit(Dimension{.kilogram = 1});

        inline constexpr Unit hertz  = upow(second, -1);
        inline constexpr Unit newton = kilogram * metre / upow(second, 2);
        inline constexpr Unit joule  = newton * metre;

        inline constexpr Unit year              = 31557600.0 * second;
        inline constexpr Unit astronomical_unit = 149597870700.0 * metre;
    } // namespace units

    namespace prefix {
        inline constexpr Unit mega = scale(1e6);
    } // namespace prefix

    namespace constants {
        inline constexpr Unit sol_mass = 1.98847e30 * units::kilogram;
        inline constexpr Unit G
            = 6.6743015e-11 * units::newton * upow(units::metre, 2) / upow(units::kilogram, 2);
    } // namespace constants

    // dimension algebra self-checks
    static_assert(units::joule.dim == (units::newton * units::metre).dim);
    static_assert(constants::G.dim == (Dimension{.second = -2, .metre = 3, .kilogram = -1}));

} // namespace shamunits

// =====================================================================
// The example
// =====================================================================
using namespace shamunits;

// `inline` is not allowed on a block-scope variable, so these live at namespace
// scope (which is what the plan does anyway). If you want them local, `static
// constexpr` works too -- static storage duration is what the reference NTTP
// needs, and since C++17 linkage is not required. See test_func_local().
inline constexpr Unit megayear = prefix::mega * units::year;
inline constexpr Unit au_sq    = upow(units::astronomical_unit, 2);

/// The realistic case: `si` is a by-value runtime parameter, so nothing here is
/// constant-folded from the caller.
double test_func(UnitSystem<double> si) {
    // You cannot declare a variable inside a return statement; construct it
    // first (or return the temporary directly).
    UnitSystem<double> astro_units{
        si.get<megayear>(),                 // unit_time   in s
        si.get<units::astronomical_unit>(), // unit_length in m
        si.get<constants::sol_mass>(),      // unit_mass   in kg
    };
    return astro_units.get<au_sq>();
}

/// Same, with the composed units declared locally.
double test_func_local(UnitSystem<double> si) {
    static constexpr Unit my_year = prefix::mega * units::year;
    static constexpr Unit my_ausq = upow(units::astronomical_unit, 2);

    UnitSystem<double> astro{
        si.get<my_year>(),
        si.get<units::astronomical_unit>(),
        si.get<constants::sol_mass>(),
    };
    return astro.get<my_ausq>();
}

/// Same computation, but the input system is a compile-time constant. This
/// should collapse to a single constant load.
double test_func_const() {
    constexpr UnitSystem<double> si{};
    UnitSystem<double> astro_units{
        si.get<megayear>(),
        si.get<units::astronomical_unit>(),
        si.get<constants::sol_mass>(),
    };
    return astro_units.get<au_sq>();
}

/// Contrast: the value-argument overload, where the seven exponents are runtime
/// values and elimination is up to the optimizer.
double test_func_runtime(UnitSystem<double> si, Unit u) { return si.get(u); }

/// Simplest possible case: one dimension, exponent -1.
double probe_hertz(const UnitSystem<double> &si) { return si.get<units::hertz>(); }

/// Same unit through the value-argument overload, argument known statically.
double probe_hertz_byval(const UnitSystem<double> &si) { return si.get(units::hertz); }

// ---------------------------------------------------------------------
// Compile-time checks. NOT exact equality -- au^2 is 2.238e22, past 2^53, and
// 1/au is inexact, so the round trip lands on 1 to within a few ulps.
// ---------------------------------------------------------------------
constexpr bool close(double a, double b, double rel) {
    double d = a - b;
    if (d < 0)
        d = -d;
    double s = (b < 0) ? -b : b;
    return d <= rel * s;
}

static_assert(UnitSystem<double>{}.get<units::astronomical_unit>() == 149597870700.0);
static_assert(UnitSystem<double>{}.get<megayear>() == 3.15576e13);
// test_func_const() itself is deliberately NOT constexpr, so that it shows up in
// the assembly; repeat the computation in a constexpr lambda to assert on it.
static_assert(close(
    [] {
        constexpr UnitSystem<double> si{};
        UnitSystem<double> astro{
            si.get<megayear>(),
            si.get<units::astronomical_unit>(),
            si.get<constants::sol_mass>(),
        };
        return astro.get<au_sq>();
    }(),
    1.0,
    1e-12));

int main() {
    constexpr UnitSystem<double> si{};
    std::cout << test_func(si) << std::endl;      // 1
    std::cout << test_func_const() << std::endl;  // 1
    std::cout << probe_hertz(si) << std::endl;    // 1
    std::cout << si.get<megayear>() << std::endl; // 3.15576e+13 (old API: 3.15576e+19)
    return 0;
}
