// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file nttp_probe.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief NTTP feasibility probe for the shamunits redesign
 *
 * Standalone C++20 program answering which non-type-template-parameter
 * spelling is available for passing a Unit at compile time. Not built by
 * CMake; compile by hand or paste into godbolt. See doc/redesign/README.md.
 */

// =====================================================================
// shamunits redesign -- NTTP feasibility probe (standalone, C++20)
//
//   Compile with:  -std=c++20 -O2
//   Godbolt-ready: only <cstdio>, runs and prints a verdict.
//
// QUESTION
//   The redesign wants a unit as a template argument so every dimension
//   exponent is a constant expression. Two spellings, not interchangeable:
//
//     template<Unit u>         takes a prvalue -- get<upow(metre,2)>() --
//                              but needs floating-point NTTP support,
//                              because Unit holds a double.
//     template<const Unit &u>  needs the unit named first, but a reference
//                              type is structural whatever it refers to.
//
// RESULT (measured)
//   clang < 18 : template<double D> and template<Unit u> REJECTED,
//                template<const Unit &u> compiles.
//   clang >= 18: all three compile.
//   -> ship the reference form; enable by-value when clang < 18 is dropped.
//
// SWITCHES -- each feature check is behind its own #if, so one failure
// cannot mask the answer to a different question. (The first version of
// this file got that wrong: PROBE 1 was unconditional, so -DUSE_REF_NTTP=1
// still died on the floating-point check and the reference form was
// reported as failing when it had never been compiled.)
//
//   -DPROBE_FLOAT_NTTP=0   skip the raw template<double> check
//   -DUSE_REF_NTTP=1       build the reference-NTTP variant
//   -DSHOW_REF_REJECTS_PRVALUE=1  (with USE_REF_NTTP=1) EXPECTED TO FAIL;
//                          demonstrates why the reference form needs a name
// =====================================================================

#include <cstdio>

#ifndef PROBE_FLOAT_NTTP
    #define PROBE_FLOAT_NTTP 1
#endif
#ifndef USE_REF_NTTP
    #define USE_REF_NTTP 0
#endif
#ifndef SHOW_REF_REJECTS_PRVALUE
    #define SHOW_REF_REJECTS_PRVALUE 0
#endif

// ---------------------------------------------------------------------
// Core value types (mirrors the planned Unit.hpp -- no STL, device-safe)
// ---------------------------------------------------------------------

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

/// A unit: its dimension, and its magnitude expressed in SI base units.
/// NOTE the `double` member -- that is the whole point of the probe.
struct Unit {
    Dimension dim{};
    double si_factor = 1;
};

constexpr Unit operator*(Unit a, Unit b) { return {a.dim * b.dim, a.si_factor * b.si_factor}; }
constexpr Unit operator/(Unit a, Unit b) { return {a.dim / b.dim, a.si_factor / b.si_factor}; }
constexpr Unit operator*(double k, Unit u) { return {u.dim, k * u.si_factor}; }
constexpr Unit operator*(Unit u, double k) { return {u.dim, u.si_factor * k}; }
constexpr Unit operator/(Unit u, double k) { return {u.dim, u.si_factor / k}; }

/// u^n. The only place in the library that can divide, and it folds at
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

// ---------------------------------------------------------------------
// PROBE 1 -- raw floating-point NTTP (P1714R1). Fails on clang < 18.
// ---------------------------------------------------------------------
#if PROBE_FLOAT_NTTP
template<double D>
struct ProbeFloatNTTP {
    static constexpr double value = D;
};
static_assert(ProbeFloatNTTP<2.5>::value == 2.5, "float NTTP value wrong");
#endif

// ---------------------------------------------------------------------
// PROBE 2 -- structural class NTTP holding a double. Fails on clang < 18.
// ---------------------------------------------------------------------
#if !USE_REF_NTTP
template<Unit U>
struct ProbeUnitNTTP {
    static constexpr Unit value = U;
};
static_assert(ProbeUnitNTTP<Unit{Dimension{.metre = 1}, 3.0}>::value.si_factor == 3.0);
#endif

// ---------------------------------------------------------------------
// Power helpers (pow_constexpr_fast_inv is copied verbatim from
// src/shamunits/include/shamunits/details/utils.hpp)
// ---------------------------------------------------------------------
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

/// Runtime twin: same shape, caller supplies the reciprocal so a negative
/// exponent never needs a division.
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

// ---------------------------------------------------------------------
// UnitSystem (public state / ctor identical to the current library)
// ---------------------------------------------------------------------
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

    /// Compile-time unit. Every exponent is a constant expression, so
    /// pow_constexpr_fast_inv<0> expands to T{1} and vanishes.
#if USE_REF_NTTP
    template<const Unit &u, class Tret = T>
#else
    template<Unit u, class Tret = T>
#endif
    constexpr Tret get() const noexcept {
        return Tret(u.si_factor) * pow_constexpr_fast_inv<u.dim.second>(Tret(s), Tret(s_inv))
               * pow_constexpr_fast_inv<u.dim.metre>(Tret(m), Tret(m_inv))
               * pow_constexpr_fast_inv<u.dim.kilogram>(Tret(kg), Tret(kg_inv))
               * pow_constexpr_fast_inv<u.dim.ampere>(Tret(A), Tret(A_inv))
               * pow_constexpr_fast_inv<u.dim.kelvin>(Tret(K), Tret(K_inv))
               * pow_constexpr_fast_inv<u.dim.mole>(Tret(mol), Tret(mol_inv))
               * pow_constexpr_fast_inv<u.dim.candela>(Tret(cd), Tret(cd_inv));
    }

    /// Unit as an ordinary argument (string-resolved units). No division: the
    /// *_inv members are stored.
    template<class Tret = T>
    constexpr Tret get(Unit u) const noexcept {
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

// ---------------------------------------------------------------------
// A few table entries, written the way the plan proposes
// ---------------------------------------------------------------------
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
    inline constexpr Unit G
        = 6.6743015e-11 * units::newton * upow(units::metre, 2) / upow(units::kilogram, 2);
    inline constexpr Unit sol_mass = 1.98847e30 * units::kilogram;
} // namespace constants

// dimension algebra self-checks (free, and they catch table typos)
static_assert(units::joule.dim == (units::newton * units::metre).dim);
static_assert(units::hertz.dim == (Dimension{} / Dimension{.second = 1}));
static_assert(constants::G.dim == (Dimension{.second = -2, .metre = 3, .kilogram = -1}));

// ---------------------------------------------------------------------
// PROBE 3 -- the syntax the plan asks for: a prvalue as NTTP
// ---------------------------------------------------------------------
inline constexpr UnitSystem<double> si{};

#if !USE_REF_NTTP
// These are the lines that decide the design.
static_assert(si.get<units::astronomical_unit>() == 149597870700.0);
static_assert(si.get<upow(units::astronomical_unit, 2)>() == 149597870700.0 * 149597870700.0);
static_assert(si.get<prefix::mega * units::year>() == 3.15576e13);
static_assert(si.get<units::metre / upow(units::second, 2)>() == 1.0);
#else
// Reference form: composed units must be named first (they need an lvalue).
inline constexpr Unit au_sq    = upow(units::astronomical_unit, 2);
inline constexpr Unit megayear = prefix::mega * units::year;
inline constexpr Unit accel    = units::metre / upow(units::second, 2);
static_assert(si.get<units::astronomical_unit>() == 149597870700.0);
static_assert(si.get<au_sq>() == 149597870700.0 * 149597870700.0);
static_assert(si.get<megayear>() == 3.15576e13);
static_assert(si.get<accel>() == 1.0);

    #if SHOW_REF_REJECTS_PRVALUE
// EXPECTED TO FAIL -- demonstrates why the reference form needs a name.
static_assert(si.get<upow(units::astronomical_unit, 2)>() > 0.0);
    #endif
#endif

// ---------------------------------------------------------------------
// Astro system: the README's own example, with the prefix bug fixed
// ---------------------------------------------------------------------
inline constexpr UnitSystem<double> astro{
    si.get(prefix::mega * units::year), // 1 Myr  (old code gave 1e6x this)
    si.get(units::astronomical_unit),   // 1 au
    si.get(constants::sol_mass),        // 1 Msol
};

/// Relative comparison -- round trips land on 1 to within a few ulps, never
/// bit-exactly: au^2 is 2.238e22, past 2^53, and 1/au is inexact.
constexpr bool close(double a, double b, double rel) {
    double d = a - b;
    if (d < 0)
        d = -d;
    double s = (b < 0) ? -b : b;
    return d <= rel * s;
}

static_assert(close(astro.get(upow(units::astronomical_unit, 2)), 1.0, 1e-12));
static_assert(close(astro.get(units::astronomical_unit, 2), 1.0, 1e-12));

// f32 return type from an f64 system -- no hidden double arithmetic
static_assert(si.get<units::astronomical_unit, float>() == float(149597870700.0));

// ---------------------------------------------------------------------
// Codegen probes -- look at these in the godbolt asm pane.
//   nttp_*    : should be ONE multiply, zero divisions, zero branches
//   byvalue_* : the honest runtime cost
// ---------------------------------------------------------------------
double codegen_nttp_hertz(const UnitSystem<double> &u) { return u.get<units::hertz>(); }

#if !USE_REF_NTTP
double codegen_nttp_metre_sq(const UnitSystem<double> &u) {
    return u.get<upow(units::metre, 2)>();
}
#else
inline constexpr Unit metre_sq = upow(units::metre, 2);
double codegen_nttp_metre_sq(const UnitSystem<double> &u) { return u.get<metre_sq>(); }
#endif

double codegen_nttp_joule(const UnitSystem<double> &u) { return u.get<units::joule>(); }
double codegen_nttp_G(const UnitSystem<double> &u) { return u.get<constants::G>(); }

double codegen_byvalue_literal(const UnitSystem<double> &u) { return u.get(units::hertz); }
double codegen_byvalue_runtime(const UnitSystem<double> &u, Unit x) { return u.get(x); }
double codegen_byvalue_pow(const UnitSystem<double> &u, Unit x, int p) { return u.get(x, p); }

// ---------------------------------------------------------------------
int main() {
    std::printf("=== shamunits NTTP probe ===\n");
#if USE_REF_NTTP
    std::printf("mode            : REFERENCE NTTP  (template<const Unit&>)\n");
#else
    std::printf("mode            : BY-VALUE NTTP   (template<Unit>)\n");
#endif
    std::printf("float NTTP probe: %s\n", PROBE_FLOAT_NTTP ? "enabled" : "skipped");
    std::printf("If you can read this, everything above compiled.\n\n");

    std::printf(
        "si.get<au>()      = %.6e   (expect 1.495979e+11)\n",
        si.get<units::astronomical_unit>());
    std::printf(
        "si.get(Myr)       = %.6e   (expect 3.155760e+13)\n",
        si.get(prefix::mega * units::year));
    std::printf(
        "astro.get(au^2)   = %.6f   (expect 1.000000)\n",
        astro.get(upow(units::astronomical_unit, 2)));
    std::printf("astro.get(G)      = %.6e   (expect 3.947813e+13)\n", astro.get(constants::G));
    std::printf(
        "astro time unit   = %.6e   (expect 3.155760e+13)\n",
        astro.get(upow(units::second, -1)));

    std::printf("\nNote: the current library prints 3.94781e+25 for G and 3.15576e+19 for\n");
    std::printf("the time unit -- off by 1e12 and 1e6, the nested-prefix bug.\n");
    return 0;
}
