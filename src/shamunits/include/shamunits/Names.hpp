// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Names.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "details/utils.hpp"
#include <string_view>
#include <stdexcept>
#include <string>

/// \brief Definition of all units
///
/// This defines units and their short name
#define XMAC_UNITS                                                                                 \
    X1(second, s) /*base units*/                                                                   \
    X1(metre, m)                                                                                   \
    X1(kilogram, kg)                                                                               \
    X1(Ampere, A)                                                                                  \
    X1(Kelvin, K)                                                                                  \
    X1(mole, mol)                                                                                  \
    X1(candela, cd)                                                                                \
    /*derived units*/                                                                              \
    X1(Hertz, Hz)    /* hertz : frequency (s-1) */                                                 \
    X1(Newton, N)    /* (kg.m.s-2)*/                                                               \
    X1(Pascal, Pa)   /* (kg.m-1.s-2) = (N/m2)*/                                                    \
    X1(Joule, J)     /* (kg.m2.s-2) = (N.m = Pa.m3)*/                                              \
    X1(Watt, W)      /* (kg.m2.s-3) = (J/s)*/                                                      \
    X1(Coulomb, C)   /* (s.A)*/                                                                    \
    X1(Volt, V)      /* (kg.m2.s-3.A-1) = (W/A) = (J/C)*/                                          \
    X1(Farad, F)     /* (kg-1.m-2.s4.A2) = (C/V) = (C2/J)*/                                        \
    X1(Ohm, ohm)     /* (kg.m2.s-3.A-2) = (V/A) = (J.s/C2)*/                                       \
    X1(Siemens, S)   /* (kg-1.m-2.s3.A2) = (ohm-1)*/                                               \
    X1(Weber, Wb)    /* (kg.m2.s-2.A-1) = (V.s)*/                                                  \
    X1(Tesla, T)     /* (kg.s-2.A-1) = (Wb/m2)*/                                                   \
    X1(Henry, H)     /* (kg.m2.s-2.A-2) = (Wb/A)*/                                                 \
    X1(lumens, lm)   /* (cd.sr) = (cd.sr)*/                                                        \
    X1(lux, lx)      /* (cd.sr.m-2) = (lm/m2)*/                                                    \
    X1(Bequerel, Bq) /* (s-1)*/                                                                    \
    X1(Gray, Gy)     /* (m2.s-2) = (J/kg)*/                                                        \
    X1(Sievert, Sv)  /* (m2.s-2) = (J/kg)*/                                                        \
    X1(katal, kat)   /* (mol.s-1) */                                                               \
    /*relative units*/                                                                             \
    X1(minutes, mn)                                                                                \
    X1(hours, hr)                                                                                  \
    X1(days, dy)                                                                                   \
    X1(years, yr)                                                                                  \
    X1(astronomical_unit, au)                                                                      \
    X1(light_year, ly)                                                                             \
    X1(parsec, pc)                                                                                 \
    X1(solar_radius, rsol)                                                                         \
    X1(earth_radius, rearth)                                                                       \
    X1(solar_mass, sol_mass)                                                                       \
    X1(electron_volt, eV)                                                                          \
    X1(ergs, erg)                                                                                  \
    X1(british_pint, pint)

/// Definition of all prefixes
#define XMAC_UNIT_PREFIX                                                                           \
    X(tera, T, 12)                                                                                 \
    X(giga, G, 9)                                                                                  \
    X(mega, M, 6)                                                                                  \
    X(kilo, k, 3)                                                                                  \
    X(hecto, hect, 2)                                                                              \
    X(deca, dec, 1)                                                                                \
    X(None, _, 0)                                                                                  \
    /*X(deci  ,deci_, -1)*/                                                                        \
    X(centi, c, -2)                                                                                \
    X(milli, m, -3)                                                                                \
    X(micro, mu, -6)                                                                               \
    X(nano, n, -9)                                                                                 \
    X(pico, p, -12)                                                                                \
    X(femto, f, -15)

namespace shamunits {

    /// Enum of all prefixes
    enum UnitPrefix {
    /// Macro expending to all units prefixes in the enum
    // clang-format off
        #define X(longname, shortname, value) longname = value, shortname = value,
        XMAC_UNIT_PREFIX
        #undef X
        // clang-format on
    };

    /// Get the value of a prefix
    template<class T, UnitPrefix p>
    inline constexpr T get_prefix_val() {
        return details::pow_constexpr_fast_inv<p, T>(10, 1e-1);
    }

    namespace details {
        /// Entry of a table associating a name to an enum value
        template<class T>
        struct NamedValue {
            std::string_view name; ///< the name
            T value;               ///< the associated enum value
        };
    } // namespace details

    // Note : the name <-> value tables below are plain constexpr arrays searched linearly rather
    // than std::unordered_map, they are only used when parsing units (e.g. from python) while
    // static maps in this header had to be instantiated and constructed in every file including it.

    /// Table to convert from a prefix name (long or short) to a prefix enum value
    /// Ideally this should be replaced by cpp reflexion one day
    inline constexpr details::NamedValue<UnitPrefix> unit_prefix_names[] = {
    // clang-format off
        #define X(longname, shortname, value) {#longname, longname}, {#shortname, shortname},
        XMAC_UNIT_PREFIX
        #undef X
        // clang-format on
    };

    /// Table to convert from unit prefix to prefix name in string
    /// Ideally this should be replaced by cpp reflexion one day
    inline constexpr details::NamedValue<UnitPrefix> unit_prefix_short_names[] = {
    // clang-format off
        #define X(longname, shortname, value) {#shortname, shortname},
        XMAC_UNIT_PREFIX
        #undef X
        // clang-format on
    };

    /// Get the prefix name for a UnitPrefix enum value
    inline const std::string get_unit_prefix_name(UnitPrefix p) {

        for (const auto &entry : unit_prefix_short_names) {
            if (entry.value == p) {
                return std::string(entry.name);
            }
        }

        return "[Unknown Unit prefix name]";
    }

    /// Get the UnitPrefix enum value from a prefix name as a string
    inline const UnitPrefix unit_prefix_from_name(std::string p) {

        for (const auto &entry : unit_prefix_names) {
            if (entry.name == p) {
                return entry.value;
            }
        }

        throw std::invalid_argument("this unit prefix name is unknown");
        return None; // to silence a warning
    }

    namespace units {

        // clang-format off
        /// List of all units name
        enum UnitName {
            /// Macro expanding to all unit names
            #define X1(longname, shortname) longname, shortname = longname,
            XMAC_UNITS
            #undef X1
        };

        /// Table to convert from string (long or short name) to unit name
        inline constexpr shamunits::details::NamedValue<UnitName> unit_names[] = {
            /// Macro expanding to the string->UnitName table
            #define X1(longname, shortname) {#longname, longname}, {#shortname, shortname},
            XMAC_UNITS
            #undef X1
        };

        /// Table to convert from unit name to string
        inline constexpr shamunits::details::NamedValue<UnitName> unit_short_names[] = {
            /// Macro expanding to the UnitName->string table
            #define X1(longname, shortname) {#shortname, shortname},
            XMAC_UNITS
            #undef X1
        };
        // clang-format on

        /// Get the unit name for a UnitName enum value
        inline const std::string get_unit_name(UnitName p) {

            for (const auto &entry : unit_short_names) {
                if (entry.value == p) {
                    return std::string(entry.name);
                }
            }

            return "[Unknown Unit name]";
        }

        /// Get the UnitName enum value from a unit name as a string
        inline const UnitName unit_from_name(std::string p) {

            for (const auto &entry : unit_names) {
                if (entry.name == p) {
                    return entry.value;
                }
            }

            throw std::invalid_argument("this unit name is unknown : " + p);
            return s; // to silence a warning
        }

    } // namespace units

} // namespace shamunits
