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
 * @file riemann_hll.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Thomas Guillet (T.A.Guillet@exeter.ac.uk) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief HLL Riemann solver for the gas equations
 * From original version by Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 */

#include "shammath/riemann_common.hpp"

namespace shammath {

    /**
     * @brief HLL flux, generic over any HydroState (see riemann_common.hpp): works for gas,
     *        dust, or any future state that supplies cons_to_prim/prim_to_cons/soundspeed/
     *        flux_x, as long as Tcons supports +, -, * Tscal.
     *        Equation (10.26) from Toro 3rd Edition, Springer 2009, wave speeds estimated
     *        following the Toro form, Equation (10.48).
     */
    template<class HS>
    inline constexpr typename HS::Tcons hll_flux_x(
        const HS &state, const typename HS::Tprim &primL, const typename HS::Tprim &primR) {

        using Tcons = typename HS::Tcons;

        const auto csL = state.soundspeed(primL);
        const auto csR = state.soundspeed(primR);

        // Toro form Equation (10.48)
        const auto S_L = sham::min(primL.vel[0] - csL, primR.vel[0] - csR);
        const auto S_R = sham::max(primL.vel[0] + csL, primR.vel[0] + csR);

        const Tcons consL = state.prim_to_cons(primL);
        const Tcons consR = state.prim_to_cons(primR);
        const Tcons fluxL = state.flux_x(primL);
        const Tcons fluxR = state.flux_x(primR);

        // Equation (10.26) from Toro 3rd Edition , Springer 2009
        if (S_L >= 0) {
            return fluxL;
        } else if (S_R <= 0) {
            return fluxR;
        } else {
            const auto S_norm = 1.0 / (S_R - S_L);
            return (fluxL * S_R - fluxR * S_L + (consR - consL) * S_R * S_L) * S_norm;
        }
    }

    /**
     * @brief Backward-compatible overload: same (consL, consR, gamma) signature as before,
     *        implemented on top of the generic HydroState overload above via the ideal-gas
     *        HydroState built by make_gas_hydro_state.
     */
    template<class Tcons>
    inline constexpr Tcons hll_flux_x(
        const Tcons consL, const Tcons consR, const typename Tcons::Tscal gamma) {
        using Tvec = typename Tcons::Tvec;
        auto state = make_gas_hydro_state<Tvec>(gamma);
        return hll_flux_x(state, cons_to_prim(consL, gamma), cons_to_prim(consR, gamma));
    }

    template<class Tcons>
    inline constexpr Tcons hll_flux_y(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_y(hll_flux_x(y_to_x(cL), y_to_x(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons hll_flux_z(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_z(hll_flux_x(z_to_x(cL), z_to_x(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons hll_flux_mx(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hll_flux_x(invert_axis(cL), invert_axis(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons hll_flux_my(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hll_flux_y(invert_axis(cL), invert_axis(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons hll_flux_mz(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hll_flux_z(invert_axis(cL), invert_axis(cR), gamma));
    }

} // namespace shammath
