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
 * @file riemann_rusanov.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Thomas Guillet (T.A.Guillet@exeter.ac.uk) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Rusanov Riemann solver for the gas equations
 * From original version by Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 */

#include "shammath/riemann_common.hpp"

namespace shammath {

    // template<class Tcons>
    // inline constexpr Tcons rusanov_flux_x(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
    //     Tcons flux;

    //     const auto primL = cons_to_prim(cL, gamma);
    //     const auto primR = cons_to_prim(cR, gamma);

    //     const auto csL = sound_speed(primL, gamma);
    //     const auto csR = sound_speed(primR, gamma);

    //     const auto S = sham::max(
    //         sham::max(sham::abs(primL.vel[0] - csL), sham::abs(primR.vel[0] - csR)),
    //         sham::max(sham::abs(primL.vel[0] + csL), sham::abs(primR.vel[0] + csR)));

    //     const auto fL = hydro_flux_x(cL, gamma);
    //     const auto fR = hydro_flux_x(cR, gamma);

    //     return (fL + fR) * 0.5 - (cR - cL) * S;
    // }

    /**
     * @brief Rusanov flux, generic over any HydroState (see riemann_common.hpp): works for gas,
     *        dust, or any future state that supplies cons_to_prim/prim_to_cons/soundspeed/
     *        flux_x, as long as Tcons supports +, -, * Tscal.
     *        Equation (10.55) from Toro 3rd Edition , Springer 2009, wave speed estimated
     *        following Equation (10.56).
     */
    template<class HS>
    inline constexpr typename HS::Tcons rusanov_flux_x(
        const HS &state, const typename HS::Tprim &primL, const typename HS::Tprim &primR) {

        using Tcons = typename HS::Tcons;

        const auto csL = state.soundspeed(primL);
        const auto csR = state.soundspeed(primR);

        // Equation (10.56) from Toro 3rd Edition , Springer 2009
        const auto S = sham::max((sham::abs(primL.vel[0]) + csL), (sham::abs(primR.vel[0]) + csR));

        const Tcons cL = state.prim_to_cons(primL);
        const Tcons cR = state.prim_to_cons(primR);
        const Tcons fL = state.flux_x(primL);
        const Tcons fR = state.flux_x(primR);

        // Equation (10.55) from Toro 3rd Edition , Springer 2009
        return 0.5 * ((fL + fR) - (cR - cL) * S);
    }

    /**
     * @brief Backward-compatible overload: same (cL, cR, gamma) signature as before,
     *        implemented on top of the generic HydroState overload above via the ideal-gas
     *        HydroState built by make_gas_hydro_state.
     */
    template<class Tcons>
    inline constexpr Tcons rusanov_flux_x(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        using Tvec = typename Tcons::Tvec;
        auto state = make_gas_hydro_state<Tvec>(gamma);
        return rusanov_flux_x(state, cons_to_prim(cL, gamma), cons_to_prim(cR, gamma));
    }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_y(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_y(rusanov_flux_x(y_to_x(cL), y_to_x(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_z(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_z(rusanov_flux_x(z_to_x(cL), z_to_x(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_mx(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(rusanov_flux_x(invert_axis(cL), invert_axis(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_my(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(rusanov_flux_y(invert_axis(cL), invert_axis(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_mz(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(rusanov_flux_z(invert_axis(cL), invert_axis(cR), gamma));
    }

} // namespace shammath
