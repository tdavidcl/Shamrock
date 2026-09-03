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
 * @file riemann_hllc.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Thomas Guillet (T.A.Guillet@exeter.ac.uk) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief HLLC Riemann solvers for the gas equations
 * From original version by Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 */

#include "shammath/riemann_common.hpp"

namespace shammath {

    /**
     * @brief HLLC solver based on section 10.4 from Toro 3rd Edition , Springer 2009.
     *         The wave speeds estimates are based on Bernd Einfeldt (SIAM, 1988), On Godunov-Type
     *          Methods for Gas Dynamics, using the pressure in the star region estimated through
     *          the primitive variable solver (valid for an adiabatic equation of state).
     *
     *        Generic over any HydroState (see riemann_common.hpp), except for the wave speed
     *        estimate itself: it needs the raw adiabatic index gamma (that is precisely why the
     *        EOS-agnostic hllc_davis_flux_x variant below exists), so gamma stays an explicit
     *        extra parameter here. Requires Tprim to expose .rho/.press/.vel and Tcons to be
     *        constructible from {rho, energy-like scalar, rhovel} (the compressible-Euler
     *        conservative-vector shape that this solver's algebra is intrinsically built on).
     * @tparam HS HydroState instantiation (see riemann_common.hpp)
     * @param state HydroState bundle (cons_to_prim, prim_to_cons, soundspeed, flux_x)
     * @param primL left  primitive state
     * @param primR right primitive state
     * @param gamma adiabatic index
     */
    template<class HS>
    inline constexpr typename HS::Tcons hllc_adiab_toro_flux_x(
        const HS &state,
        const typename HS::Tprim &primL,
        const typename HS::Tprim &primR,
        const typename HS::Tscal gamma) {

        using Tscal = typename HS::Tscal;
        using Tcons = typename HS::Tcons;
        using Tvec  = typename HS::Tprim::Tvec;

        // sound speeds
        const auto csL = state.soundspeed(primL);
        const auto csR = state.soundspeed(primR);

        // conservative states and left/right fluxes
        const Tcons cL = state.prim_to_cons(primL);
        const Tcons cR = state.prim_to_cons(primR);
        const Tcons FL = state.flux_x(cL);
        const Tcons FR = state.flux_x(cR);

        // Left variables
        const auto rhoL   = primL.rho;
        const auto pressL = primL.press;
        const auto velxL  = primL.vel[0];

        // Right variables
        const auto rhoR   = primR.rho;
        const auto pressR = primR.press;
        const auto velxR  = primR.vel[0];

        /////////////////// Pressure based wave speed estimation //////////////
        // First compute the pressure estimation in the star region using the primitive variable
        // solver
        //
        // Toro from section 9.3 or Equation (10.67).
        //
        // TODO: It will be interresting to implement and test various pressure estimate algorithms
        // such as : / Two-Rarefaction Riemann Solver (TRRS), Two-Shock Riemann Solver (TSRS) and
        // Adaptive / Riemann Solvers(AIRS or ANRS)
        ////////////////////////////////////////////////////////////////////////
        Tscal rho_bar = 0.5 * (rhoL + rhoR);
        Tscal cs_bar  = 0.5 * (csL + csR);
        Tscal p_pvrs  = 0.5 * (pressL + pressR) - 0.5 * (velxR - velxL) * rho_bar * cs_bar;
        // Pressure in the star region estimate
        Tscal press_star = sham::max(0., p_pvrs);

        // Once the pressure in the star region is known, we then estimates the wave speeds
        // following https://ui.adsabs.harvard.edu/abs/1994ShWav...4...25T/abstract or Equations
        // (10.59 - 10.60) from Toro
        Tscal qL = 0, qR = 0;
        if (press_star <= pressL) {
            qL = 1.;
        } else {
            qL = sycl::sqrt(
                1. + (0.5 * (1. + gamma) / (Tscal) gamma) * (press_star / (Tscal) pressL - 1.));
        }

        if (press_star <= pressR) {
            qR = 1.;
        } else {
            qR = sycl::sqrt(
                1. + (0.5 * (1. + gamma) / (Tscal) gamma) * (press_star / (Tscal) pressR - 1.));
        }

        // wave speed Toro from Equation (10.59)
        Tscal SL = velxL - csL * qL;
        Tscal SR = velxR + csR * qR;

        // lagrangian sound speed
        const Tscal var_L = rhoL * (SL - velxL);
        const Tscal var_R = rhoR * (SR - velxR);

        // S* speed estimate
        // Equation (10.37) from Toro 3rd Edition , Springer 2009
        const Tscal S_star
            = (primR.press - primL.press + velxL * var_L - velxR * var_R) / (var_L - var_R);

        // New pressure estimate in the star region as average the pressure estimate at right
        // and left of S_star in the star region
        // Equation (10.42) from Toro 3rd Edition , Springer 2009
        const Tscal press_LR
            = 0.5 * (pressL + pressR + var_L * (S_star - velxL) + var_R * (S_star - velxR));
        Tvec D{1, 0, 0};
        Tcons D_star{0, S_star, D};

        // Equation (10.40) from Toro 3rd Edition , Springer 2009
        // Left intermediate conservative state in the star region
        // Tcons cL_star = (SL * cL - FL + press_star * D_star) * (1.0 / (SL - S_star));
        Tcons cL_star = (SL * cL - FL + press_LR * D_star) * (1.0 / (SL - S_star));

        // Equation (10.40) from Toro 3rd Edition , Springer 2009
        // Right intermediate conservative state in the star region
        // Tcons cR_star = (SR * cR - FR + press_star * D_star) * (1.0 / (SR - S_star));
        Tcons cR_star = (SR * cR - FR + press_LR * D_star) * (1.0 / (SR - S_star));

        // intemediate Flux in the star region
        // Equation (10.38) from Toro 3rd Edition , Springer 2009
        Tcons FL_star = FL + SL * (cL_star - cL);
        Tcons FR_star = FR + SR * (cR_star - cR);

        // HLLC flux
        auto hllc_flux = [=]() {
            if (SL >= 0) {
                return FL;
            } else if (S_star >= 0) {
                return FL_star;
            } else if (SR >= 0) {
                return FR_star;
            } else
                return FR;
        };

        return hllc_flux();
    }

    /**
     * @brief Backward-compatible overload: same (cL, cR, gamma) signature as before,
     *        implemented on top of the generic HydroState overload above via the ideal-gas
     *        HydroState built by make_gas_hydro_state.
     */
    template<class Tcons>
    inline constexpr Tcons hllc_adiab_toro_flux_x(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        using Tvec = typename Tcons::Tvec;
        auto state = make_gas_hydro_state<Tvec>(gamma);
        return hllc_adiab_toro_flux_x(
            state, cons_to_prim(cL, gamma), cons_to_prim(cR, gamma), gamma);
    }

    /**
     * @brief HLLC flux in the +y direction (adiabatic p* wave speed estimate)
     */
    template<class Tcons>
    inline constexpr Tcons hllc_adiab_toro_flux_y(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_y(hllc_adiab_toro_flux_x(y_to_x(cL), y_to_x(cR), gamma));
    }

    /**
     * @brief HLLC flux in the +z direction (adiabatic p* wave speed estimate)
     */
    template<class Tcons>
    inline constexpr Tcons hllc_adiab_toro_flux_z(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_z(hllc_adiab_toro_flux_x(z_to_x(cL), z_to_x(cR), gamma));
    }

    /**
     * @brief HLLC flux in the -x direction (adiabatic p* wave speed estimate)
     */
    template<class Tcons>
    inline constexpr Tcons hllc_adiab_toro_flux_mx(
        Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hllc_adiab_toro_flux_x(invert_axis(cL), invert_axis(cR), gamma));
    }

    /**
     * @brief HLLC flux in the -y direction (adiabatic p* wave speed estimate)
     */
    template<class Tcons>
    inline constexpr Tcons hllc_adiab_toro_flux_my(
        Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hllc_adiab_toro_flux_y(invert_axis(cL), invert_axis(cR), gamma));
    }

    /**
     * @brief HLLC flux in the -z direction (adiabatic p* wave speed estimate)
     */
    template<class Tcons>
    inline constexpr Tcons hllc_adiab_toro_flux_mz(
        Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hllc_adiab_toro_flux_z(invert_axis(cL), invert_axis(cR), gamma));
    }

    /**
     * @brief HLLC solver based on section 10.4 from Toro 3rd Edition , Springer 2009, using the
     *        Davis (1988) wave speed estimate instead of the pressure based (p*) estimate, i.e.
     *          SL = min(velxL - csL, velxR - csR)
     *          SR = max(velxL + csL, velxR + csR)
     *        This estimate does not rely on an adiabatic equation of state for the pressure in
     *        the star region and can therefore be used for other equations of state.
     *
     *        Generic over any HydroState (see riemann_common.hpp) whose Tprim exposes
     *        .rho/.press/.vel and whose Tcons is constructible from {rho, energy-like scalar,
     *        rhovel} -- unlike hllc_adiab_toro_flux_x, this variant needs no raw gamma at all,
     *        only what soundspeed(prim) already provides.
     * @tparam HS HydroState instantiation (see riemann_common.hpp)
     * @param state HydroState bundle (cons_to_prim, prim_to_cons, soundspeed, flux_x)
     * @param primL left  primitive state
     * @param primR right primitive state
     */
    template<class HS>
    inline constexpr typename HS::Tcons hllc_davis_flux_x(
        const HS &state, const typename HS::Tprim &primL, const typename HS::Tprim &primR) {

        using Tscal = typename HS::Tscal;
        using Tcons = typename HS::Tcons;
        using Tvec  = typename HS::Tprim::Tvec;

        // sound speeds
        const auto csL = state.soundspeed(primL);
        const auto csR = state.soundspeed(primR);

        // conservative states and left/right fluxes
        const Tcons cL = state.prim_to_cons(primL);
        const Tcons cR = state.prim_to_cons(primR);
        const Tcons FL = state.flux_x(cL);
        const Tcons FR = state.flux_x(cR);

        // Left variables
        const auto rhoL   = primL.rho;
        const auto pressL = primL.press;
        const auto velxL  = primL.vel[0];

        // Right variables
        const auto rhoR   = primR.rho;
        const auto pressR = primR.press;
        const auto velxR  = primR.vel[0];

        // Davis estimate, but we'll see later
        Tscal SL = sham::min(velxL - csL, velxR - csR);
        Tscal SR = sham::max(velxL + csL, velxR + csR);

        // lagrangian sound speed
        const Tscal var_L = rhoL * (SL - velxL);
        const Tscal var_R = rhoR * (SR - velxR);

        // S* speed estimate
        // Equation (10.37) from Toro 3rd Edition , Springer 2009
        const Tscal S_star
            = (primR.press - primL.press + velxL * var_L - velxR * var_R) / (var_L - var_R);

        // New pressure estimate in the star region as average the pressure estimate at right
        // and left of S_star in the star region
        // Equation (10.42) from Toro 3rd Edition , Springer 2009
        const Tscal press_LR
            = 0.5 * (pressL + pressR + var_L * (S_star - velxL) + var_R * (S_star - velxR));
        Tvec D{1, 0, 0};
        Tcons D_star{0, S_star, D};

        // Equation (10.40) from Toro 3rd Edition , Springer 2009
        // Left intermediate conservative state in the star region
        Tcons cL_star = (SL * cL - FL + press_LR * D_star) * (1.0 / (SL - S_star));

        // Equation (10.40) from Toro 3rd Edition , Springer 2009
        // Right intermediate conservative state in the star region
        Tcons cR_star = (SR * cR - FR + press_LR * D_star) * (1.0 / (SR - S_star));

        // intemediate Flux in the star region
        // Equation (10.38) from Toro 3rd Edition , Springer 2009
        Tcons FL_star = FL + SL * (cL_star - cL);
        Tcons FR_star = FR + SR * (cR_star - cR);

        // HLLC flux
        auto hllc_flux = [=]() {
            if (SL >= 0) {
                return FL;
            } else if (S_star >= 0) {
                return FL_star;
            } else if (SR >= 0) {
                return FR_star;
            } else
                return FR;
        };

        return hllc_flux();
    }

    /**
     * @brief Backward-compatible overload: same (cL, cR, gamma) signature as before,
     *        implemented on top of the generic HydroState overload above via the ideal-gas
     *        HydroState built by make_gas_hydro_state.
     */
    template<class Tcons>
    inline constexpr Tcons hllc_davis_flux_x(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        using Tvec = typename Tcons::Tvec;
        auto state = make_gas_hydro_state<Tvec>(gamma);
        return hllc_davis_flux_x(state, cons_to_prim(cL, gamma), cons_to_prim(cR, gamma));
    }

    /**
     * @brief HLLC flux in the +y direction (Davis wave speed estimate)
     */
    template<class Tcons>
    inline constexpr Tcons hllc_davis_flux_y(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_y(hllc_davis_flux_x(y_to_x(cL), y_to_x(cR), gamma));
    }

    /**
     * @brief HLLC flux in the +z direction (Davis wave speed estimate)
     */
    template<class Tcons>
    inline constexpr Tcons hllc_davis_flux_z(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_z(hllc_davis_flux_x(z_to_x(cL), z_to_x(cR), gamma));
    }

    /**
     * @brief HLLC flux in the -x direction (Davis wave speed estimate)
     */
    template<class Tcons>
    inline constexpr Tcons hllc_davis_flux_mx(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hllc_davis_flux_x(invert_axis(cL), invert_axis(cR), gamma));
    }

    /**
     * @brief HLLC flux in the -y direction (Davis wave speed estimate)
     */
    template<class Tcons>
    inline constexpr Tcons hllc_davis_flux_my(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hllc_davis_flux_y(invert_axis(cL), invert_axis(cR), gamma));
    }

    /**
     * @brief HLLC flux in the -z direction (Davis wave speed estimate)
     */
    template<class Tcons>
    inline constexpr Tcons hllc_davis_flux_mz(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hllc_davis_flux_z(invert_axis(cL), invert_axis(cR), gamma));
    }

} // namespace shammath
