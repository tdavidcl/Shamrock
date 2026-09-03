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
 * @file riemann_common.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Thomas Guillet (T.A.Guillet@exeter.ac.uk) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Gas and dust conservative/primitive states and axis-transform helpers
 *        shared by every gas and dust Riemann solver
 * From original version by Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 */

#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
namespace shammath {

    template<class Tvec_>
    struct ConsState {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal rho{}, rhoe{};
        Tvec rhovel{};

        const ConsState &operator+=(const ConsState &);
        const ConsState &operator-=(const ConsState &);
        const ConsState &operator*=(const Tscal);
    };

    template<class Tvec_>
    struct PrimState {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal rho{}, press{};
        Tvec vel{};
    };

    template<class Tvec>
    const ConsState<Tvec> &ConsState<Tvec>::operator+=(const ConsState<Tvec> &cst) {
        rho += cst.rho;
        rhoe += cst.rhoe;
        rhovel += cst.rhovel;
        return *this;
    }

    template<class Tvec>
    const ConsState<Tvec> operator+(const ConsState<Tvec> &lhs, const ConsState<Tvec> &rhs) {
        return ConsState<Tvec>(lhs) += rhs;
    }

    template<class Tvec>
    const ConsState<Tvec> &ConsState<Tvec>::operator-=(const ConsState<Tvec> &cst) {
        rho -= cst.rho;
        rhoe -= cst.rhoe;
        rhovel -= cst.rhovel;
        return *this;
    }

    template<class Tvec>
    const ConsState<Tvec> operator-(const ConsState<Tvec> &lhs, const ConsState<Tvec> &rhs) {
        return ConsState<Tvec>(lhs) -= rhs;
    }

    template<class Tvec>
    const ConsState<Tvec> &ConsState<Tvec>::operator*=(
        const typename ConsState<Tvec>::Tscal factor) {
        rho *= factor;
        rhoe *= factor;
        rhovel *= factor;
        return *this;
    }

    template<class Tvec>
    const ConsState<Tvec> operator*(
        const typename ConsState<Tvec>::Tscal factor, const ConsState<Tvec> &rhs) {
        return ConsState<Tvec>(rhs) *= factor;
    }

    template<class Tvec>
    const ConsState<Tvec> operator*(
        const ConsState<Tvec> &lhs, const typename ConsState<Tvec>::Tscal factor) {
        return ConsState<Tvec>(lhs) *= factor;
    }

    template<class Tvec_>
    struct Fluxes {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        std::array<ConsState<Tvec>, 3> F;
    };

    template<class Tvec>
    inline constexpr shambase::VecComponent<Tvec> rhoekin(
        shambase::VecComponent<Tvec> rho, Tvec v) {
        using Tscal    = shambase::VecComponent<Tvec>;
        const Tscal v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        return 0.5 * rho * v2;
    }

    template<class Tvec>
    inline constexpr ConsState<Tvec> prim_to_cons(
        const PrimState<Tvec> prim, typename PrimState<Tvec>::Tscal gamma) {
        ConsState<Tvec> cons;

        cons.rho = prim.rho;

        const auto rhoeint = prim.press / (gamma - 1.0);
        cons.rhoe          = rhoeint + rhoekin(prim.rho, prim.vel);

        cons.rhovel[0] = prim.rho * prim.vel[0];
        cons.rhovel[1] = prim.rho * prim.vel[1];
        cons.rhovel[2] = prim.rho * prim.vel[2];

        return cons;
    }

    template<class Tvec>
    inline constexpr PrimState<Tvec> cons_to_prim(
        const ConsState<Tvec> cons, typename ConsState<Tvec>::Tscal gamma) {
        PrimState<Tvec> prim;

        prim.rho = cons.rho;

        prim.vel[0] = cons.rhovel[0] / cons.rho;
        prim.vel[1] = cons.rhovel[1] / cons.rho;
        prim.vel[2] = cons.rhovel[2] / cons.rho;

        const auto rhoeint = cons.rhoe - rhoekin(prim.rho, prim.vel);
        prim.press         = (gamma - 1.0) * rhoeint;

        return prim;
    }

    template<class Tvec>
    inline constexpr ConsState<Tvec> hydro_flux_x(
        const ConsState<Tvec> cons, typename ConsState<Tvec>::Tscal gamma) {
        ConsState<Tvec> flux;

        const PrimState<Tvec> prim = cons_to_prim(cons, gamma);

        flux.rho = cons.rhovel[0];

        flux.rhoe = (cons.rhoe + prim.press) * prim.vel[0];

        flux.rhovel[0] = cons.rho * prim.vel[0] * prim.vel[0] + prim.press;
        flux.rhovel[1] = cons.rho * prim.vel[0] * prim.vel[1];
        flux.rhovel[2] = cons.rho * prim.vel[0] * prim.vel[2];

        return flux;
    }

    /**
     * @brief Same physical flux as hydro_flux_x(cons, gamma) above, computed directly from a
     *        primitive state instead of a conservative one. Saves the cons_to_prim round trip
     *        that hydro_flux_x(cons, gamma) otherwise has to redo internally whenever the
     *        caller already holds the primitive state (which every Riemann solver below does).
     */
    template<class Tvec>
    inline constexpr ConsState<Tvec> hydro_flux_x(
        const PrimState<Tvec> prim, typename PrimState<Tvec>::Tscal gamma) {
        ConsState<Tvec> flux;

        const auto rhoeint = prim.press / (gamma - 1.0);
        const auto rhoe    = rhoeint + rhoekin(prim.rho, prim.vel);

        flux.rho = prim.rho * prim.vel[0];

        flux.rhoe = (rhoe + prim.press) * prim.vel[0];

        flux.rhovel[0] = prim.rho * prim.vel[0] * prim.vel[0] + prim.press;
        flux.rhovel[1] = prim.rho * prim.vel[0] * prim.vel[1];
        flux.rhovel[2] = prim.rho * prim.vel[0] * prim.vel[2];

        return flux;
    }

    template<class Tvec>
    inline constexpr shambase::VecComponent<Tvec> sound_speed(
        PrimState<Tvec> prim, shambase::VecComponent<Tvec> gamma) {
        return sycl::sqrt(gamma * prim.press / prim.rho);
    }

    /**
     * @brief Generic bundle tying together everything the Riemann solvers below need in order
     *        to operate on a state without hardcoding gas or dust field layouts: the primitive
     *        and conservative types, the conversions between them, the sound speed as a
     *        function of the primitive state, and the physical flux as a function of the
     *        primitive state (already expressed in the local x-aligned frame). flux_x takes a
     *        Tprim (not a Tcons) so the solvers below never have to round-trip a state they
     *        already hold as a primitive through prim_to_cons/cons_to_prim just to get its flux.
     *
     *        A new physics variant (e.g. a tracer field) does not need a new Riemann solver: it
     *        only needs its own Prim/Cons structs -- with the usual +, -, * Tscal operators on
     *        Cons -- plus these four callables, and the same hll_flux_x/hllc_*_flux_x/
     *        rusanov_flux_x templates below apply to it unchanged.
     */
    template<class Prim_, class Cons_, class F_c2p, class F_p2c, class F_cs, class F_flux>
    struct HydroState {
        using Tprim = Prim_;
        using Tcons = Cons_;
        using Tscal = typename Tprim::Tscal;

        F_c2p cons_to_prim; ///< Tcons -> Tprim
        F_p2c prim_to_cons; ///< Tprim -> Tcons
        F_cs soundspeed;    ///< Tprim -> Tscal
        F_flux flux_x;      ///< Tprim -> Tcons, physical flux in the local x direction
    };

    /**
     * @brief Builds a HydroState<Prim, Cons, ...> from its four callables, deducing their
     *        (usually closure) types so callers only ever spell out Prim and Cons, e.g.
     *        `make_hydro_state<MyPrim, MyCons>(my_cons_to_prim, my_prim_to_cons, my_soundspeed,
     *        my_flux_x)`. This is the entry point a new physics variant (gas, dust, a tracer
     *        field, ...) is expected to use.
     */
    template<class Prim, class Cons, class F_c2p, class F_p2c, class F_cs, class F_flux>
    inline constexpr auto make_hydro_state(
        F_c2p cons_to_prim, F_p2c prim_to_cons, F_cs soundspeed, F_flux flux_x) {
        return HydroState<Prim, Cons, F_c2p, F_p2c, F_cs, F_flux>{
            cons_to_prim, prim_to_cons, soundspeed, flux_x};
    }

    /**
     * @brief Builds the HydroState for the ideal-gas Euler equations (the ConsState/PrimState
     *        pair above). Used internally by the backward-compatible (consL, consR, gamma)
     *        overloads of the solvers below so that existing call sites do not need to change.
     */
    template<class Tvec>
    inline constexpr auto make_gas_hydro_state(const shambase::VecComponent<Tvec> gamma) {
        using Prim = PrimState<Tvec>;
        using Cons = ConsState<Tvec>;

        return make_hydro_state<Prim, Cons>(
            [gamma](Cons c) {
                return cons_to_prim(c, gamma);
            },
            [gamma](Prim p) {
                return prim_to_cons(p, gamma);
            },
            [gamma](Prim p) {
                return sound_speed(p, gamma);
            },
            [gamma](Prim p) {
                return hydro_flux_x(p, gamma);
            });
    }

    template<class Tcons>
    inline constexpr Tcons y_to_x(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = c.rhovel[1];
        cprime.rhovel[1] = -c.rhovel[0];
        cprime.rhovel[2] = c.rhovel[2];
        return cprime;
    }

    template<class Tcons>
    inline constexpr Tcons x_to_y(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = -c.rhovel[1];
        cprime.rhovel[1] = c.rhovel[0];
        cprime.rhovel[2] = c.rhovel[2];
        return cprime;
    }

    template<class Tcons>
    inline constexpr Tcons z_to_x(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = c.rhovel[2];
        cprime.rhovel[1] = c.rhovel[1];
        cprime.rhovel[2] = -c.rhovel[0];
        return cprime;
    }

    template<class Tcons>
    inline constexpr Tcons x_to_z(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = -c.rhovel[2];
        cprime.rhovel[1] = c.rhovel[1];
        cprime.rhovel[2] = c.rhovel[0];
        return cprime;
    }

    template<class Tcons>
    inline constexpr Tcons invert_axis(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = -c.rhovel[0];
        cprime.rhovel[1] = -c.rhovel[1];
        cprime.rhovel[2] = -c.rhovel[2];
        return cprime;
    }

    template<class Tvec_>
    struct DustConsState {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal rho{};
        Tvec rhovel{};

        const DustConsState &operator+=(const DustConsState &);
        const DustConsState &operator-=(const DustConsState &);
        const DustConsState &operator*=(const Tscal);
    };

    template<class Tvec_>
    struct DustPrimState {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;
        Tscal rho{};
        Tvec vel{};
    };

    template<class Tvec>
    const DustConsState<Tvec> &DustConsState<Tvec>::operator+=(const DustConsState<Tvec> &d_cst) {
        rho += d_cst.rho;
        rhovel += d_cst.rhovel;
        return *this;
    }

    template<class Tvec>
    const DustConsState<Tvec> operator+(
        const DustConsState<Tvec> &lhs, const DustConsState<Tvec> &rhs) {
        return DustConsState<Tvec>(lhs) += rhs;
    }

    template<class Tvec>
    const DustConsState<Tvec> &DustConsState<Tvec>::operator-=(const DustConsState<Tvec> &d_cst) {
        rho -= d_cst.rho;
        rhovel -= d_cst.rhovel;
        return *this;
    }

    template<class Tvec>
    const DustConsState<Tvec> operator-(
        const DustConsState<Tvec> &lhs, const DustConsState<Tvec> &rhs) {
        return DustConsState<Tvec>(lhs) -= rhs;
    }

    template<class Tvec>
    const DustConsState<Tvec> &DustConsState<Tvec>::operator*=(
        const typename DustConsState<Tvec>::Tscal factor) {
        rho *= factor;
        rhovel *= factor;
        return *this;
    }

    template<class Tvec>
    const DustConsState<Tvec> operator*(
        const DustConsState<Tvec> &lhs, const typename DustConsState<Tvec>::Tscal factor) {
        return DustConsState<Tvec>(lhs) *= factor;
    }

    template<class Tvec>
    const DustConsState<Tvec> operator*(
        const typename DustConsState<Tvec>::Tscal factor, const DustConsState<Tvec> &rhs) {
        return DustConsState<Tvec>(rhs) *= factor;
    }

    template<class Tvec_>
    struct DustFluxes {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;
        std::array<DustConsState<Tvec>, 3> F;
    };

    template<class Tvec>
    inline constexpr DustConsState<Tvec> d_prim_to_cons(const DustPrimState<Tvec> d_prim) {
        DustConsState<Tvec> d_cons;
        d_cons.rho    = d_prim.rho;
        d_cons.rhovel = (d_prim.vel * d_prim.rho);
        return d_cons;
    }

    template<class Tvec>
    inline constexpr DustPrimState<Tvec> d_cons_to_prim(const DustConsState<Tvec> d_cons) {
        DustPrimState<Tvec> d_prim;
        d_prim.rho = d_cons.rho;
        d_prim.vel = (d_cons.rhovel * (1 / d_cons.rho));
        return d_prim;
    }

    template<class Tvec>
    inline constexpr DustConsState<Tvec> d_hydro_flux_x(const DustConsState<Tvec> d_cons) {
        DustConsState<Tvec> d_flux;
        const DustPrimState<Tvec> d_prim = d_cons_to_prim<Tvec>(d_cons);
        const typename DustConsState<Tvec>::Tscal x_vel{d_prim.vel[0]};
        d_flux.rho    = d_cons.rhovel[0];
        d_flux.rhovel = d_prim.vel * (d_cons.rho * x_vel);
        return d_flux;
    }

    template<class Tcons>
    inline constexpr Tcons d_x_to_y(const Tcons c) {
        Tcons d_cst;
        d_cst.rho       = c.rho;
        d_cst.rhovel[0] = -c.rhovel[1];
        d_cst.rhovel[1] = c.rhovel[0];
        d_cst.rhovel[2] = c.rhovel[2];

        return d_cst;
    }

    template<class Tcons>
    inline constexpr Tcons d_y_to_x(const Tcons c) {
        Tcons d_cst;
        d_cst.rho       = c.rho;
        d_cst.rhovel[0] = c.rhovel[1];
        d_cst.rhovel[1] = -c.rhovel[0];
        d_cst.rhovel[2] = c.rhovel[2];
        return d_cst;
    }

    template<class Tcons>
    inline constexpr Tcons d_x_to_z(const Tcons c) {
        Tcons d_cst;
        d_cst.rho       = c.rho;
        d_cst.rhovel[0] = -c.rhovel[2];
        d_cst.rhovel[1] = c.rhovel[1];
        d_cst.rhovel[2] = c.rhovel[0];
        return d_cst;
    }

    template<class Tcons>
    inline constexpr Tcons d_z_to_x(const Tcons c) {
        Tcons d_cst;
        d_cst.rho       = c.rho;
        d_cst.rhovel[0] = c.rhovel[2];
        d_cst.rhovel[1] = c.rhovel[1];
        d_cst.rhovel[2] = -c.rhovel[0];
        return d_cst;
    }

    template<class Tcons>
    inline constexpr Tcons d_invert_axis(const Tcons c) {
        Tcons d_cst;
        d_cst.rho    = c.rho;
        d_cst.rhovel = -(c.rhovel);
        return d_cst;
    }

} // namespace shammath
