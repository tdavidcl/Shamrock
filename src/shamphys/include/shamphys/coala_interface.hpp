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
 * @file SodTube.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/assert.hpp"
#include <experimental/mdspan>
#include <concepts>

namespace shamphys {

    template<class T, class RhoGetter, class E1, class L1, class A1, class E2, class L2, class A2>
    inline void compute_gij_k0(
        RhoGetter &&rho_dust,
        T rho_eps,
        const std::mdspan<const T, E1, L1, A1> &massgrid,
        std::mdspan<T, E2, L2, A2> &gij) {

        static_assert(E1::rank() == 1, "massgrid must be rank 1");
        static_assert(E2::rank() == 1, "gij must be rank 1");

        SHAM_ASSERT(massgrid.extent(0) == gij.extent(0));

        for (int j = 0; j < massgrid.extent(0); j++) {
            T rho_d = rho_dust(j);
            gij(j)  = (rho_d > rho_eps) ? rho_d / (massgrid[j + 1] - massgrid[j]) : 0;
        }
    }

    template<
        class T,
        class E1,
        class E2,
        class E3,
        class L1,
        class L2,
        class L3,
        class A1,
        class A2,
        class A3,
        class Func>
        requires requires(Func f, int a, int b) {
            { f(a, b) } -> std::same_as<T>;
        }
    inline void compute_flux_coag_k0_kdv(
        int nbins,
        const std::mdspan<T, E1, L1, A1> &gij,
        const std::mdspan<const T, E2, L2, A2> &tensor_tabflux_coag,
        Func &&dv,
        std::mdspan<T, E3, L3, A3> &flux) {
        // --- Compile-time rank checks ---
        static_assert(E1::rank() == 1, "gij must be rank 1");
        static_assert(E2::rank() == 3, "tensor_tabflux_coag must be rank 3");
        static_assert(E3::rank() == 1, "flux must be rank 1");

        // --- Runtime extent checks ---
        SHAM_ASSERT(gij.extent(0) == nbins);
        SHAM_ASSERT(gij.extent(0) == nbins);

        SHAM_ASSERT(tensor_tabflux_coag.extent(1) == nbins);
        SHAM_ASSERT(tensor_tabflux_coag.extent(2) == nbins);

        SHAM_ASSERT(tensor_tabflux_coag.extent(0) == flux.extent(0));

        /*
         * Python version:
         * flux = np.einsum("jlm,lm,l,m->j", tensor_tabflux_coag, dv, gij, gij)
         */

        for (int j = 0; j < nbins; ++j) {
            double sum = 0.0;
            for (int l = 0; l < nbins; ++l) {
                for (int m = 0; m < nbins; ++m) {
                    sum += tensor_tabflux_coag(j, l, m) * dv(l, m) * gij[l] * gij[m];
                }
            }
            flux[j] = sum;
        }
    }

    template<class T, class E1, class L1, class A1, class E2, class L2, class A2>
    void coala_flux_diff(
        const std::mdspan<T, E1, L1, A1> &flux, const std::mdspan<T, E2, L2, A2> &S_coag) {

        // --- Compile-time rank checks ---
        static_assert(E1::rank() == 1, "flux must be rank 1");
        static_assert(E2::rank() == 1, "S_coag must be rank 1");

        // --- Runtime extent checks ---
        SHAM_ASSERT(flux.extent(0) == S_coag.extent(0));

        S_coag(0) = -flux(0);
        for (int j = 1; j < flux.extent(0); ++j) {
            S_coag(j) = flux(j - 1) - flux(j);
        }
    }

} // namespace shamphys
