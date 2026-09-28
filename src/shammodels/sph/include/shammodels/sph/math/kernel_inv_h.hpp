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
 * @file kernel_inv_h.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief SPH kernel evaluation from a precomputed inverse smoothing length.
 *
 * Same quantities as shammath::SPHKernelGen::W_3d, dW_3d and dhW_3d, but taking \f$1/h\f$
 * instead of \f$h\f$, so that the neighbour loops can hoist the divisions out of the loop.
 * The results only differ from the SPHKernelGen ones by floating point rounding.
 */

namespace shamrock::sph {

    /// SPH kernel evaluation from a precomputed inverse smoothing length
    template<class Kernel>
    struct KernelInvH {
        using Tscal = typename Kernel::Tscal;

        /// \f$ W(r,h) = C_{\rm norm} f(r/h) / h^3 \f$
        inline static Tscal W_3d(Tscal r, Tscal hinv) {
            return Kernel::Generator::norm_3d * Kernel::f(r * hinv) * (hinv * hinv * hinv);
        }

        /// \f$ \partial_r W(r,h) = C_{\rm norm} f'(r/h) / h^4 \f$
        inline static Tscal dW_3d(Tscal r, Tscal hinv) {
            Tscal hinv2 = hinv * hinv;
            return Kernel::Generator::norm_3d * Kernel::df(r * hinv) * (hinv2 * hinv2);
        }

        /// \f$ \partial_h W(r,h) = - C_{\rm norm} (3 f(q) + q f'(q)) / h^4 \f$ with \f$ q = r/h \f$
        inline static Tscal dhW_3d(Tscal r, Tscal hinv) {
            Tscal q     = r * hinv;
            Tscal hinv2 = hinv * hinv;
            return -(Kernel::Generator::norm_3d) * (3 * Kernel::f(q) + q * Kernel::df(q))
                   * (hinv2 * hinv2);
        }
    };

} // namespace shamrock::sph
