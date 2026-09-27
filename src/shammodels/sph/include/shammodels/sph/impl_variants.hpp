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
 * @file impl_variants.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Runtime selectable implementations of sections of the SPH solver.
 *
 * Performance oriented rewrites of a section of the SPH solver are kept alongside the original
 * implementation, and the one used at runtime is picked through a shamalgs::ImplVariantGlobal
 * selector, exactly like the shamalgs primitives. All the implementations of a given section
 * compute the same physics, they only differ in how the work is scheduled (kernel fusion,
 * pruning, ...).
 *
 * Every selector is registered by name (its "section"), so that they can all be listed and
 * changed at runtime, e.g. from python :
 *
 * @code{.py}
 * shamrock.model_sph.impl.get_impl_sections()
 * shamrock.model_sph.impl.get_default_impl_list("diff_operators")
 * shamrock.model_sph.impl.set_impl("diff_operators", '{"implementation": "separate_kernels"}')
 * @endcode
 */

#include "shamalgs/ImplVariant.hpp"
#include <string_view>
#include <string>
#include <variant>
#include <vector>

namespace shammodels::sph::impl {

    /**
     * @brief Computation of divv, curlv and dtdivv (Cullen & Dehnen 2010 switch).
     *
     * @note The "combined_dtdiv_divcurlv_compute" solver config flag, when set, still forces
     * the fused kernel.
     */
    namespace diff_operators {
        /// One neighbour loop per operator (divv, then curlv, then dtdivv)
        struct SeparateKernels {
            static constexpr std::string_view variant_type_name = "separate_kernels";
        };
        /// A single neighbour loop computing divv, curlv and dtdivv at once
        struct FusedKernel {
            static constexpr std::string_view variant_type_name = "fused_kernel";
        };
        using Variant = std::variant<SeparateKernels, FusedKernel>;
    } // namespace diff_operators

    /**
     * @brief Computation of the signal velocity used by the courant CFL condition.
     */
    namespace cfl_vsig {
        /// Dedicated neighbour loop, after the leapfrog corrector
        struct SeparatePass {
            static constexpr std::string_view variant_type_name = "separate_pass";
        };
        /// Computed within the force kernel neighbour loop (when the force kernel supports it,
        /// otherwise falls back to the dedicated loop)
        struct FusedWithDerivs {
            static constexpr std::string_view variant_type_name = "fused_with_derivs";
        };
        using Variant = std::variant<SeparatePass, FusedWithDerivs>;
    } // namespace cfl_vsig

    /// Currently selected implementation for the diff operators section
    const diff_operators::Variant &get_impl_diff_operators();

    /// Currently selected implementation for the CFL signal velocity section
    const cfl_vsig::Variant &get_impl_cfl_vsig();

    /// List the names of the SPH sections having selectable implementations
    std::vector<std::string> get_impl_sections();

    /// List the available implementations of a section, as config json strings
    std::vector<std::string> get_default_impl_list(const std::string &section);

    /// Get the current implementation of a section, as a config json string (null if unset)
    std::string get_current_impl(const std::string &section);

    /// Select the implementation of a section from a config json string
    void set_impl(const std::string &section, const std::string &impl);

    /// Select the default implementation of every section
    void autoselect_all_impl();

} // namespace shammodels::sph::impl
