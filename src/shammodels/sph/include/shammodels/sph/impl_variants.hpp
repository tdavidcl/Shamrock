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

#include "shambase/aliases_int.hpp"
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

    /**
     * @brief Particle level passes (count & fill) of the two stages neighbour cache build.
     */
    namespace neigh_cache_particle_pass {
        /// Test every particle of every neighbour leaf of the particle's leaf
        struct AllLeafParticles {
            static constexpr std::string_view variant_type_name = "all_leaf_particles";
        };
        /// Skip the neighbour leaves whose AABB is out of reach of the particle (exact, the
        /// resulting neighbour lists are identical)
        struct PruneLeaves {
            static constexpr std::string_view variant_type_name = "prune_leaves";
        };
        using Variant = std::variant<AllLeafParticles, PruneLeaves>;
    } // namespace neigh_cache_particle_pass

    /**
     * @brief Leaf level passes (count & fill) of the two stages neighbour cache build.
     */
    namespace neigh_cache_leaf_pass {
        /// One tree traversal per leaf to count its neighbour leaves, a second one to store them
        struct CountThenFill {
            static constexpr std::string_view variant_type_name = "count_then_fill";
        };
        /// A single tree traversal per leaf, storing the neighbour leaves in a temporary buffer
        /// of `capacity` entries per leaf (leaves with more neighbours are traversed again)
        struct SingleTraversal {
            static constexpr std::string_view variant_type_name = "single_traversal";
            u32 capacity                                        = 128;
        };
        using Variant = std::variant<CountThenFill, SingleTraversal>;
    } // namespace neigh_cache_leaf_pass

    /**
     * @brief Arithmetic of the main neighbour loops (smoothing length iteration, omega, fused
     * diff operators, varying alpha force kernel).
     */
    namespace neigh_loop_arithmetic {
        /// Divisions by the smoothing lengths, densities, ... evaluated for every pair, as in
        /// the reference implementation (bitwise reproducible with the reference)
        struct Divisions {
            static constexpr std::string_view variant_type_name = "divisions";
        };
        /// Divisions replaced by multiplications with inverses computed once per particle or
        /// per pair. Same physics, but the results differ from the reference by rounding
        struct Reciprocals {
            static constexpr std::string_view variant_type_name = "reciprocals";
        };
        using Variant = std::variant<Divisions, Reciprocals>;
    } // namespace neigh_loop_arithmetic

    /// Currently selected implementation for the diff operators section
    const diff_operators::Variant &get_impl_diff_operators();

    /// Currently selected implementation for the CFL signal velocity section
    const cfl_vsig::Variant &get_impl_cfl_vsig();

    /// Currently selected implementation for the neighbour cache particle passes section
    const neigh_cache_particle_pass::Variant &get_impl_neigh_cache_particle_pass();

    /// Currently selected implementation for the neighbour cache leaf passes section
    const neigh_cache_leaf_pass::Variant &get_impl_neigh_cache_leaf_pass();

    /// Currently selected implementation for the neighbour loops arithmetic section
    const neigh_loop_arithmetic::Variant &get_impl_neigh_loop_arithmetic();

    /// Whether the neighbour loops should use the reciprocals arithmetic
    inline bool use_reciprocal_arithmetic() {
        return std::holds_alternative<neigh_loop_arithmetic::Reciprocals>(
            get_impl_neigh_loop_arithmetic());
    }

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

/// json (de)serialization of the capacity of the single traversal leaf pass
template<>
struct shamalgs::ImplVariantParams<shammodels::sph::impl::neigh_cache_leaf_pass::SingleTraversal> {
    using Alt = shammodels::sph::impl::neigh_cache_leaf_pass::SingleTraversal;
    static nlohmann::json to_json(const Alt &p) { return {{"capacity", p.capacity}}; }
    static Alt from_json(const nlohmann::json &j) {
        Alt p{};
        if (j.contains("capacity")) {
            p.capacity = j.at("capacity").get<u32>();
        }
        return p;
    }
};
