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
     * @brief Memory layout of the particle neighbour lists (two stages neighbour cache build).
     */
    namespace neigh_cache_particle_layout {
        /// Lists stored contiguously, requiring a count pass then a fill pass
        struct Compact {
            static constexpr std::string_view variant_type_name = "compact";
        };
        /// Every particle owns `capacity` slots, a single pass counts and stores the neighbours
        /// (falls back to the compact layout if a particle has more neighbours than that)
        struct Slots {
            static constexpr std::string_view variant_type_name = "slots";
            u32 capacity                                        = 96;
        };
        using Variant = std::variant<Compact, Slots>;
    } // namespace neigh_cache_particle_layout

    /**
     * @brief Candidate particle data read by the slotted particle pass of the neighbour cache.
     */
    namespace neigh_cache_candidate_data {
        /// Positions and smoothing lengths read through the tree sort map
        struct Indirect {
            static constexpr std::string_view variant_type_name = "indirect";
        };
        /// Positions and smoothing lengths first copied in tree order, so that the candidates of
        /// a leaf are read contiguously (identical lists)
        struct LeafSortedCopy {
            static constexpr std::string_view variant_type_name = "leaf_sorted_copy";
        };
        using Variant = std::variant<Indirect, LeafSortedCopy>;
    } // namespace neigh_cache_candidate_data

    /**
     * @brief Storage of the interacting candidates in the slotted particle pass of the
     * neighbour cache (with the sorted candidate copy).
     */
    namespace neigh_cache_compaction {
        /// Only the interacting candidates are written (one branch per candidate)
        struct Branch {
            static constexpr std::string_view variant_type_name = "branch";
        };
        /// Every candidate is written at the next free slot, which is only kept for the
        /// interacting ones (branch free, the last slot of a particle is a scratch slot)
        struct BranchFree {
            static constexpr std::string_view variant_type_name = "branch_free";
        };
        using Variant = std::variant<Branch, BranchFree>;
    } // namespace neigh_cache_compaction

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

    /**
     * @brief Evaluation of the kernel sums in the neighbour loops (smoothing length iteration,
     * omega).
     *
     * The blocked evaluation is the default on CPU like devices, the scalar one on GPUs (where
     * the per work-item block arrays would not fit in registers).
     */
    namespace neigh_loop_evaluation {
        /// One neighbour at a time, skipping the ones out of the kernel support
        struct Scalar {
            static constexpr std::string_view variant_type_name = "scalar";
        };
        /// Kernel evaluated by blocks of neighbours with a branch free (vectorizable) loop,
        /// then accumulated in the neighbour order (identical sums)
        struct Blocked {
            static constexpr std::string_view variant_type_name = "blocked";
        };
        using Variant = std::variant<Scalar, Blocked>;
    } // namespace neigh_loop_evaluation

    /**
     * @brief Evaluation of the neighbour loop of the fused diff operators kernel.
     *
     * The blocked evaluation is the default on CPU like devices, the scalar one on GPUs.
     */
    namespace diff_operators_evaluation {
        /// One neighbour at a time, skipping the ones out of the kernel supports
        struct Scalar {
            static constexpr std::string_view variant_type_name = "scalar";
        };
        /// Square roots, inverses and kernel derivatives evaluated by blocks of neighbours with a
        /// branch free (vectorizable) loop, then accumulated in the neighbour order (identical
        /// sums)
        struct Blocked {
            static constexpr std::string_view variant_type_name = "blocked";
        };
        using Variant = std::variant<Scalar, Blocked>;
    } // namespace diff_operators_evaluation

    /**
     * @brief Evaluation of the neighbour loop of the varying alpha force kernel (reciprocals
     * arithmetic only, the divisions arithmetic always uses the scalar loop).
     *
     * The blocked evaluation is the default on CPU like devices, the scalar one on GPUs.
     */
    namespace derivs_evaluation {
        /// One neighbour at a time, skipping the ones out of the kernel supports
        struct Scalar {
            static constexpr std::string_view variant_type_name = "scalar";
        };
        /// Square roots, inverses and kernel derivatives evaluated by blocks of neighbours with a
        /// branch free (vectorizable) loop, the interacting pairs being then accumulated in the
        /// neighbour order (identical sums)
        struct Blocked {
            static constexpr std::string_view variant_type_name = "blocked";
        };
        using Variant = std::variant<Scalar, Blocked>;
    } // namespace derivs_evaluation

    /**
     * @brief Tightening of the neighbour lists once the smoothing lengths are final.
     *
     * The neighbour cache is built with the interaction radius including the smoothing length
     * tolerance of the h iteration, while the following loops (diff operators, forces, CFL)
     * only use the pairs within the kernel support of either particle.
     */
    namespace neigh_cache_tighten {
        /// The lists are kept as built
        struct None {
            static constexpr std::string_view variant_type_name = "none";
        };
        /// The omega loop (blocked evaluation only) compacts the lists in place to the pairs
        /// within the kernel support of either particle, in the same order (identical results)
        struct AfterOmega {
            static constexpr std::string_view variant_type_name = "after_omega";
        };
        using Variant = std::variant<None, AfterOmega>;
    } // namespace neigh_cache_tighten

    /// Currently selected implementation for the diff operators section
    const diff_operators::Variant &get_impl_diff_operators();

    /// Currently selected implementation for the diff operators evaluation section
    const diff_operators_evaluation::Variant &get_impl_diff_operators_evaluation();

    /// Currently selected implementation for the force kernel evaluation section
    const derivs_evaluation::Variant &get_impl_derivs_evaluation();

    /// Currently selected implementation for the neighbour lists tightening section
    const neigh_cache_tighten::Variant &get_impl_neigh_cache_tighten();

    /// Currently selected implementation for the CFL signal velocity section
    const cfl_vsig::Variant &get_impl_cfl_vsig();

    /// Currently selected implementation for the neighbour cache particle passes section
    const neigh_cache_particle_pass::Variant &get_impl_neigh_cache_particle_pass();

    /// Currently selected implementation for the neighbour cache particle layout section
    const neigh_cache_particle_layout::Variant &get_impl_neigh_cache_particle_layout();

    /// Currently selected implementation for the neighbour cache candidate data section
    const neigh_cache_candidate_data::Variant &get_impl_neigh_cache_candidate_data();

    /// Currently selected implementation for the neighbour cache compaction section
    const neigh_cache_compaction::Variant &get_impl_neigh_cache_compaction();

    /// Currently selected implementation for the neighbour cache leaf passes section
    const neigh_cache_leaf_pass::Variant &get_impl_neigh_cache_leaf_pass();

    /// Currently selected implementation for the neighbour loops arithmetic section
    const neigh_loop_arithmetic::Variant &get_impl_neigh_loop_arithmetic();

    /// Currently selected implementation for the neighbour loops evaluation section
    const neigh_loop_evaluation::Variant &get_impl_neigh_loop_evaluation();

    /// Whether the neighbour loops should use the blocked evaluation
    inline bool use_blocked_neigh_evaluation() {
        return std::holds_alternative<neigh_loop_evaluation::Blocked>(
            get_impl_neigh_loop_evaluation());
    }

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

/// json (de)serialization of the capacity of the slotted particle neighbour layout
template<>
struct shamalgs::ImplVariantParams<shammodels::sph::impl::neigh_cache_particle_layout::Slots> {
    using Alt = shammodels::sph::impl::neigh_cache_particle_layout::Slots;
    static nlohmann::json to_json(const Alt &p) { return {{"capacity", p.capacity}}; }
    static Alt from_json(const nlohmann::json &j) {
        Alt p{};
        if (j.contains("capacity")) {
            p.capacity = j.at("capacity").get<u32>();
        }
        return p;
    }
};
