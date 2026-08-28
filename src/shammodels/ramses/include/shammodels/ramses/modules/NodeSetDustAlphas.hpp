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
 * @file NodeSetDustAlphas.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Fill the per cell, per dust species drag rates consumed by the drag integrators
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <vector>

namespace shammodels::basegodunov::modules {

    /**
     * @brief Broadcast a user supplied constant drag rate per dust species to every cell
     *
     * \f$ \alpha_{i,j} = \alpha_j \f$ for every cell \f$ i \f$ and species \f$ j \f$.
     */
    template<class Tscal>
    class NodeSetDustAlphasConstant : public shamrock::solvergraph::INode {
        u32 block_size;
        u32 ndust;
        std::vector<Tscal> alphas;

        public:
        NodeSetDustAlphasConstant(u32 block_size, u32 ndust, std::vector<Tscal> alphas)
            : block_size(block_size), ndust(ndust), alphas(std::move(alphas)) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            shamrock::solvergraph::IFieldRefs<Tscal> &alphas_field;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tscal>> alphas_field) {
            __internal_set_ro_edges({sizes});
            __internal_set_rw_edges({alphas_field});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_rw_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(0)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "SetDustAlphasConstant"; };

        virtual std::string _impl_get_tex() const;
    };

    /**
     * @brief Derive the drag rates from the local gas state in the Epstein regime
     *
     * For each cell \f$ i \f$ and dust species \f$ j \f$
     * \f[
     *     \alpha_{i,j} = \frac{1}{t_{s,j}(\rho_{{\rm g},i}, c_{s,i})}
     * \f]
     * with \f$ t_s \f$ given by shamphys::epstein_stopping_time evaluated with the **gas**
     * density, as required by the two fluid drag ODE the integrators solve.
     *
     * The gas state is taken from the post flux, pre drag buffers, so that the drag rate is
     * consistent in time with the other state dependent coefficients of the drag operator.
     */
    template<class Tvec>
    class NodeSetDustAlphasEpstein : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_size;
        u32 ndust;
        Tscal gamma;
        bool supersonic_correction;
        std::vector<Tscal> grains_sizes;
        std::vector<Tscal> grains_densities;

        public:
        NodeSetDustAlphasEpstein(
            u32 block_size,
            u32 ndust,
            Tscal gamma,
            bool supersonic_correction,
            std::vector<Tscal> grains_sizes,
            std::vector<Tscal> grains_densities)
            : block_size(block_size), ndust(ndust), gamma(gamma),
              supersonic_correction(supersonic_correction), grains_sizes(std::move(grains_sizes)),
              grains_densities(std::move(grains_densities)) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IFieldRefs<Tscal> &spans_rho;
            const shamrock::solvergraph::IFieldRefs<Tvec> &spans_rhov;
            const shamrock::solvergraph::IFieldRefs<Tscal> &spans_rhoe;
            const shamrock::solvergraph::IFieldRefs<Tscal> &spans_rho_dust;
            const shamrock::solvergraph::IFieldRefs<Tvec> &spans_rhov_dust;
            shamrock::solvergraph::IFieldRefs<Tscal> &alphas_field;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tscal>> spans_rho,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tvec>> spans_rhov,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tscal>> spans_rhoe,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tscal>> spans_rho_dust,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tvec>> spans_rhov_dust,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tscal>> alphas_field) {
            __internal_set_ro_edges(
                {sizes, spans_rho, spans_rhov, spans_rhoe, spans_rho_dust, spans_rhov_dust});
            __internal_set_rw_edges({alphas_field});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<Tvec>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(3),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(4),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<Tvec>>(5),
                get_rw_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(0)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "SetDustAlphasEpstein"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules
