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
 * @file NodeComputeDragAlphas.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Compute the dust collision rates (inverse stopping times) of each cell, read by the drag
 * integrators.
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <variant>
#include <vector>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)   /* gas density */                  \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_press) /* gas pressure */                 \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_alphas) /* ndust per cell */

namespace shammodels::basegodunov::modules {

    /**
     * @brief Compute the dust collision rates alpha_j (the inverse of the stopping times) of each
     * cell, ndust values per cell.
     *
     * - Static: the same alpha_j in every cell.
     * - Epstein: alpha_j = 1 / t_j with t_j = rho_grain_j s_j sqrt(pi gamma / 8) / (rho_g c_s)
     *   (shamphys::epstein_stopping_time with the gas density), c_s = sqrt(gamma P / rho_g) being
     *   the adiabatic sound speed of the gas.
     */
    template<class Tvec>
    class NodeComputeDragAlphas : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        public:
        struct Static {
            std::vector<Tscal> alphas;
        };

        struct Epstein {
            Tscal gamma;
            std::vector<Tscal> grain_sizes;
            std::vector<Tscal> grain_densities;
        };

        using Mode = std::variant<Static, Epstein>;

        private:
        u32 block_size;
        u32 ndust;
        Mode mode;

        public:
        NodeComputeDragAlphas(u32 block_size, u32 ndust, Mode mode)
            : block_size(block_size), ndust(ndust), mode(std::move(mode)) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "NodeComputeDragAlphas"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
