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
 * @file NodeDragUpdateNoSrc.hpp
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Advance the conservative state by the fluxes only, leaving the drag source aside
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, dt)                                             \
    X_RO(shamrock::solvergraph::IFieldRefs<Tscal>, spans_rho)                                      \
    X_RO(shamrock::solvergraph::IFieldRefs<Tvec>, spans_rhov)                                      \
    X_RO(shamrock::solvergraph::IFieldRefs<Tscal>, spans_rhoe)                                     \
    X_RO(shamrock::solvergraph::IFieldRefs<Tscal>, spans_rho_dust)                                 \
    X_RO(shamrock::solvergraph::IFieldRefs<Tvec>, spans_rhov_dust)                                 \
    X_RO(shamrock::solvergraph::IFieldRefs<Tscal>, spans_dtrho)                                    \
    X_RO(shamrock::solvergraph::IFieldRefs<Tvec>, spans_dtrhov)                                    \
    X_RO(shamrock::solvergraph::IFieldRefs<Tscal>, spans_dtrhoe)                                   \
    X_RO(shamrock::solvergraph::IFieldRefs<Tscal>, spans_dtrho_dust)                               \
    X_RO(shamrock::solvergraph::IFieldRefs<Tvec>, spans_dtrhov_dust)                               \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldRefs<Tscal>, rho_next)                                       \
    X_RW(shamrock::solvergraph::IFieldRefs<Tvec>, rhov_next)                                       \
    X_RW(shamrock::solvergraph::IFieldRefs<Tscal>, rhoe_next)                                      \
    X_RW(shamrock::solvergraph::IFieldRefs<Tscal>, rho_dust_next)                                  \
    X_RW(shamrock::solvergraph::IFieldRefs<Tvec>, rhov_dust_next)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Compute \f$ U^* = U^n + \Delta t \, L(U^n) \f$ for the gas and the dust
     *
     * This is the flux update of the drag enabled time integrators: the drag source term is
     * applied afterwards by NodeDragIntegratorIRK1 / NodeDragIntegratorEXPO, which read the
     * result of this node without overwriting the \f$ U^n \f$ patch fields.
     */
    template<class Tvec>
    class NodeDragUpdateNoSrc : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_size;
        u32 ndust;

        public:
        NodeDragUpdateNoSrc(u32 block_size, u32 ndust) : block_size(block_size), ndust(ndust) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "DragUpdateNoSrc"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
