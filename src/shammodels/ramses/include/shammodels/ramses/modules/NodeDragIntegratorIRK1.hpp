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
 * @file NodeDragIntegratorIRK1.hpp
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief First order implicit Runge-Kutta drag integrator
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
    X_RO(shamrock::solvergraph::IFieldRefs<Tscal>, alphas)                                         \
    X_RO(shamrock::solvergraph::IFieldRefs<Tscal>, rho_next)                                       \
    X_RO(shamrock::solvergraph::IFieldRefs<Tvec>, rhov_next)                                       \
    X_RO(shamrock::solvergraph::IFieldRefs<Tscal>, rhoe_next)                                      \
    X_RO(shamrock::solvergraph::IFieldRefs<Tscal>, rho_dust_next)                                  \
    X_RO(shamrock::solvergraph::IFieldRefs<Tvec>, rhov_dust_next)                                  \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldRefs<Tscal>, spans_rho)                                      \
    X_RW(shamrock::solvergraph::IFieldRefs<Tvec>, spans_rhov)                                      \
    X_RW(shamrock::solvergraph::IFieldRefs<Tscal>, spans_rhoe)                                     \
    X_RW(shamrock::solvergraph::IFieldRefs<Tscal>, spans_rho_dust)                                 \
    X_RW(shamrock::solvergraph::IFieldRefs<Tvec>, spans_rhov_dust)

namespace shammodels::basegodunov::modules {

    /**
     * @brief First order implicit Runge-Kutta drag integrator
     *
     * Reads the post flux, pre drag state (\f$ U^* \f$, produced by NodeDragUpdateNoSrc) and
     * the per cell drag rates, and writes the drag updated state into the patch fields.
     */
    template<class Tvec>
    class NodeDragIntegratorIRK1 : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_size;
        u32 ndust;
        bool enable_frictional_heating;

        public:
        NodeDragIntegratorIRK1(u32 block_size, u32 ndust, bool enable_frictional_heating)
            : block_size(block_size), ndust(ndust),
              enable_frictional_heating(enable_frictional_heating) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "DragIntegratorIRK1"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
