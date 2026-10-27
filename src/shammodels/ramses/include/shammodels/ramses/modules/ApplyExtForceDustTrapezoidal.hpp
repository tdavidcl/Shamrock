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
 * @file ApplyExtForceDustTrapezoidal.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node applying the external forces accelerations at t^n and t^{n+1} to the dust
 * conservative state (trapezoidal rule)
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, dt_over2)                                       \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho_dust)                                 \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dt_rho_dust)                              \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_a_ext)                                     \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_a_ext_next)                                \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_rhov_dust)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Apply the external forces accelerations to the momentum of every dust species with
     * the trapezoidal rule, half of the time step with a_ext at t^n and the dust density at t^n,
     * half with a_ext_next at t^{n+1} and the dust density at t^{n+1}.
     *
     * It updates the state at t^n in place, before the time integration (which then adds the
     * fluxes, and the drag if enabled).
     */
    template<class Tvec>
    class ApplyExtForceDustTrapezoidal : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;
        u32 ndust;

        public:
        ApplyExtForceDustTrapezoidal(u32 block_size, u32 ndust)
            : block_size(block_size), ndust(ndust) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "ApplyExtForceDustTrapezoidal";
        }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
