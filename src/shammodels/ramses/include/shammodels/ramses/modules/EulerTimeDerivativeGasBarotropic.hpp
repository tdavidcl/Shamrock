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
 * @file EulerTimeDerivativeGasBarotropic.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Per cell Euler time derivatives of the gas primitive state for a barotropic equation
 *        of state
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_vel)                                       \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_press)                                    \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_grad_rho)                                  \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dx_v)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dy_v)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dz_v)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_grad_P)                                    \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dt_rho)                                   \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dt_vel)                                    \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dt_press)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Compute the Euler time derivatives of the gas primitive state per cell using
     *        shammath::FluidStateBarotropic
     *
     * Same as NodeEulerTimeDerivativeGasAdiabatic, except that for \f$ P = P(\rho) \f$ the
     * pressure equation reads \f$ \partial_t P = -(\rho c_s^2(\rho) \nabla \cdot \mathbf{v}
     * + \mathbf{v} \cdot \nabla P) \f$, i.e. \f$ \gamma P \f$ is replaced by
     * \f$ \rho c_s^2(\rho) \f$.
     */
    template<class Tvec>
    class NodeEulerTimeDerivativeGasBarotropic : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;
        Tscal rho_crit;
        Tscal cs0;
        Tscal gamma;

        public:
        NodeEulerTimeDerivativeGasBarotropic(u32 block_size, Tscal rho_crit, Tscal cs0, Tscal gamma)
            : block_size(block_size), rho_crit(rho_crit), cs0(cs0), gamma(gamma) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "EulerTimeDerivativeGasBarotropic";
        };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
