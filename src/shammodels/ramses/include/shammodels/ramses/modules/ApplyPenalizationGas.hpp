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
 * @file ApplyPenalizationGas.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node applying the volume penalization of a zone to the gas state
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, dt)                                             \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_sdf)                                      \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_rhov)                                      \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rhoe)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Apply the volume penalization of a zone to the gas state, in place
     *
     * The mask is \f$ \chi_i = 1 \f$ if \f$ \phi_i \leq 0 \f$ (cell center inside the zone) and 0
     * otherwise. In the cells with \f$ \chi_i = 1 \f$, the source term
     * \f$ \partial_t q = -(q - q_{\rm target}) / \eta \f$ is integrated with an implicit Euler step
     * \f[
     *   q \leftarrow \frac{\eta}{\eta + \Delta t} q + \frac{\Delta t}{\eta + \Delta t}
     *   q_{\rm target}
     * \f]
     * applied to the velocity, to the density if the zone has a target density and to the
     * internal energy density \f$ P / (\gamma - 1) \f$ if the zone has a target pressure
     * (otherwise the specific internal energy is kept). The total energy is then rebuilt from the
     * new internal and kinetic energies. Does nothing if \f$ \Delta t = 0 \f$.
     */
    template<class Tvec>
    class ApplyPenalizationGas : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;
        Tscal eta;
        Tvec target_vel;
        bool has_rho;
        Tscal target_rho;
        bool has_eint;
        Tscal target_eint;

        public:
        /**
         * @param target_eint target internal energy density \f$ P_{\rm target} / (\gamma - 1) \f$
         */
        ApplyPenalizationGas(
            u32 block_size,
            Tscal eta,
            Tvec target_vel,
            bool has_rho,
            Tscal target_rho,
            bool has_eint,
            Tscal target_eint)
            : block_size(block_size), eta(eta), target_vel(target_vel), has_rho(has_rho),
              target_rho(target_rho), has_eint(has_eint), target_eint(target_eint) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ApplyPenalizationGas"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
