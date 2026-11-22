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
 * @file ApplyShearingBoxForceGas.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node applying the shearing box forces to the gas conservative state
 */

#include "shambackends/vec.hpp"
#include "shammath/AABB.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, dt)                                             \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>, sim_box)                     \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_coordinates)                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dt_rho)                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dt_rhov)                                   \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_rhov)                                      \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rhoe)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Apply the shearing box forces (tidal, Coriolis and radial pressure-gradient forcing
     * eta) to the gas
     *
     * x' = x - x_c, x_c the center of the simulation box along x. The momentum update is the
     * trapezoidal rule, implicit in the momentum at t^{n+1}, of shearing_box_momentum_update
     * and the
     * total energy gets the kinetic energy change of the kick (shearing_box_energy_update), so
     * that the forces do not change the internal energy.
     *
     * It updates the state at t^n in place, before the time integration (which then adds the
     * fluxes, and the drag if enabled).
     */
    template<class Tvec, class TgridVec>
    class ApplyShearingBoxForceGas : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;

        Tscal grid_coord_to_pos_fact;
        Tscal Omega_0;
        Tscal eta;
        Tscal q;

        public:
        ApplyShearingBoxForceGas(
            u32 block_size, Tscal grid_coord_to_pos_fact, Tscal Omega_0, Tscal eta, Tscal q)
            : block_size(block_size), grid_coord_to_pos_fact(grid_coord_to_pos_fact),
              Omega_0(Omega_0), eta(eta), q(q) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ApplyShearingBoxForceGas"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
