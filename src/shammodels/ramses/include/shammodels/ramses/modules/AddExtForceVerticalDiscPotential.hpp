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
 * @file AddExtForceVerticalDiscPotential.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Add the thin disc vertical acceleration of a central mass at the center of each cell.
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(                                                                                          \
        shamrock::solvergraph::IFieldSpan<Tvec>,                                                   \
        spans_coordinates) /* center coordinates of each cell */                                   \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, constant_G) /* Grav Constant*/                  \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_a_ext)

namespace shammodels::basegodunov::modules {

    /// Add the thin disc vertical acceleration of a central mass seen at the cylindrical radius
    /// R0, a_z = -G M (z - z0) / (R0^2 sqrt(R0^2 + (z - z0)^2)), with z0 the midplane (same
    /// form as the SPH solver). It reduces to -Omega_K^2 (z - z0) for |z - z0| << R0.
    template<class Tvec>
    class AddExtForceVerticalDiscPotential : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;     ///< number of cells per block
        Tscal central_mass; ///< mass of the central object
        Tscal R0;           ///< cylindrical radius of the box
        Tscal z0;           ///< position of the midplane

        public:
        AddExtForceVerticalDiscPotential(u32 block_size, Tscal central_mass, Tscal R0, Tscal z0)
            : block_size(block_size), central_mass(central_mass), R0(R0), z0(z0) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "AddExtForceVerticalDiscPotential";
        }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
