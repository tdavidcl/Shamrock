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
 * @file AddPenalizationSDF.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node adding a shape to the signed distance field of a penalization zone
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<TgridVec>, spans_block_min)                             \
    X_RO(shamrock::solvergraph::IFieldSpan<TgridVec>, spans_block_max)                             \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_sdf)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Add a shape to the signed distance field of a penalization zone (union), evaluated
     * at the cell centers
     *
     * \f[
     *   \phi_i \leftarrow \min(\phi_i, \phi_{\rm shape}(\mathbf{x}_i))
     * \f]
     *
     * @tparam Shape a shape of PenalizationConfig, providing `Tscal sdf(Tvec pos) const`
     */
    template<class Tvec, class TgridVec, class Shape>
    class AddPenalizationSDF : public shamrock::solvergraph::INode {
        using Tscal    = shambase::VecComponent<Tvec>;
        using AMRBlock = typename SolverConfig<Tvec, TgridVec>::AMRBlock;

        Shape shape;
        Tscal grid_coord_to_pos_fact;

        public:
        AddPenalizationSDF(Shape shape, Tscal grid_coord_to_pos_fact)
            : shape(shape), grid_coord_to_pos_fact(grid_coord_to_pos_fact) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        virtual std::string _impl_get_label() const;

        virtual std::string _impl_get_tex() const;
    };

    /// Add an infinite cylinder to the signed distance field of a penalization zone
    template<class Tvec, class TgridVec>
    using AddPenalizationSDFCylinder
        = AddPenalizationSDF<Tvec, TgridVec, typename PenalizationConfig<Tvec>::Cylinder>;

    /// Add a sphere to the signed distance field of a penalization zone
    template<class Tvec, class TgridVec>
    using AddPenalizationSDFSphere
        = AddPenalizationSDF<Tvec, TgridVec, typename PenalizationConfig<Tvec>::Sphere>;

    /// Add an axis aligned box to the signed distance field of a penalization zone
    template<class Tvec, class TgridVec>
    using AddPenalizationSDFBox
        = AddPenalizationSDF<Tvec, TgridVec, typename PenalizationConfig<Tvec>::Box>;

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
