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
 * @file FindShearSourceGhosts.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Find the blocks to send to the receivers of the shearing-periodic boundary
 *
 * For each receiver patch touching a shearing face, the ghost cells beyond that face are
 * interpolated from the cells of the opposite face shifted along the shear direction. This node
 * lists, for each (sender, receiver) pair and each periodic tile, the sender blocks that cover
 * that shifted region, padded by one block along the shear direction for the slopes.
 */

#include "shammath/AABB.hpp"
#include "shammodels/ramses/modules/ShearingPeriodicUtils.hpp"
#include "shamrock/solvergraph/DDSharedBuffers.hpp"
#include "shamrock/solvergraph/DDSharedScalar.hpp"
#include "shamrock/solvergraph/IPatchDataLayerRefs.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamrock/solvergraph/SerialPatchTreeEdge.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>, sim_box)                     \
    X_RO(shamrock::solvergraph::SerialPatchTreeRefEdge<TgridVec>, patch_tree)                      \
    X_RO(shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>, patch_boxes)                \
    X_RO(shamrock::solvergraph::IPatchDataLayerRefs, patch_data_layers)                            \
    X_RO(shamrock::solvergraph::ScalarEdge<f64>, shear_offset)                                     \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::DDSharedScalar<ShearSourceInfos<TgridVec>>, shear_src_infos)       \
    X_RW(shamrock::solvergraph::DDSharedBuffers<u32>, idx_shear_src)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Find the sender blocks feeding the ghosts of the shearing-periodic faces
     *
     * Inputs: `patch_data_layers` must hold the local blocks only (before the ghost layers are
     * fused into it), block bounds in the fields 0 (min) and 1 (max). `shear_offset` is the shear
     * offset in grid units, wrapped in \f$ [-L/2, L/2] \f$.
     *
     * Outputs, in matching order: `shear_src_infos` (side and integer translation of each
     * interface) and `idx_shear_src` (indexes of the sender blocks to send).
     */
    template<class TgridVec>
    class FindShearSourceGhosts : public shamrock::solvergraph::INode {

        ShearingPeriodicGeom geom;

        public:
        FindShearSourceGhosts(ShearingPeriodicGeom geom) : geom(geom) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "FindShearSourceGhosts"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
