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
 * @file TransformShearSource.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Move the shear source blocks to the frame of their receiver
 *
 * The translation is the integer part of the shear shift (plus the periodic tile offsets), so
 * that the blocks stay on the integer grid. The data is left untouched: the velocity jump across
 * the shearing faces is applied after the interpolation (see InterpolateShearGhosts).
 */

#include "shammodels/ramses/modules/ShearingPeriodicUtils.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/solvergraph/DDSharedScalar.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::DDSharedScalar<ShearSourceInfos<TgridVec>>, shear_src_infos)       \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::PatchDataLayerDDShared, ghost_layer)

namespace shammodels::basegodunov::modules {

    template<class TgridVec>
    class TransformShearSource : public shamrock::solvergraph::INode {

        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout;

        public:
        TransformShearSource(
            std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout)
            : ghost_layer_layout(std::move(ghost_layer_layout)) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "TransformShearSource"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
