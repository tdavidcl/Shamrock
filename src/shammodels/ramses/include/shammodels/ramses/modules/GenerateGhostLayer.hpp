// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file GenerateGhostLayer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shammath/AABB.hpp"
#include "shammath/paving_function.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamrock/solvergraph/SerialPatchTreeEdge.hpp"

namespace shammodels::basegodunov::modules {

    enum class GhostType { None, Periodic, Reflective };

    struct GhostLayerGenMode {
        GhostType ghost_type_x;
        GhostType ghost_type_y;
        GhostType ghost_type_z;
    };

    template<class Tvec, class TgridVec>
    class GenerateGhostLayer : public shamrock::solvergraph::INode {

        shammath::paving_function_general_3d<TgridVec> get_paving();

        public:
        GenerateGhostLayer(GhostLayerGenMode mode) : mode(mode) {}

        private:
        GhostLayerGenMode mode;

        struct Edges {
            // inputs
            shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>> &sim_box;
            shamrock::solvergraph::PatchDataLayerRefs &patch_data_layers;
            shamrock::solvergraph::SerialPatchTreeRefEdge<TgridVec> &patch_tree;
            shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>> &patch_boxes;
            // outputs
            shamrock::solvergraph::PatchDataLayerDDShared &ghost_layers;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>> sim_box,
            std::shared_ptr<shamrock::solvergraph::PatchDataLayerRefs> patch_data_layers,
            std::shared_ptr<shamrock::solvergraph::SerialPatchTreeRefEdge<TgridVec>> patch_tree,
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>
                patch_boxes,
            std::shared_ptr<shamrock::solvergraph::PatchDataLayerDDShared> ghost_layers) {
            __internal_set_ro_edges({sim_box, patch_data_layers, patch_tree, patch_boxes});
            __internal_set_rw_edges({ghost_layers});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>>(0),
                get_ro_edge<shamrock::solvergraph::PatchDataLayerRefs>(1),
                get_ro_edge<shamrock::solvergraph::SerialPatchTreeRefEdge<TgridVec>>(2),
                get_ro_edge<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>(3),
                get_rw_edge<shamrock::solvergraph::PatchDataLayerDDShared>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "GenerateGhostLayer"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::basegodunov::modules
