// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file GridToMorton.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammath/AABB.hpp"
#include "shammodels/ramses/solvegraph/TreeEdge.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamtree/RadixTree.hpp"

namespace shammodels::basegodunov::modules {

    template<class Umorton, class TgridVec>
    class GridToMortonList : public shamrock::solvergraph::INode {

        u32 reduction_level = 0;

        using RTree = RadixTree<Umorton, TgridVec>;

        public:
        GridToMortonList() {}

        struct Edges {
            const shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>> &block_bounds;
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::FieldRefs<TgridVec> &block_min;
            shamrock::solvergraph::FieldRefs<Umorton> &morton_codes;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>
                block_bounds,
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<TgridVec>> block_min,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Umorton>> morton_codes) {
            __internal_set_ro_edges({block_bounds, sizes, block_min});
            __internal_set_rw_edges({morton_codes});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>(0),
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(1),
                get_ro_edge<shamrock::solvergraph::FieldRefs<TgridVec>>(2),
                get_rw_edge<shamrock::solvergraph::FieldRefs<Umorton>>(0)};
        }

        void _impl_evaluate_internal();

        void _impl_reset_internal() {};

        inline virtual std::string _impl_get_label() { return "GridToMortonList"; };

        virtual std::string _impl_get_tex();
    };

} // namespace shammodels::basegodunov::modules
