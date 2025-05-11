// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeBuildTrees.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/time.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/ramses/modules/NodeBuildTrees.hpp"

namespace shammodels::basegodunov::modules {

    template<class Umorton, class TgridVec>
    void NodeBuildTrees<Umorton, TgridVec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        auto &block_min = edges.block_min;
        auto &block_max = edges.block_max;

        const shambase::DistributedData<u32> &indexes_dd = edges.sizes.indexes;

        // TODO move the bounding box computation to another node.

        shambase::DistributedData<shammath::AABB<TgridVec>> bounds = {};

        bounds = indexes_dd.template map<shammath::AABB<TgridVec>>([&](u64 id, auto &merged) {
            TgridVec min_bound = block_min.get_field(id).compute_min();
            TgridVec max_bound = block_max.get_field(id).compute_max();

            logger::raw_ln("AABB", id, min_bound, max_bound);

            return shammath::AABB<TgridVec>{min_bound, max_bound};
        });

        shambase::DistributedData<RTree> trees
            = indexes_dd.template map<RTree>([&](u64 id, auto &merged) {
                  logger::debug_ln("AMR", "compute tree for merged patch", id);

                  auto aabb = bounds.get(id);

                  TgridVec bmin = aabb.lower;
                  TgridVec bmax = aabb.upper;

                  TgridVec diff = bmax - bmin;
                  diff.x()      = shambase::roundup_pow2(diff.x());
                  diff.y()      = shambase::roundup_pow2(diff.y());
                  diff.z()      = shambase::roundup_pow2(diff.z());
                  bmax          = bmin + diff;

                  auto &field_pos = block_min.get_field(id);

                  RTree tree(
                      shamsys::instance::get_compute_scheduler_ptr(),
                      {bmin, bmax},
                      field_pos.get_buf(),
                      field_pos.get_obj_cnt(),
                      reduction_level);

                  return tree;
              });

        trees.for_each([](u64 id, RTree &tree) {
            tree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
            tree.convert_bounding_box(shamsys::instance::get_compute_queue());
        });

        edges.trees.trees = std::move(trees);
    }

    template<class Umorton, class TgridVec>
    std::string NodeBuildTrees<Umorton, TgridVec>::_impl_get_tex() {
        return "TODO";
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeBuildTrees<u64, i64_3>;
