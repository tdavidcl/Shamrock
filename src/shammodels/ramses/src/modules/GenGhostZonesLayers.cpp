// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GenGhostZonesLayers.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/assert.hpp"
#include "shambase/exception.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shammath/AABB.hpp"
#include "shammath/paving_function.hpp"
#include "shammodels/ramses/modules/FindGhostLayerCandidates.hpp"
#include "shammodels/ramses/modules/GenGhostZonesLayers.hpp"
#include "shamsys/NodeInstance.hpp"
#include <stdexcept>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::GenGhostZonesLayers<Tvec, TgridVec>::
    _impl_evaluate_internal() {
    auto edges = get_edges();

    // inputs
    auto &sim_box                 = edges.sim_box.value;
    auto &patch_data_layers       = edges.patch_data_layers;
    auto &ghost_layers_candidates = edges.ghost_layers_candidates;
    auto &patch_boxes             = edges.patch_boxes;

    // outputs
    auto &ghost_zones_layers = edges.ghost_zones_layers;

    shambase::DistributedDataShared<sham::DeviceBuffer<u32>> idx_in_ghost;

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    // map candidates to indexes in ghosts
    ghost_layers_candidates.values.template map<sham::DeviceBuffer<u32>>(
        [&](u64 sender, u64 receiver, const GhostLayerCandidateInfos &infos) -> sham::DeviceBuffer<u32> {
            sham::DeviceBuffer<u32> ret(0, sched);

            return ret;
        });
}

template<class Tvec, class TgridVec>
std::string shammodels::basegodunov::modules::GenGhostZonesLayers<Tvec, TgridVec>::_impl_get_tex() {
    return "TODO";
}

template class shammodels::basegodunov::modules::GenGhostZonesLayers<f64_3, i64_3>;
