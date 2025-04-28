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
 * @file SolverStorage.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/StorageComponent.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/SinkPartStruct.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/TreeTraversalCache.hpp"

namespace shammodels::sph {

    template<class T>
    using Component = shambase::StorageComponent<T>;

    template<class Tvec, class Tmorton>
    class SolverStorage {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using GhostHandle        = BasicSPHGhostHandler<Tvec>;
        using GhostHandleCache   = typename GhostHandle::CacheMap;
        using PreStepMergedField = typename GhostHandle::PreStepMergedField;

        using RTree = RadixTree<Tmorton, Tvec>;

        Component<SerialPatchTree<Tvec>> serial_patch_tree;

        Component<GhostHandle> ghost_handler;

        Component<GhostHandleCache> ghost_patch_cache;

        Component<shambase::DistributedData<PreStepMergedField>> merged_xyzh;

        Component<shambase::DistributedData<RTree>> merged_pos_trees;

        Component<shambase::DistributedData<RadixTreeField<Tscal>>> rtree_rint_field;

        Component<shamrock::tree::ObjectCacheHandler> neighbors_cache;

        Component<shamrock::ComputeField<Tscal>> omega;

        Component<shamrock::patch::PatchDataLayout> ghost_layout;

        Component<shambase::DistributedData<shamrock::MergedPatchData>> merged_patchdata_ghost;

        Component<shamrock::ComputeField<Tscal>> alpha_av_updated;
        Component<shambase::DistributedData<PatchDataField<Tscal>>> alpha_av_ghost;

        Component<shamrock::ComputeField<Tscal>> pressure;
        Component<shamrock::ComputeField<Tscal>> soundspeed;

        Component<shamrock::ComputeField<Tvec>> old_axyz;
        Component<shamrock::ComputeField<Tscal>> old_duint;

        Component<shamrock::ComputeField<Tvec>> old_dB_on_rho;
        Component<shamrock::ComputeField<Tscal>> old_dpsi_on_ch;

        Component<shamrock::ComputeField<Tscal>> old_dtepsilon;
        Component<shamrock::ComputeField<Tvec>> old_dtdeltav;

        Component<std::vector<SinkParticle<Tvec>>> sinks;

        struct Timings {
            f64 interface = 0;
            f64 neighbors = 0;
            f64 io        = 0;

            /// Reset the timings logged in the storage
            void reset() { *this = {}; }
        } timings_details;
    };

} // namespace shammodels::sph
