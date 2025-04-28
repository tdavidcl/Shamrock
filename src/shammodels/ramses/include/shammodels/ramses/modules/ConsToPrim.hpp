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
 * @file ConsToPrim.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/Solver.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ComputeField.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class ConsToPrim {

        public:
        using Tscal                      = shambase::VecComponent<Tvec>;
        using Tgridscal                  = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim         = shambase::VectorProperties<Tvec>::dimension;
        static constexpr u32 split_count = shambase::pow_constexpr<dim>(2);

        using Config           = SolverConfig<Tvec, TgridVec>;
        using Storage          = SolverStorage<Tvec, TgridVec, u64>;
        using u_morton         = u64;
        using AMRBlock         = typename Config::AMRBlock;
        using OrientedAMRGraph = OrientedAMRGraph<Tvec, TgridVec>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        ConsToPrim(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void cons_to_prim_gas_spans(
            // inputs
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rho,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_rhov,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rhoe,
            // outputs
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_vel,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_P,
            shambase::DistributedData<u32> &sizes);

        void cons_to_prim_dust_spans(
            // inputs
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rho_dust,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_rhov_dust,
            // outputs
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_vel_dust,
            shambase::DistributedData<u32> &sizes);

        void cons_to_prim_gas();
        void cons_to_prim_dust();

        void reset_gas();
        void reset_dust();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::basegodunov::modules
