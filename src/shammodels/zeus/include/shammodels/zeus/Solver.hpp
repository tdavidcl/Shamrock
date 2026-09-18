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
 * @file Solver.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shambase/exception.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/zeus/SolverConfig.hpp"
#include "shammodels/zeus/modules/SolverStorage.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::zeus {

    template<class Tvec, class TgridVec>
    class Solver {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using u_morton = u64;

        using Config = SolverConfig<Tvec, TgridVec>;

        using AMRBlock = typename Config::AMRBlock;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        Config solver_config;

        SolverStorage<Tvec, TgridVec, u_morton> storage{};

        inline void init_required_fields() {
            context.pdata_layout_add_field<TgridVec>("cell_min", 1);
            context.pdata_layout_add_field<TgridVec>("cell_max", 1);
            context.pdata_layout_add_field<Tscal>("rho", AMRBlock::block_size);
            context.pdata_layout_add_field<Tscal>("eint", AMRBlock::block_size);
            context.pdata_layout_add_field<Tvec>("vel", AMRBlock::block_size);
        }

        Solver(ShamrockCtx &context) : context(context) {}

        Tscal evolve_once(Tscal t_current, Tscal dt_input);

        /**
         * @brief Evolve the simulation until target_time is reached.
         *
         * t_current and dt_input are updated in place as the simulation progresses, so they can
         * be read back by the caller (e.g. to resume with evolve_until again).
         *
         * @param t_current current simulation time, updated in place
         * @param dt_input timestep to use for the next iteration, updated in place with the CFL
         * timestep computed for the following iteration
         * @param target_time time to reach
         * @param niter_max maximum number of iterations to perform (-1 for no limit)
         * @return true if target_time was reached, false if niter_max was reached first
         */
        inline bool evolve_until(
            Tscal &t_current, Tscal &dt_input, Tscal target_time, i32 niter_max) {
            auto step = [&]() {
                if (t_current > target_time) {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "the target time is lower than the current time");
                }

                if (t_current + dt_input > target_time) {
                    dt_input = target_time - t_current;
                }

                Tscal next_dt = evolve_once(t_current, dt_input);
                t_current += dt_input;
                dt_input = next_dt;
            };

            i32 iter_count = 0;

            while (t_current < target_time) {
                step();
                iter_count++;

                if ((iter_count >= niter_max) && (niter_max != -1)) {
                    logger::info_ln(
                        "amr::Zeus", "stopping evolve until because of niter =", iter_count);
                    return false;
                }
            }

            return true;
        }
    };

} // namespace shammodels::zeus
