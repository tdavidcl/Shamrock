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
 * @file Penalization.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Volume penalization source term (immersed obstacles, forcing bands)
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/Solver.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"

namespace shammodels::basegodunov::modules {

    /**
     * @brief Apply the volume penalization zones of PenalizationConfig to the gas state
     *
     * For every zone (in config order) and every cell whose center is inside it, in place in the
     * patch data, the source term \f$ \partial_t q = -(q - q_{\rm target}) / \eta \f$ is
     * integrated with an implicit Euler step:
     * \f[
     *   q \leftarrow \frac{\eta}{\eta + \Delta t} q + \frac{\Delta t}{\eta + \Delta t}
     *   q_{\rm target}
     * \f]
     * applied to the velocity, to the density if the zone has a target density and to the
     * internal energy density \f$ P / (\gamma - 1) \f$ if the zone has a target pressure
     * (otherwise the specific internal energy is kept). The total energy is then rebuilt from the
     * new internal and kinetic energies.
     */
    template<class Tvec, class TgridVec>
    class Penalization {

        public:
        using Tscal = shambase::VecComponent<Tvec>;

        using Config   = SolverConfig<Tvec, TgridVec>;
        using Storage  = SolverStorage<Tvec, TgridVec, u64>;
        using AMRBlock = typename Config::AMRBlock;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        Penalization(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /// Apply the penalization over a timestep dt, does nothing if there is no zone or dt == 0
        void apply(Tscal dt);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::basegodunov::modules
