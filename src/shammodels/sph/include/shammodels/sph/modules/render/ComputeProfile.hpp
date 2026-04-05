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
 * @file CartesianRender.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
#include "shammodels/sph/modules/render/RenderFieldGetter.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <functional>
#include <optional>

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class ComputeProfile {

        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        ComputeProfile(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        using field_getter_t = const sham::DeviceBuffer<Tscal> &(
            const shamrock::patch::Patch cur_p, shamrock::patch::PatchDataLayer &pdat);

        inline sham::DeviceBuffer<Tscal> compute_profile_patch(
            const sham::DeviceBuffer<Tscal> &bin_inf,
            const sham::DeviceBuffer<Tscal> &bin_sup,

            std::function<field_getter_t> x_getter,
            std::function<field_getter_t> &y_getter,
            std::optional<std::function<field_getter_t>> x_size_getter,
            bool do_average,
            Tscal min_normlization) {

            sham::DeviceBuffer<Tscal> ret{
                bin_inf.get_size(), shamsys::instance::get_compute_scheduler_ptr()};
            ret.fill(sham::VectorProperties<Tscal>::get_zero());

            scheduler().for_each_patchdata_nonempty(
                [&](const shamrock::patch::Patch cur_p, shamrock::patch::PatchDataLayer &pdat) {
                    sham::DeviceBuffer<Tscal> x_buf
                        = x_getter.runner_function(std::string field_name, lamda_runner lambda)
                });

            shamalgs::collective::reduce_buffer_in_place_sum(ret, MPI_COMM_WORLD);
            return ret;
        }

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };
} // namespace shammodels::sph::modules
