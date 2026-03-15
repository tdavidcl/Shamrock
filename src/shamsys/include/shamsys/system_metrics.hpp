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
 * @file system_metrics.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_float.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcomm/wrapper.hpp"
#include <memory>
#include <optional>

inline std::optional<std::string> SHAM_SYSTEM_METRICS_REPORTER = shamcmdopt::getenv_str_register(
    "SHAM_SYSTEM_METRICS_REPORTER", "The name of the system metrics reporter to use");

namespace shamsys {

    class ISystemMetricReporter {
        public:
        virtual ~ISystemMetricReporter() = default;

        virtual std::optional<f64> get_rank_energy_consummed() = 0;
        virtual std::optional<f64> get_gpu_energy_consummed()  = 0;
        virtual std::optional<f64> get_cpu_energy_consummed()  = 0;
        virtual std::optional<f64> get_dram_energy_consummed() = 0;

        virtual bool support_rank_energy_consummed() = 0;
        virtual bool support_gpu_energy_consummed()  = 0;
        virtual bool support_cpu_energy_consummed()  = 0;
        virtual bool support_dram_energy_consummed() = 0;
    };

    std::unique_ptr<ISystemMetricReporter> &current_reporter();

    inline std::optional<f64> get_rank_energy_consummed() {
        return shambase::get_check_ref(current_reporter()).get_rank_energy_consummed();
    }

    inline bool support_rank_energy_consummed() {
        return shambase::get_check_ref(current_reporter()).support_rank_energy_consummed();
    }

    inline std::optional<f64> get_gpu_energy_consummed() {
        return shambase::get_check_ref(current_reporter()).get_gpu_energy_consummed();
    }

    inline bool support_gpu_energy_consummed() {
        return shambase::get_check_ref(current_reporter()).support_gpu_energy_consummed();
    }

    inline std::optional<f64> get_cpu_energy_consummed() {
        return shambase::get_check_ref(current_reporter()).get_cpu_energy_consummed();
    }

    inline bool support_cpu_energy_consummed() {
        return shambase::get_check_ref(current_reporter()).support_cpu_energy_consummed();
    }

    inline std::optional<f64> get_dram_energy_consummed() {
        return shambase::get_check_ref(current_reporter()).get_dram_energy_consummed();
    }

    inline bool support_dram_energy_consummed() {
        return shambase::get_check_ref(current_reporter()).support_dram_energy_consummed();
    }

    struct SystemMetrics {
        f64 wall_time;
        std::optional<f64> rank_energy_consummed;
        std::optional<f64> gpu_energy_consummed;
        std::optional<f64> cpu_energy_consummed;
        std::optional<f64> dram_energy_consummed;
    };

    inline SystemMetrics get_system_metrics(bool barrier = true) {
        if (barrier) {
            shamcomm::mpi::Barrier(MPI_COMM_WORLD);
        }
        auto ret = SystemMetrics{
            shambase::details::get_wtime(),
            get_rank_energy_consummed(),
            get_gpu_energy_consummed(),
            get_cpu_energy_consummed(),
            get_dram_energy_consummed()};
        if (barrier) {
            shamcomm::mpi::Barrier(MPI_COMM_WORLD);
        }
        return ret;
    }

    inline SystemMetrics operator-(const SystemMetrics &lhs, const SystemMetrics &rhs) {
        auto optional_sub = [](const std::optional<f64> &lhs,
                               const std::optional<f64> &rhs) -> std::optional<f64> {
            return (lhs.has_value() && rhs.has_value())
                       ? std::optional<f64>(lhs.value() - rhs.value())
                       : std::nullopt;
        };
        return SystemMetrics{
            lhs.wall_time - rhs.wall_time,
            optional_sub(lhs.rank_energy_consummed, rhs.rank_energy_consummed),
            optional_sub(lhs.gpu_energy_consummed, rhs.gpu_energy_consummed),
            optional_sub(lhs.cpu_energy_consummed, rhs.cpu_energy_consummed),
            optional_sub(lhs.dram_energy_consummed, rhs.dram_energy_consummed)};
    }

    // Returns true if the current reporter is not a NoopSystemMetricReporter (defined below).
    bool has_reporter();

} // namespace shamsys
