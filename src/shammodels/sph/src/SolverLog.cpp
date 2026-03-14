// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SolverLog.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shammodels/sph/SolverLog.hpp"
#include <stdexcept>

f64 shammodels::sph::SolverLog::get_last_rate() {

    if (step_logs.size() == 0) {
        throw shambase::make_except_with_loc<std::runtime_error>("");
    }

    u64 last_id = step_logs.size() - 1;

    u64 obj_loc  = step_logs[last_id].rank_count;
    f64 time_loc = step_logs[last_id].elasped_sec;

    u64 obj_total = shamalgs::collective::allreduce_sum(obj_loc);
    f64 max_t     = shamalgs::collective::allreduce_max(time_loc);

    return f64(obj_total) / max_t;
}

u64 shammodels::sph::SolverLog::get_last_obj_count() {

    if (step_logs.size() == 0) {
        throw shambase::make_except_with_loc<std::runtime_error>("");
    }

    u64 last_id = step_logs.size() - 1;

    u64 cnt_loc = step_logs[last_id].rank_count;

    u64 cnt_tot = shamalgs::collective::allreduce_sum(cnt_loc);

    return cnt_tot;
}

shamsys::SystemMetrics shammodels::sph::SolverLog::get_last_system_metrics() {

    if (step_logs.size() == 0) {
        throw shambase::make_except_with_loc<std::runtime_error>("");
    }

    auto &last_log = step_logs.back();

    bool report_power_usage = shamsys::has_reporter();

    auto optional_gather_power = [&](const std::optional<f64> &value) -> std::vector<f64> {
        return (report_power_usage) ? shamalgs::collective::gather(value ? value.value() : 0._f64)
                                    : std::vector<f64>{};
    };

    std::vector<f64> rank_energy_consummed_all_ranks
        = optional_gather_power(last_log.system_metrics.rank_energy_consummed);
    std::vector<f64> gpu_energy_consummed_all_ranks
        = optional_gather_power(last_log.system_metrics.gpu_energy_consummed);
    std::vector<f64> cpu_energy_consummed_all_ranks
        = optional_gather_power(last_log.system_metrics.cpu_energy_consummed);
    std::vector<f64> dram_energy_consummed_all_ranks
        = optional_gather_power(last_log.system_metrics.dram_energy_consummed);

    f64 sum_rank_energy_consummed = std::accumulate(
        rank_energy_consummed_all_ranks.begin(), rank_energy_consummed_all_ranks.end(), 0._f64);
    f64 sum_gpu_energy_consummed = std::accumulate(
        gpu_energy_consummed_all_ranks.begin(), gpu_energy_consummed_all_ranks.end(), 0._f64);
    f64 sum_cpu_energy_consummed = std::accumulate(
        cpu_energy_consummed_all_ranks.begin(), cpu_energy_consummed_all_ranks.end(), 0._f64);
    f64 sum_dram_energy_consummed = std::accumulate(
        dram_energy_consummed_all_ranks.begin(), dram_energy_consummed_all_ranks.end(), 0._f64);

    shamsys::SystemMetrics system_metrics;
    system_metrics.rank_energy_consummed = (shamsys::support_rank_energy_consummed())
                                               ? sum_rank_energy_consummed
                                               : std::optional<f64>{};
    system_metrics.gpu_energy_consummed  = (shamsys::support_gpu_energy_consummed())
                                               ? sum_gpu_energy_consummed
                                               : std::optional<f64>{};
    system_metrics.cpu_energy_consummed  = (shamsys::support_cpu_energy_consummed())
                                               ? sum_cpu_energy_consummed
                                               : std::optional<f64>{};
    system_metrics.dram_energy_consummed = (shamsys::support_dram_energy_consummed())
                                               ? sum_dram_energy_consummed
                                               : std::optional<f64>{};

    return system_metrics;
}
