// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file timestep_report.cpp
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/numeric_limits.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/tabulate.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/common/timestep_report.hpp"
#include <numeric>
#include <string>
#include <variant>
#include <vector>

std::string shammodels::report_perf_timestep(
    f64 rate,
    u64 nobj,
    u64 npatch,
    f64 tcompute,
    f64 mpi_timer,
    f64 alloc_time_device,
    f64 alloc_time_host,
    size_t max_mem_device,
    size_t max_mem_host,
    std::optional<f64> rank_energy_consummed,
    bool report_power_usage) {

    __shamrock_stack_entry();

    std::vector<f64> rate_all_ranks              = shamalgs::collective::gather(rate);
    std::vector<u64> nobj_all_ranks              = shamalgs::collective::gather(nobj);
    std::vector<u64> npatch_all_ranks            = shamalgs::collective::gather(npatch);
    std::vector<f64> tcompute_all_ranks          = shamalgs::collective::gather(tcompute);
    std::vector<f64> mpi_timer_all_ranks         = shamalgs::collective::gather(mpi_timer);
    std::vector<f64> alloc_time_device_all_ranks = shamalgs::collective::gather(alloc_time_device);
    std::vector<f64> alloc_time_host_all_ranks   = shamalgs::collective::gather(alloc_time_host);
    std::vector<size_t> max_mem_device_all_ranks = shamalgs::collective::gather(max_mem_device);
    std::vector<size_t> max_mem_host_all_ranks   = shamalgs::collective::gather(max_mem_host);
    std::vector<f64> rank_energy_consummed_all_ranks
        = (report_power_usage)
              ? shamalgs::collective::gather(
                    (rank_energy_consummed) ? rank_energy_consummed.value() : 0._f64)
              : std::vector<f64>{};

    if (shamcomm::world_rank() != 0) {
        return "";
    }

    // be careful with overflows
    u64 obj_total    = std::accumulate(nobj_all_ranks.begin(), nobj_all_ranks.end(), 0_u64);
    u64 npatch_total = std::accumulate(npatch_all_ranks.begin(), npatch_all_ranks.end(), 0_u64);
    f64 max_t        = *std::max_element(tcompute_all_ranks.begin(), tcompute_all_ranks.end());
    f64 sum_t        = std::accumulate(tcompute_all_ranks.begin(), tcompute_all_ranks.end(), 0.0);
    f64 sum_mpi      = std::accumulate(mpi_timer_all_ranks.begin(), mpi_timer_all_ranks.end(), 0.0);
    f64 sum_alloc_device = std::accumulate(
        alloc_time_device_all_ranks.begin(), alloc_time_device_all_ranks.end(), 0.0);
    f64 sum_alloc_host
        = std::accumulate(alloc_time_host_all_ranks.begin(), alloc_time_host_all_ranks.end(), 0.0);
    size_t sum_mem_device_total
        = std::accumulate(max_mem_device_all_ranks.begin(), max_mem_device_all_ranks.end(), 0_u64);
    size_t sum_mem_host_total
        = std::accumulate(max_mem_host_all_ranks.begin(), max_mem_host_all_ranks.end(), 0_u64);

    std::vector<std::string> rank_power_step_all_ranks;
    std::string sum_power_step;
    if (report_power_usage) {
        for (u32 i = 0; i < shamcomm::world_size(); i++) {
            if (rank_energy_consummed_all_ranks[i] > 0._f64) {
                rank_power_step_all_ranks.push_back(shambase::format(
                    "{:.1f} W", f64(rank_energy_consummed_all_ranks[i]) / max_t));
            } else {
                rank_power_step_all_ranks.push_back("N/A");
            }
        }
        f64 sum_rank_energy_consummed = std::accumulate(
            rank_energy_consummed_all_ranks.begin(), rank_energy_consummed_all_ranks.end(), 0._f64);
        sum_power_step = shambase::format("{:.1e} W", sum_rank_energy_consummed / max_t);
    }

    u32 cols_count = report_power_usage ? 10_u32 : 9_u32;

    using Table = shambase::table;

    Table table(cols_count);

    std::vector<std::string> header = {"rank",
        "rate (N/s)",
        "Nobj",
        "Npatch",
        "tstep",
        "MPI",
        "alloc d% h%",
        "mem (max) d",
        "mem (max) h"};
    if (report_power_usage) {
        header.push_back("power");
    }

    table.add_double_rule();
    table.add_data(header, Table::center);
    table.add_double_rule();
    for (u32 i = 0; i < shamcomm::world_size(); i++) {
        std::vector<std::string> row = {
            shambase::format("{:<4}", i),
            shambase::format("{:.4e}", rate_all_ranks[i]),
            shambase::format("{:}", nobj_all_ranks[i]),
            shambase::format("{:}", npatch_all_ranks[i]),
            shambase::format("{:.3e}", tcompute_all_ranks[i]),
            shambase::format("{:.1f}%", 100 * (mpi_timer_all_ranks[i] / tcompute_all_ranks[i])),
            shambase::format(
                "{:>.1f}% {:<.1f}%",
                100 * (alloc_time_device_all_ranks[i] / tcompute_all_ranks[i]),
                100 * (alloc_time_host_all_ranks[i] / tcompute_all_ranks[i])),
            shambase::format("{}", shambase::readable_sizeof(max_mem_device_all_ranks[i])),
            shambase::format("{}", shambase::readable_sizeof(max_mem_host_all_ranks[i])),
        };
        if (report_power_usage) {
            row.push_back(rank_power_step_all_ranks[i]);
        }
        table.add_data(row, Table::right);
    }
    if (shamcomm::world_size() > 1) {
        std::vector<std::string> ruled = {"", "<sum N/max t>", "<sum>", "<sum>", "<max>", "<avg>",
            "<avg>", "<sum>", "<sum>"};
        if (report_power_usage) {
            ruled.push_back("<sum>");
        }
        table.add_rulled_data(ruled);
        std::vector<std::string> all_row = {
            "all",
            shambase::format("{:.4e}", f64(obj_total) / max_t),
            shambase::format("{:}", obj_total),
            shambase::format("{:}", npatch_total),
            shambase::format("{:.3e}", max_t),
            shambase::format("{:.1f}%", 100 * (sum_mpi / sum_t)),
            shambase::format(
                "{:>.1f}% {:<.1f}%",
                100 * (sum_alloc_device / sum_t),
                100 * (sum_alloc_host / sum_t)),
            shambase::format("{}", shambase::readable_sizeof(sum_mem_device_total)),
            shambase::format("{}", shambase::readable_sizeof(sum_mem_host_total)),
        };
        if (report_power_usage) {
            all_row.push_back(sum_power_step);
        }
        table.add_data(all_row, Table::right);
    }
    table.add_rule();

    return "Timestep perf report:" + table.render();
}
