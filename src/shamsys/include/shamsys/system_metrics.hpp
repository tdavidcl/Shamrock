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

#include "shambase/aliases_int.hpp"
#include "shambase/popen.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcomm/local_rank.hpp"
#include <cstdlib>
#include <optional>

std::optional<std::string> SHAMSYS_SYSTEM_METRICS_REPORTER_NAME = shamcmdopt::getenv_str_register(
    "SHAMSYS_SYSTEM_METRICS_REPORTER_NAME", "The name of the system metrics reporter to use");

namespace shamsys {

    class ISystemMetricReporter {
        public:
        virtual ~ISystemMetricReporter() = default;

        virtual std::optional<u64> get_rank_energy_consummed() = 0;
    };

    std::unique_ptr<ISystemMetricReporter> &current_reporter();

} // namespace shamsys

namespace shamsys {

    class AuroraSystemMetricReporter : public ISystemMetricReporter {
        public:
        std::optional<u64> get_rank_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                std::string output = shambase::popen_fetch_output("geopmread BOARD_ENERGY board 0");
                return std::stoull(output.c_str());
            }
            return std::nullopt;
        }
    };

    class NoopSystemMetricReporter : public ISystemMetricReporter {
        public:
        std::optional<u64> get_rank_energy_consummed() override { return std::nullopt; }
    };

    inline std::unique_ptr<ISystemMetricReporter> make_reporter(std::string_view reporter_name) {
        if (reporter_name == "aurora") {
            return std::make_unique<AuroraSystemMetricReporter>();
        }
        return std::make_unique<NoopSystemMetricReporter>();
    }

    inline std::unique_ptr<ISystemMetricReporter> make_reporter() {
        if (SHAMSYS_SYSTEM_METRICS_REPORTER_NAME) {
            return make_reporter(*SHAMSYS_SYSTEM_METRICS_REPORTER_NAME);
        }
        return std::make_unique<NoopSystemMetricReporter>();
    }

    inline std::unique_ptr<ISystemMetricReporter> &current_reporter() {
        static std::unique_ptr<ISystemMetricReporter> reporter = nullptr;
        if (!reporter) {
            reporter = make_reporter();
        }
        return reporter;
    }
} // namespace shamsys
