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
#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shambase/popen.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcomm/local_rank.hpp"
#include <cstdlib>
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
        std::optional<f64> rank_energy_consummed;
        std::optional<f64> gpu_energy_consummed;
        std::optional<f64> cpu_energy_consummed;
        std::optional<f64> dram_energy_consummed;
    };

    inline SystemMetrics get_system_metrics() {
        return SystemMetrics{
            get_rank_energy_consummed(),
            get_gpu_energy_consummed(),
            get_cpu_energy_consummed(),
            get_dram_energy_consummed()};
    }

    inline SystemMetrics operator-(const SystemMetrics &lhs, const SystemMetrics &rhs) {
        auto optional_sub = [](const std::optional<f64> &lhs,
                               const std::optional<f64> &rhs) -> std::optional<f64> {
            return (lhs.has_value() && rhs.has_value())
                       ? std::optional<f64>(lhs.value() - rhs.value())
                       : std::nullopt;
        };
        return SystemMetrics{
            optional_sub(lhs.rank_energy_consummed, rhs.rank_energy_consummed),
            optional_sub(lhs.gpu_energy_consummed, rhs.gpu_energy_consummed),
            optional_sub(lhs.cpu_energy_consummed, rhs.cpu_energy_consummed),
            optional_sub(lhs.dram_energy_consummed, rhs.dram_energy_consummed)};
    }

    // Returns true if the current reporter is not a NoopSystemMetricReporter (defined below).
    bool has_reporter();

} // namespace shamsys

namespace shamsys {

    class AuroraSystemMetricReporter : public ISystemMetricReporter {
        public:
        std::optional<f64> get_rank_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                std::string output = shambase::popen_fetch_output("geopmread BOARD_ENERGY board 0");
                return std::stoull(output.c_str());
            }
            return std::nullopt;
        }

        std::optional<f64> get_gpu_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                std::string output = shambase::popen_fetch_output("geopmread GPU_ENERGY board 0");
                return std::stoull(output.c_str());
            }
            return std::nullopt;
        }

        std::optional<f64> get_cpu_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                std::string output = shambase::popen_fetch_output("geopmread CPU_ENERGY board 0");
                return std::stoull(output.c_str());
            }
            return std::nullopt;
        }

        std::optional<f64> get_dram_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                std::string output = shambase::popen_fetch_output("geopmread DRAM_ENERGY board 0");
                return std::stoull(output.c_str());
            }
            return std::nullopt;
        }

        bool support_rank_energy_consummed() override { return true; }
        bool support_gpu_energy_consummed() override { return true; }
        bool support_cpu_energy_consummed() override { return true; }
        bool support_dram_energy_consummed() override { return true; }
    };

    class IntelRAPLSystemMetricReport : public ISystemMetricReporter {
        public:
        std::optional<f64> get_rank_energy_consummed() override {
            if (shamcomm::is_main_node_rank()) {
                std::string output = shambase::popen_fetch_output(
                    "cat /sys/class/powercap/intel-rapl:0/energy_uj");
                return f64(std::stoull(output.c_str())) * 1e-6;
            }
            return std::nullopt;
        }

        std::optional<f64> get_gpu_energy_consummed() override { return std::nullopt; }

        std::optional<f64> get_cpu_energy_consummed() override { return std::nullopt; }

        std::optional<f64> get_dram_energy_consummed() override { return std::nullopt; }

        bool support_rank_energy_consummed() override { return true; }
        bool support_gpu_energy_consummed() override { return false; }
        bool support_cpu_energy_consummed() override { return false; }
        bool support_dram_energy_consummed() override { return false; }
    };

    class NoopSystemMetricReporter : public ISystemMetricReporter {
        public:
        std::optional<f64> get_rank_energy_consummed() override { return std::nullopt; }
        std::optional<f64> get_gpu_energy_consummed() override { return std::nullopt; }
        std::optional<f64> get_cpu_energy_consummed() override { return std::nullopt; }
        std::optional<f64> get_dram_energy_consummed() override { return std::nullopt; }

        bool support_rank_energy_consummed() override { return false; }
        bool support_gpu_energy_consummed() override { return false; }
        bool support_cpu_energy_consummed() override { return false; }
        bool support_dram_energy_consummed() override { return false; }
    };

    inline bool has_reporter() {
        auto &reporter = current_reporter();
        if (!reporter) {
            return false;
        }
        // dynamic_cast returns nullptr if the cast fails, so we check for that
        return dynamic_cast<NoopSystemMetricReporter *>(reporter.get()) == nullptr;
    }

    inline std::unique_ptr<ISystemMetricReporter> make_reporter(std::string_view reporter_name) {
        if (reporter_name == "aurora") {
            return std::make_unique<AuroraSystemMetricReporter>();
        } else if (reporter_name == "intel-rapl") {
            return std::make_unique<IntelRAPLSystemMetricReport>();
        }
        return std::make_unique<NoopSystemMetricReporter>();
    }

    inline std::unique_ptr<ISystemMetricReporter> make_reporter() {
        if (SHAM_SYSTEM_METRICS_REPORTER) {
            return make_reporter(*SHAM_SYSTEM_METRICS_REPORTER);
        }
        return std::make_unique<NoopSystemMetricReporter>();
    }

    /// test that there is no crashes
    inline void test_reporter(std::unique_ptr<ISystemMetricReporter> &reporter) {
        shambase::get_check_ref(reporter).get_rank_energy_consummed();
        shambase::get_check_ref(reporter).get_gpu_energy_consummed();
        shambase::get_check_ref(reporter).get_cpu_energy_consummed();
        shambase::get_check_ref(reporter).get_dram_energy_consummed();
    }

    inline std::unique_ptr<ISystemMetricReporter> &current_reporter() {
        static std::unique_ptr<ISystemMetricReporter> reporter = nullptr;
        if (!reporter) {
            reporter = make_reporter();
            test_reporter(reporter);
        }
        return reporter;
    }
} // namespace shamsys
