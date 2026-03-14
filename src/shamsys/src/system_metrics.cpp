// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file system_metrics.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamsys/system_metrics.hpp"
#include "shamcomm/local_rank.hpp"
#include <cstdlib>
#include "shambase/aliases_int.hpp"
#include "shambase/popen.hpp"

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

    bool has_reporter() {
        auto &reporter = current_reporter();
        if (!reporter) {
            return false;
        }
        // dynamic_cast returns nullptr if the cast fails, so we check for that
        return dynamic_cast<NoopSystemMetricReporter *>(reporter.get()) == nullptr;
    }

    std::unique_ptr<ISystemMetricReporter> make_reporter(std::string_view reporter_name) {
        if (reporter_name == "aurora") {
            return std::make_unique<AuroraSystemMetricReporter>();
        } else if (reporter_name == "intel-rapl") {
            return std::make_unique<IntelRAPLSystemMetricReport>();
        } else if (reporter_name == "noop" || reporter_name == "none" || reporter_name == "") {
            return std::make_unique<NoopSystemMetricReporter>();
        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "Unknown system metrics reporter: {}, valid reporters are: aurora, intel-rapl, "
                "noop",
                reporter_name));
        }
        return std::make_unique<NoopSystemMetricReporter>();
    }

    std::unique_ptr<ISystemMetricReporter> make_reporter() {
        if (SHAM_SYSTEM_METRICS_REPORTER) {
            return make_reporter(*SHAM_SYSTEM_METRICS_REPORTER);
        }
        return std::make_unique<NoopSystemMetricReporter>();
    }

    /// test that there is no crashes
    void test_reporter(std::unique_ptr<ISystemMetricReporter> &reporter) {
        shambase::get_check_ref(reporter).get_rank_energy_consummed();
        shambase::get_check_ref(reporter).get_gpu_energy_consummed();
        shambase::get_check_ref(reporter).get_cpu_energy_consummed();
        shambase::get_check_ref(reporter).get_dram_energy_consummed();
    }

    std::unique_ptr<ISystemMetricReporter> &current_reporter() {
        static std::unique_ptr<ISystemMetricReporter> reporter = nullptr;
        if (!reporter) {
            reporter = make_reporter();
            test_reporter(reporter);
        }
        return reporter;
    }
} // namespace shamsys
