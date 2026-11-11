// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SolverConfigDragTests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Unit tests for the RAMSES drag config (Epstein drag, JSON, checks)
 */

#include "shammodels/ramses/SolverConfig.hpp"
#include "shamtest/shamtest.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

namespace {

    using Config = shammodels::basegodunov::SolverConfig<f64_3, i64_3>;
    using shammodels::basegodunov::EpsteinDragConfig;

    Config json_roundtrip(const Config &in) {
        nlohmann::json j = in;
        return nlohmann::json::parse(j.dump(4)).get<Config>();
    }

    /// REQUIRE_EQUAL needs a formattable type, so vectors are compared through a bool
    bool same(const std::vector<f64> &a, const std::vector<f64> &b) { return a == b; }

    Config dusty_config() {
        Config cfg{};
        cfg.dust_config                    = {shammodels::basegodunov::HB, 2};
        cfg.drag_config.drag_solver_config = shammodels::basegodunov::IRK1;
        return cfg;
    }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/SolverConfig::drag_config", 1) {

    { // static alphas by default, no Epstein drag
        Config cfg{};
        REQUIRE(!cfg.drag_config.epstein.has_value());
    }

    { // Epstein json roundtrip
        Config in              = dusty_config();
        in.drag_config.epstein = EpsteinDragConfig{{1e-3, 1e-2}, {2.3, 3.3}};

        Config out = json_roundtrip(in);
        REQUIRE(out.drag_config.epstein.has_value());
        if (out.drag_config.epstein) {
            REQUIRE(same(out.drag_config.epstein->grain_sizes, {1e-3, 1e-2}));
            REQUIRE(same(out.drag_config.epstein->grain_densities, {2.3, 3.3}));
        }
    }

    { // configs dumped before the Epstein drag was added only have static alphas
        Config in              = dusty_config();
        in.drag_config.alphas  = {1., 2.};
        in.drag_config.epstein = EpsteinDragConfig{{1e-3, 1e-2}, {2.3, 3.3}};

        nlohmann::json j = in;
        j.at("drag_config").erase("epstein");

        Config out = j.get<Config>();
        REQUIRE(!out.drag_config.epstein.has_value());
        REQUIRE(same(out.drag_config.alphas, {1., 2.}));
    }

    { // one value per dust species is required
        Config cfg             = dusty_config();
        cfg.drag_config.alphas = {1., 2.};
        cfg.check_config();

        cfg.drag_config.alphas = {1.};
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);

        cfg.drag_config.epstein = EpsteinDragConfig{{1e-3, 1e-2}, {2.3, 3.3}};
        cfg.check_config();

        cfg.drag_config.epstein->grain_densities = {2.3};
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);
    }
}
