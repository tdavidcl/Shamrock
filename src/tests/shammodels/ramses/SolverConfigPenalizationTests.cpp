// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/math.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamtest/shamtest.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <variant>

namespace {

    using Config       = shammodels::basegodunov::SolverConfig<f64_3, i64_3>;
    using Penalization = shammodels::basegodunov::PenalizationConfig<f64_3>;
    using Cylinder     = Penalization::Cylinder;
    using Sphere       = Penalization::Sphere;
    using Box          = Penalization::Box;

    Config json_roundtrip(const Config &in) {
        nlohmann::json j = in;
        return nlohmann::json::parse(j.dump(4)).get<Config>();
    }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/SolverConfig::penalization", 1) {

    { // off by default
        Config cfg{};
        REQUIRE(!cfg.is_penalization_on());
        cfg.check_config();
    }

    { // the cylinder axis is normalized
        Penalization p{};
        p.add_cylinder({1, 1, 0}, {0, 0, 3}, 0.5, 1e-3, {0, 0, 0});
        const Cylinder &c = std::get<Cylinder>(p.zones.at(0).shape);
        REQUIRE(sham::equals(c.axis, f64_3(0, 0, 1)));

        REQUIRE_EXCEPTION_THROW(
            p.add_cylinder({1, 1, 0}, {0, 0, 0}, 0.5, 1e-3, {0, 0, 0}), std::invalid_argument);
    }

    { // shape containment
        Cylinder c{{1, 1, 0}, {0, 0, 1}, 0.5};
        REQUIRE(c.contains({1.2, 1.2, 42}));
        REQUIRE(!c.contains({1.6, 1, 0}));

        Sphere s{{0, 0, 0}, 1};
        REQUIRE(s.contains({0.5, 0.5, 0.5}));
        REQUIRE(!s.contains({0.6, 0.6, 0.6}));

        Box b{{0, 0, 0}, {1, 2, 3}};
        REQUIRE(b.contains({0.5, 1.5, 2.5}));
        REQUIRE(!b.contains({0.5, 2.5, 2.5}));
    }

    { // json roundtrip
        Config in{};
        in.penalization_config.add_cylinder({1, 1, 0}, {0, 0, 1}, 0.125, 1e-3, {0, 0, 0});
        in.penalization_config.add_sphere({0, 1, 0}, 0.5, 0.1, {1, 0, 0}, 2.0);
        in.penalization_config.add_box({0, 0, 0}, {0.25, 2, 1}, 0, {0.3, 0, 0}, 1.0, 0.6);

        Config out = json_roundtrip(in);
        REQUIRE(out.is_penalization_on());
        REQUIRE_EQUAL(out.penalization_config.zones.size(), size_t(3));

        const auto &z0 = out.penalization_config.zones.at(0);
        const auto &c  = std::get<Cylinder>(z0.shape);
        REQUIRE(sham::equals(c.center, f64_3(1, 1, 0)));
        REQUIRE(sham::equals(c.axis, f64_3(0, 0, 1)));
        REQUIRE_EQUAL(c.radius, 0.125);
        REQUIRE_EQUAL(z0.eta, 1e-3);
        REQUIRE(!z0.target_rho.has_value());
        REQUIRE(!z0.target_press.has_value());

        const auto &z1 = out.penalization_config.zones.at(1);
        REQUIRE_EQUAL(std::get<Sphere>(z1.shape).radius, 0.5);
        REQUIRE(sham::equals(z1.target_vel, f64_3(1, 0, 0)));
        REQUIRE_EQUAL(z1.target_rho.value_or(0), 2.0);
        REQUIRE(!z1.target_press.has_value());

        const auto &z2 = out.penalization_config.zones.at(2);
        REQUIRE(sham::equals(std::get<Box>(z2.shape).upper, f64_3(0.25, 2, 1)));
        REQUIRE_EQUAL(z2.eta, 0.0);
        REQUIRE_EQUAL(z2.target_rho.value_or(0), 1.0);
        REQUIRE_EQUAL(z2.target_press.value_or(0), 0.6);
    }

    { // configs dumped before penalization existed have no zone
        nlohmann::json j = Config{};
        j.erase("penalization_config");
        Config out = j.get<Config>();
        REQUIRE(!out.is_penalization_on());
    }

    { // invalid parameters are rejected
        Config cfg{};
        cfg.penalization_config.add_sphere({0, 0, 0}, 1, -1, {0, 0, 0});
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);

        cfg.penalization_config.zones.clear();
        cfg.penalization_config.add_sphere({0, 0, 0}, 1, 0, {0, 0, 0}, 0.0);
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);

        cfg.penalization_config.zones.clear();
        cfg.penalization_config.add_sphere({0, 0, 0}, 1, 0, {0, 0, 0}, 1.0, -1.0);
        REQUIRE_EXCEPTION_THROW(cfg.check_config(), std::invalid_argument);
    }
}
