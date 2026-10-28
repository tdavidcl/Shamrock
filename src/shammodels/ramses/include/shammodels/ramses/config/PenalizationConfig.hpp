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
 * @file PenalizationConfig.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Volume penalization zones (immersed obstacles, forcing bands) + json serialization
 *
 */

#include "shambase/exception.hpp"
#include "shambase/overloaded.hpp"
#include "shambase/string.hpp"
#include "shambackends/math.hpp"
#include "shambackends/type_convert.hpp"
#include "shambackends/vec.hpp"
#include <nlohmann/json.hpp>
#include <shamrock/io/json_std_optional.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace shammodels::basegodunov {

    /**
     * @brief Volume penalization (Brinkman) zones of the Ramses solver
     *
     * Each zone is a region of space in which the gas state is relaxed towards a target state on
     * a timescale \f$ \eta \f$ by the source term
     * \f[
     *   \partial_t q = - \frac{\chi(\mathbf{x})}{\eta} (q - q_{\rm target})
     * \f]
     * where \f$ \chi \f$ is 1 for cells whose center is inside the zone (signed distance
     * \f$ \phi \leq 0 \f$, see the shapes sdf functions) and 0 elsewhere. The
     * relaxed quantities are the velocity (always), the density and the pressure (if a target is
     * set). It is integrated implicitly, so any \f$ \eta \geq 0 \f$ is stable (\f$ \eta = 0 \f$
     * sets the target state in one step).
     *
     * Typical uses are an immersed solid obstacle (target velocity 0, no density or pressure
     * target) and an inflow forcing band next to an outflow boundary (full target state).
     */
    template<class Tvec>
    struct PenalizationConfig {
        using Tscal = shambase::VecComponent<Tvec>;

        /// Infinite cylinder of axis passing through center along axis (unit vector)
        struct Cylinder {
            Tvec center;
            Tvec axis;
            Tscal radius;

            /// signed distance to the surface (negative inside)
            inline Tscal sdf(Tvec pos) const {
                Tvec d      = pos - center;
                Tvec d_perp = d - sycl::dot(d, axis) * axis;
                return sycl::length(d_perp) - radius;
            }
        };

        /// Sphere of given center and radius
        struct Sphere {
            Tvec center;
            Tscal radius;

            /// signed distance to the surface (negative inside)
            inline Tscal sdf(Tvec pos) const { return sycl::length(pos - center) - radius; }
        };

        /// Axis aligned box [lower, upper]
        struct Box {
            Tvec lower;
            Tvec upper;

            /// signed distance to the surface (negative inside)
            inline Tscal sdf(Tvec pos) const {
                // distance to the box faces along each axis, positive outside
                Tvec q = sycl::fabs(pos - (lower + upper) / 2) - (upper - lower) / 2;

                Tvec q_out    = sycl::fmax(q, Tvec{});
                Tscal max_q   = sycl::fmax(sycl::fmax(q.x(), q.y()), q.z());
                Tscal dist_in = sycl::fmin(max_q, Tscal{0});
                return sycl::length(q_out) + dist_in;
            }
        };

        using Shape = std::variant<Cylinder, Sphere, Box>;

        /// A penalization zone: a shape, a relaxation timescale and a target state
        struct Zone {
            Shape shape;
            /// relaxation timescale (0 sets the target state in one step)
            Tscal eta;
            Tvec target_vel;
            /// if set the density is relaxed toward it, otherwise it is left unchanged
            std::optional<Tscal> target_rho;
            /// if set the pressure is relaxed toward it, otherwise the specific internal energy is
            /// kept
            std::optional<Tscal> target_press;
        };

        std::vector<Zone> zones;

        inline bool is_enabled() const { return !zones.empty(); }

        inline void add_zone(Zone zone) {
            if (Cylinder *cyl = std::get_if<Cylinder>(&zone.shape)) {
                Tscal norm = sycl::length(cyl->axis);
                if (!(norm > 0)) {
                    shambase::throw_with_loc<std::invalid_argument>(
                        "the axis of a penalization cylinder must be non zero");
                }
                cyl->axis /= norm;
            }
            zones.push_back(std::move(zone));
        }

        inline void add_cylinder(
            Tvec center,
            Tvec axis,
            Tscal radius,
            Tscal eta,
            Tvec target_vel,
            std::optional<Tscal> target_rho   = std::nullopt,
            std::optional<Tscal> target_press = std::nullopt) {
            add_zone(
                Zone{Cylinder{center, axis, radius}, eta, target_vel, target_rho, target_press});
        }

        inline void add_sphere(
            Tvec center,
            Tscal radius,
            Tscal eta,
            Tvec target_vel,
            std::optional<Tscal> target_rho   = std::nullopt,
            std::optional<Tscal> target_press = std::nullopt) {
            add_zone(Zone{Sphere{center, radius}, eta, target_vel, target_rho, target_press});
        }

        inline void add_box(
            Tvec lower,
            Tvec upper,
            Tscal eta,
            Tvec target_vel,
            std::optional<Tscal> target_rho   = std::nullopt,
            std::optional<Tscal> target_press = std::nullopt) {
            add_zone(Zone{Box{lower, upper}, eta, target_vel, target_rho, target_press});
        }

        inline void check_config() const {
            for (const Zone &zone : zones) {
                if (!(zone.eta >= 0)) {
                    shambase::throw_with_loc<std::invalid_argument>(
                        sham::format("penalization zone eta must be >= 0, got {}", zone.eta));
                }
                if (zone.target_rho && !(*zone.target_rho > 0)) {
                    shambase::throw_with_loc<std::invalid_argument>(sham::format(
                        "penalization zone target_rho must be > 0, got {}", *zone.target_rho));
                }
                if (zone.target_press && !(*zone.target_press >= 0)) {
                    shambase::throw_with_loc<std::invalid_argument>(sham::format(
                        "penalization zone target_press must be >= 0, got {}", *zone.target_press));
                }
            }
        }
    };

    template<class Tvec>
    inline void penalization_zone_to_json(
        nlohmann::json &j, const typename PenalizationConfig<Tvec>::Zone &p) {
        using Cfg = PenalizationConfig<Tvec>;

        nlohmann::json shape_j;
        std::visit(
            shambase::overloaded{
                [&](const typename Cfg::Cylinder &s) {
                    shape_j
                        = {{"type", "cylinder"},
                           {"center", s.center},
                           {"axis", s.axis},
                           {"radius", s.radius}};
                },
                [&](const typename Cfg::Sphere &s) {
                    shape_j = {{"type", "sphere"}, {"center", s.center}, {"radius", s.radius}};
                },
                [&](const typename Cfg::Box &s) {
                    shape_j = {{"type", "box"}, {"lower", s.lower}, {"upper", s.upper}};
                },
            },
            p.shape);

        j = nlohmann::json{
            {"shape", shape_j},
            {"eta", p.eta},
            {"target_vel", p.target_vel},
            {"target_rho", p.target_rho},
            {"target_press", p.target_press}};
    }

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const PenalizationConfig<Tvec> &p) {
        j = nlohmann::json::array();
        for (const auto &zone : p.zones) {
            nlohmann::json zone_j;
            penalization_zone_to_json<Tvec>(zone_j, zone);
            j.push_back(zone_j);
        }
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, PenalizationConfig<Tvec> &p) {
        using Cfg   = PenalizationConfig<Tvec>;
        using Tscal = shambase::VecComponent<Tvec>;

        p.zones.clear();
        for (const nlohmann::json &zone_j : j) {
            const nlohmann::json &shape_j = zone_j.at("shape");
            const std::string type        = shape_j.at("type").get<std::string>();

            typename Cfg::Shape shape;
            if (type == "cylinder") {
                shape = typename Cfg::Cylinder{
                    shape_j.at("center").get<Tvec>(),
                    shape_j.at("axis").get<Tvec>(),
                    shape_j.at("radius").get<Tscal>()};
            } else if (type == "sphere") {
                shape = typename Cfg::Sphere{
                    shape_j.at("center").get<Tvec>(), shape_j.at("radius").get<Tscal>()};
            } else if (type == "box") {
                shape = typename Cfg::Box{
                    shape_j.at("lower").get<Tvec>(), shape_j.at("upper").get<Tvec>()};
            } else {
                shambase::throw_with_loc<std::runtime_error>(
                    "Invalid penalization shape type: " + type);
            }

            p.add_zone(
                typename Cfg::Zone{
                    shape,
                    zone_j.at("eta").get<Tscal>(),
                    zone_j.at("target_vel").get<Tvec>(),
                    zone_j.at("target_rho").get<std::optional<Tscal>>(),
                    zone_j.at("target_press").get<std::optional<Tscal>>()});
        }
    }

} // namespace shammodels::basegodunov
