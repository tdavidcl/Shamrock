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
 * @file ExtForceConfig.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Noé Brucy (noe.brucy@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief External forces config of the Ramses solver + json serialization/deserialization
 */

#include "shambase/exception.hpp"
#include "shambase/overloaded.hpp"
#include "shambackends/type_convert.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/config/enum_ExtForceCoupling.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace shammodels::basegodunov {

    /**
     * @brief List of the external forces applied to the gas
     *
     * The accelerations of all the forces are summed (in the order of the list) before being
     * applied to the conservative state, with the time quadrature selected by `coupling`.
     */
    template<class Tvec>
    struct ExtForceConfig {
        using Tscal = shambase::VecComponent<Tvec>;

        /// Constant gravitational acceleration g
        struct ConstantGravity {
            Tvec g;
        };

        // Point mass gravity
        struct PointMassGravity {
            Tscal mass         = 1;
            Tvec pos           = Tvec(0.5, 0.5, 0.5);
            Tscal epsilon_soft = 0.05;
        };

        using Variant = std::variant<ConstantGravity, PointMassGravity>;

        /**
         * @brief Forces of the local shearing box frame (Hill's equations, unstratified)
         *
         * Same parameters as the common ``ShearingBoxForce(Omega_0, eta, q)`` of the SPH solver,
         * with x the radial, y the azimuthal and z the vertical axis (rotation along z), and
         * x' = x - x_c the radial coordinate relative to the center x_c of the box:
         *
         * \f[
         *  \mathbf{a} = 2 q \Omega_0^2 x' \mathbf{e}_x - 2 \Omega_0 \mathbf{e}_z \times \mathbf{v}
         *      + 2 \Omega_0 \eta \mathbf{e}_x \quad (\text{gas only for the } \eta \text{ term})
         * \f]
         *
         * i.e. a_x = 2 Omega_0 (q Omega_0 x' + v_y) (+ 2 Omega_0 eta for the gas),
         * a_y = -2 Omega_0 v_x. The equilibrium flow is v_y = -q Omega_0 x' (with the
         * shearing-periodic boundary, shear_speed = -q Omega_0 L_x).
         *
         * ``eta`` is the constant radial pressure-gradient forcing of the gas, given as the
         * headwind velocity eta_r v_K (eta_r the usual dimensionless parameter, v_K the Keplerian
         * velocity): without dust, the gas equilibrium is v_y = -q Omega_0 x' - eta. The dust does
         * not feel it (it is a pressure force).
         *
         * It is integrated with its own time quadrature (applied after the flux update, trapezoidal
         * rule implicit in the momentum, see ShearingBoxForceUtils.hpp), whatever the coupling of
         * the other forces.
         */
        struct ShearingBox {
            Tscal Omega_0;
            Tscal eta;
            Tscal q;
        };

        std::vector<Variant> ext_forces;

        /// Shearing box forces, unset if they are not enabled
        std::optional<ShearingBox> shearing_box = std::nullopt;

        /// Time quadrature of the external forces source term
        ExtForceCoupling coupling = ExtForceCoupling::Midpoint;

        inline void set_coupling(ExtForceCoupling mode) { coupling = mode; }

        inline void add_constant_gravity(Tvec g) { ext_forces.push_back(ConstantGravity{g}); }
        inline void add_point_mass_gravity(Tscal mass, Tvec pos, Tscal epsilon_soft) {
            if (!(std::isfinite(epsilon_soft) && epsilon_soft > 0)) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "point mass gravity: epsilon_soft must be finite and > 0");
            }
            ext_forces.push_back(PointMassGravity{mass, pos, epsilon_soft});
        }

        inline void add_shearing_box(Tscal Omega_0, Tscal eta, Tscal q) {
            if (shearing_box.has_value()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "shearing box forces: they can only be added once");
            }
            if (!(std::isfinite(Omega_0) && std::isfinite(eta) && std::isfinite(q))) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "shearing box forces: Omega_0, eta and q must be finite");
            }
            shearing_box = ShearingBox{Omega_0, eta, q};
        }

        inline bool has_shearing_box() const { return shearing_box.has_value(); }

        inline bool is_coordinates_needed() const {
            if (has_shearing_box()) {
                return true;
            }
            return std::any_of(ext_forces.begin(), ext_forces.end(), [](const Variant &f) {
                return std::holds_alternative<PointMassGravity>(f);
            });
        }

        /// true if forces applied through the acceleration field a_ext are enabled (all of them
        /// except the shearing box forces)
        inline bool has_ext_forces() const { return !ext_forces.empty(); }
    };

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const ExtForceConfig<Tvec> &p) {
        using EF = ExtForceConfig<Tvec>;

        nlohmann::json force_list = nlohmann::json::array();
        for (const auto &var_force : p.ext_forces) {
            std::visit(
                shambase::overloaded{
                    [&](const typename EF::ConstantGravity &cfg) {
                        force_list.push_back(
                            nlohmann::json{{"force_type", "constant_gravity"}, {"g", cfg.g}});
                    },
                    [&](const typename EF::PointMassGravity &cfg) {
                        force_list.push_back(
                            nlohmann::json{
                                {"force_type", "point_mass_gravity"},
                                {"mass", cfg.mass},
                                {"pos", cfg.pos},
                                {"epsilon_soft", cfg.epsilon_soft}});
                    },
                },
                var_force);
        }

        j = nlohmann::json{{"force_list", force_list}, {"coupling", p.coupling}};

        // only dumped when enabled, so that configs without it are unchanged
        if (p.shearing_box.has_value()) {
            const auto &sb = *p.shearing_box;
            j["shearing_box"]
                = nlohmann::json{{"Omega_0", sb.Omega_0}, {"eta", sb.eta}, {"q", sb.q}};
        }
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, ExtForceConfig<Tvec> &p) {
        using Tscal = shambase::VecComponent<Tvec>;

        p.ext_forces.clear();

        // configs dumped before the shearing box forces existed have none
        p.shearing_box = std::nullopt;
        if (j.contains("shearing_box") && !j.at("shearing_box").is_null()) {
            const auto &jsb = j.at("shearing_box");
            p.add_shearing_box(
                jsb.at("Omega_0").get<Tscal>(),
                jsb.at("eta").get<Tscal>(),
                jsb.at("q").get<Tscal>());
        }

        // configs dumped before the coupling was configurable used the midpoint rule
        p.coupling = ExtForceCoupling::Midpoint;
        if (j.contains("coupling")) {
            j.at("coupling").get_to(p.coupling);
        }

        for (const auto &jforce : j.at("force_list")) {
            const std::string force_type = jforce.at("force_type").get<std::string>();
            if (force_type == "constant_gravity") {
                p.add_constant_gravity(jforce.at("g").get<Tvec>());
            } else if (force_type == "point_mass_gravity") {
                p.add_point_mass_gravity(
                    jforce.at("mass").get<Tscal>(),
                    jforce.at("pos").get<Tvec>(),
                    jforce.at("epsilon_soft").get<Tscal>());
            } else {
                shambase::throw_with_loc<std::runtime_error>(
                    "Invalid external force type: " + force_type);
            }
        }
    }

} // namespace shammodels::basegodunov
