// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ExtForceConfig.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Out-of-line JSON (de)serialization for ExtForceConfig.
 */

#include "shambase/exception.hpp"
#include "shambackends/type_convert_json.hpp"
#include "shammodels/common/ExtForceConfig.hpp"
#include <nlohmann/json.hpp>

namespace shammodels {

    template<class Tvec>
    void to_json(nlohmann::json &j, const ExtForceVariant<Tvec> &p) {
        using T = ExtForceVariant<Tvec>;

        using PointMass             = typename T::PointMass;
        using PN_PW                 = typename T::PN_PW;
        using LenseThirring         = typename T::LenseThirring;
        using ShearingBoxForce      = typename T::ShearingBoxForce;
        using VerticalDiscPotential = typename T::VerticalDiscPotential;
        using VelocityDissipation   = typename T::VelocityDissipation;

        if (const PointMass *v = std::get_if<PointMass>(&p.val)) {
            j = {
                {"force_type", "point_mass"}, {"central_mass", v->central_mass}, {"Racc", v->Racc}};

        } else if (const PN_PW *v = std::get_if<PN_PW>(&p.val)) {
            j
                = {{"force_type", "paczynski_wiita"},
                   {"central_mass", v->central_mass},
                   {"central_pos", v->central_pos},
                   {"Racc", v->Racc}};
        } else if (const LenseThirring *v = std::get_if<LenseThirring>(&p.val)) {
            j = {
                {"force_type", "lense_thirring"},
                {"central_mass", v->central_mass},
                {"Racc", v->Racc},
                {"a_spin", v->a_spin},
                {"dir_spin", v->dir_spin},
            };
        } else if (const ShearingBoxForce *v = std::get_if<ShearingBoxForce>(&p.val)) {
            j = {
                {"force_type", "shearing_box_force"},
                {"shear_base", v->shear_base},
                {"shear_dir", v->shear_dir},
                {"Omega_0", v->Omega_0},
                {"eta", v->eta},
                {"q", v->q},
            };
        } else if (const VerticalDiscPotential *v = std::get_if<VerticalDiscPotential>(&p.val)) {
            j
                = {{"force_type", "vertical_disc_potential"},
                   {"central_mass", v->central_mass},
                   {"R0", v->R0}};
        } else if (const VelocityDissipation *v = std::get_if<VelocityDissipation>(&p.val)) {
            j = {{"force_type", "velocity_dissipation"}, {"eta", v->eta}};
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    void from_json(const nlohmann::json &j, ExtForceVariant<Tvec> &p) {
        using Tscal = shambase::VecComponent<Tvec>;
        using T     = ExtForceVariant<Tvec>;

        if (!j.contains("force_type")) {
            shambase::throw_with_loc<std::runtime_error>("no field eos_type is found in this json");
        }

        std::string force_type;
        j.at("force_type").get_to(force_type);

        using PointMass             = typename T::PointMass;
        using PN_PW                 = typename T::PN_PW;
        using LenseThirring         = typename T::LenseThirring;
        using ShearingBoxForce      = typename T::ShearingBoxForce;
        using VerticalDiscPotential = typename T::VerticalDiscPotential;
        using VelocityDissipation   = typename T::VelocityDissipation;

        if (force_type == "point_mass") {
            p.val = PointMass{
                j.at("central_mass").get<Tscal>(),
                j.at("Racc").get<Tscal>(),
            };
        } else if (force_type == "paczynski_wiita") {
            p.val = PN_PW{
                j.at("central_mass").get<Tscal>(),
                j.at("central_pos").get<Tvec>(),
                j.at("Racc").get<Tscal>(),
            };
        } else if (force_type == "lense_thirring") {
            p.val = LenseThirring{
                j.at("central_mass").get<Tscal>(),
                j.at("Racc").get<Tscal>(),
                j.at("a_spin").get<Tscal>(),
                j.at("dir_spin").get<Tvec>(),
            };
        } else if (force_type == "shearing_box_force") {
            p.val = ShearingBoxForce{
                j.at("shear_base").get<i32_3>(),
                j.at("shear_dir").get<i32_3>(),
                j.at("Omega_0").get<Tscal>(),
                j.at("eta").get<Tscal>(),
                j.at("q").get<Tscal>(),
            };
        } else if (force_type == "vertical_disc_potential") {
            p.val = VerticalDiscPotential{
                j.at("central_mass").get<Tscal>(),
                j.at("R0").get<Tscal>(),
            };
        } else if (force_type == "velocity_dissipation") {
            p.val = VelocityDissipation{j.at("eta").get<Tscal>()};
        } else {
            shambase::throw_unimplemented("wtf !");
        }
    }

    template<class Tvec>
    void to_json(nlohmann::json &j, const ExtForceConfig<Tvec> &p) {
        j = {{"force_list", p.ext_forces}};
    }

    template<class Tvec>
    void from_json(const nlohmann::json &j, ExtForceConfig<Tvec> &p) {
        j.at("force_list").get_to(p.ext_forces);
    }

    template void to_json<f64_3>(nlohmann::json &j, const ExtForceVariant<f64_3> &p);
    template void from_json<f64_3>(const nlohmann::json &j, ExtForceVariant<f64_3> &p);
    template void to_json<f64_3>(nlohmann::json &j, const ExtForceConfig<f64_3> &p);
    template void from_json<f64_3>(const nlohmann::json &j, ExtForceConfig<f64_3> &p);

} // namespace shammodels
