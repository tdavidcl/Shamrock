// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BCConfig.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Out-of-line JSON (de)serialization for BCConfig.
 */

#include "shambase/exception.hpp"
#include "shambackends/type_convert_json.hpp"
#include "shammodels/sph/config/BCConfig.hpp"
#include <nlohmann/json.hpp>

namespace shammodels::sph {

    template<class Tvec>
    void to_json(nlohmann::json &j, const BCConfig<Tvec> &p) {
        using T = BCConfig<Tvec>;

        using Free             = typename T::Free;
        using Periodic         = typename T::Periodic;
        using ShearingPeriodic = typename T::ShearingPeriodic;

        if (const Free *v = std::get_if<Free>(&p.config)) {
            j = {
                {"bc_type", "free"},
            };
        } else if (const Periodic *v = std::get_if<Periodic>(&p.config)) {
            j = {
                {"bc_type", "periodic"},
            };
        } else if (const ShearingPeriodic *v = std::get_if<ShearingPeriodic>(&p.config)) {
            j = {
                {"bc_type", "shearing_periodic"},
                {"shear_base", v->shear_base},
                {"shear_dir", v->shear_dir},
                {"shear_speed", v->shear_speed},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    void from_json(const nlohmann::json &j, BCConfig<Tvec> &p) {
        using T     = BCConfig<Tvec>;
        using Tscal = shambase::VecComponent<Tvec>;

        if (!j.contains("bc_type")) {
            shambase::throw_with_loc<std::runtime_error>("no field eos_type is found in this json");
        }

        std::string bc_type;
        j.at("bc_type").get_to(bc_type);

        if (bc_type == "free") {
            p.set_free();
        } else if (bc_type == "periodic") {
            p.set_periodic();
        } else if (bc_type == "shearing_periodic") {
            p.set_shearing_periodic(
                j.at("shear_base").get<i32_3>(),
                j.at("shear_dir").get<i32_3>(),
                j.at("speed").get<Tscal>());
        } else {
            shambase::throw_unimplemented("wtf !");
        }
    }

    template void to_json<f64_3>(nlohmann::json &j, const BCConfig<f64_3> &p);
    template void from_json<f64_3>(const nlohmann::json &j, BCConfig<f64_3> &p);

} // namespace shammodels::sph
