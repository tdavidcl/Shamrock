// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MHDConfig.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Out-of-line JSON (de)serialization for MHDConfig.
 */

#include "shambase/exception.hpp"
#include "shammodels/sph/config/MHDConfig.hpp"
#include <nlohmann/json.hpp>

namespace shammodels::sph {

    template<class Tvec>
    void to_json(nlohmann::json &j, const MHDConfig<Tvec> &p) {
        using T = MHDConfig<Tvec>;

        using None        = typename T::None;
        using IMHD        = typename T::IdealMHD_constrained_hyper_para;
        using NonIdealMHD = typename T::NonIdealMHD;

        if (const None *v = std::get_if<None>(&p.config)) {
            j = {
                {"mhd_type", "none"},
            };
        } else if (const IMHD *v = std::get_if<IMHD>(&p.config)) {
            j = {
                {"mhd_type", "ideal_mhd_constrained_hyper_para"},
                {"sigma_mhd", v->sigma_mhd},
                {"alpha_u", v->alpha_u},
            };
        } else if (const NonIdealMHD *v = std::get_if<NonIdealMHD>(&p.config)) {
            j = {
                {"mhd_type", "non_ideal_mhd"},
                {"sigma_mhd", v->sigma_mhd},
                {"alpha_u", v->alpha_u},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    void from_json(const nlohmann::json &j, MHDConfig<Tvec> &p) {
        using T     = MHDConfig<Tvec>;
        using Tscal = shambase::VecComponent<Tvec>;

        if (!j.contains("mhd_type")) {
            shambase::throw_with_loc<std::runtime_error>("no field mhd_type is found in this json");
        }

        std::string mhd_type;
        j.at("mhd_type").get_to(mhd_type);

        using None        = typename T::None;
        using IMHD        = typename T::IdealMHD_constrained_hyper_para;
        using NonIdealMHD = typename T::NonIdealMHD;

        if (mhd_type == "none") {
            p.set(None{});
        } else if (mhd_type == "ideal_mhd_constrained_hyper_para") {
            p.set(
                IMHD{
                    j.at("sigma_mhd").get<Tscal>(),
                    j.at("alpha_u").get<Tscal>(),
                });
        } else if (mhd_type == "non_ideal_mhd") {
            p.set(
                NonIdealMHD{
                    j.at("sigma_mhd").get<Tscal>(),
                    j.at("alpha_u").get<Tscal>(),
                });
        } else {
            shambase::throw_unimplemented("wtf !");
        }
    }

    template void to_json<f64_3>(nlohmann::json &j, const MHDConfig<f64_3> &p);
    template void from_json<f64_3>(const nlohmann::json &j, MHDConfig<f64_3> &p);

} // namespace shammodels::sph
