// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AVConfig.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Out-of-line JSON (de)serialization for AVConfig.
 */

#include "shammodels/sph/config/AVConfig.hpp"
#include "shamrock/io/json_variant.hpp"
#include <nlohmann/json.hpp>

namespace shammodels::sph {

    template<class Tscal>
    void to_json(nlohmann::json &j, const AVConfig_None<Tscal> &p) {}

    template<class Tscal>
    void from_json(const nlohmann::json &j, AVConfig_None<Tscal> &p) {
        p = {};
    }

    template<class Tscal>
    void to_json(nlohmann::json &j, const AVConfig_Constant<Tscal> &p) {
        j = {
            {"alpha_u", p.alpha_u},
            {"alpha_AV", p.alpha_AV},
            {"beta_AV", p.beta_AV},
        };
    }

    template<class Tscal>
    void from_json(const nlohmann::json &j, AVConfig_Constant<Tscal> &p) {
        j.at("alpha_u").get_to(p.alpha_u);
        j.at("alpha_AV").get_to(p.alpha_AV);
        j.at("beta_AV").get_to(p.beta_AV);
    }

    template<class Tscal>
    void to_json(nlohmann::json &j, const AVConfig_VaryingMM97<Tscal> &p) {
        j = {
            {"alpha_min", p.alpha_min},
            {"alpha_max", p.alpha_max},
            {"sigma_decay", p.sigma_decay},
            {"alpha_u", p.alpha_u},
            {"beta_AV", p.beta_AV},
        };
    }

    template<class Tscal>
    void from_json(const nlohmann::json &j, AVConfig_VaryingMM97<Tscal> &p) {
        j.at("alpha_min").get_to(p.alpha_min);
        j.at("alpha_max").get_to(p.alpha_max);
        j.at("sigma_decay").get_to(p.sigma_decay);
        j.at("alpha_u").get_to(p.alpha_u);
        j.at("beta_AV").get_to(p.beta_AV);
    }

    template<class Tscal>
    void to_json(nlohmann::json &j, const AVConfig_VaryingCD10<Tscal> &p) {
        j = {
            {"alpha_min", p.alpha_min},
            {"alpha_max", p.alpha_max},
            {"sigma_decay", p.sigma_decay},
            {"alpha_u", p.alpha_u},
            {"beta_AV", p.beta_AV},
        };
    }

    template<class Tscal>
    void from_json(const nlohmann::json &j, AVConfig_VaryingCD10<Tscal> &p) {
        j.at("alpha_min").get_to(p.alpha_min);
        j.at("alpha_max").get_to(p.alpha_max);
        j.at("sigma_decay").get_to(p.sigma_decay);
        j.at("alpha_u").get_to(p.alpha_u);
        j.at("beta_AV").get_to(p.beta_AV);
    }

    template<class Tscal>
    void to_json(nlohmann::json &j, const AVConfig_ConstantDisc<Tscal> &p) {
        j = {
            {"alpha_AV", p.alpha_AV},
            {"alpha_u", p.alpha_u},
            {"beta_AV", p.beta_AV},
        };
    }

    template<class Tscal>
    void from_json(const nlohmann::json &j, AVConfig_ConstantDisc<Tscal> &p) {
        j.at("alpha_AV").get_to(p.alpha_AV);
        j.at("alpha_u").get_to(p.alpha_u);
        j.at("beta_AV").get_to(p.beta_AV);
    }

    template<class Tvec>
    void to_json(nlohmann::json &j, const AVConfig<Tvec> &p) {
        std::visit(
            [&](const auto &value) {
                j         = value;
                j["type"] = value.variant_type_name;
            },
            p.config);
    }

    template<class Tvec>
    void from_json(const nlohmann::json &j, AVConfig<Tvec> &p) {
        if (!j.contains("type") && !j.contains("av_type")) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "neither \"type\" nor \"av_type\" in this json, can not infer type json=\n"
                + j.dump(4));
        }

        std::string av_type;
        if (j.contains("type")) {
            j.at("type").get_to(av_type);
        } else {
            j.at("av_type").get_to(av_type);
        }

        shamrock::json_deserialize_variant(j, av_type, p.config);
    }

    template void to_json<f64>(nlohmann::json &j, const AVConfig_None<f64> &p);
    template void from_json<f64>(const nlohmann::json &j, AVConfig_None<f64> &p);
    template void to_json<f64>(nlohmann::json &j, const AVConfig_Constant<f64> &p);
    template void from_json<f64>(const nlohmann::json &j, AVConfig_Constant<f64> &p);
    template void to_json<f64>(nlohmann::json &j, const AVConfig_VaryingMM97<f64> &p);
    template void from_json<f64>(const nlohmann::json &j, AVConfig_VaryingMM97<f64> &p);
    template void to_json<f64>(nlohmann::json &j, const AVConfig_VaryingCD10<f64> &p);
    template void from_json<f64>(const nlohmann::json &j, AVConfig_VaryingCD10<f64> &p);
    template void to_json<f64>(nlohmann::json &j, const AVConfig_ConstantDisc<f64> &p);
    template void from_json<f64>(const nlohmann::json &j, AVConfig_ConstantDisc<f64> &p);
    template void to_json<f64_3>(nlohmann::json &j, const AVConfig<f64_3> &p);
    template void from_json<f64_3>(const nlohmann::json &j, AVConfig<f64_3> &p);

} // namespace shammodels::sph
