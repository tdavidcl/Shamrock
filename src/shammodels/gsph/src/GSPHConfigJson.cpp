// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GSPHConfigJson.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Out-of-line JSON (de)serialization for GSPH config types.
 */

#include "shambase/exception.hpp"
#include "shammodels/gsph/config/ForceFormulationConfig.hpp"
#include "shammodels/gsph/config/ReconstructConfig.hpp"
#include "shammodels/gsph/config/RiemannConfig.hpp"
#include <nlohmann/json.hpp>

namespace shammodels::gsph {

    template<class Tvec>
    void to_json(nlohmann::json &j, const RiemannConfig<Tvec> &p) {
        using T         = RiemannConfig<Tvec>;
        using Iterative = typename T::Iterative;
        using Exact     = typename T::Exact;
        using HLLC      = typename T::HLLC;
        using Roe       = typename T::Roe;

        if (const Iterative *v = std::get_if<Iterative>(&p.config)) {
            j = {
                {"riemann_type", "iterative"},
                {"tol", v->tol},
                {"max_iter", v->max_iter},
            };
        } else if (const Exact *v = std::get_if<Exact>(&p.config)) {
            j = {
                {"riemann_type", "exact"},
                {"tol", v->tol},
                {"max_iter", v->max_iter},
            };
        } else if (std::get_if<HLLC>(&p.config)) {
            j = {
                {"riemann_type", "hllc"},
            };
        } else if (const Roe *v = std::get_if<Roe>(&p.config)) {
            j = {
                {"riemann_type", "roe"},
                {"entropy_fix", v->entropy_fix},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    void from_json(const nlohmann::json &j, RiemannConfig<Tvec> &p) {
        using T     = RiemannConfig<Tvec>;
        using Tscal = shambase::VecComponent<Tvec>;

        if (!j.contains("riemann_type")) {
            shambase::throw_with_loc<std::runtime_error>(
                "no field riemann_type is found in this json");
        }

        std::string riemann_type;
        j.at("riemann_type").get_to(riemann_type);

        using Iterative = typename T::Iterative;
        using Exact     = typename T::Exact;
        using HLLC      = typename T::HLLC;
        using Roe       = typename T::Roe;

        if (riemann_type == "iterative") {
            p.set(Iterative{j.at("tol").get<Tscal>(), j.at("max_iter").get<u32>()});
        } else if (riemann_type == "exact") {
            p.set(Exact{j.at("tol").get<Tscal>(), j.value("max_iter", Exact{}.max_iter)});
        } else if (riemann_type == "hllc") {
            p.set(HLLC{});
        } else if (riemann_type == "roe") {
            p.set(Roe{j.at("entropy_fix").get<Tscal>()});
        } else {
            shambase::throw_unimplemented("Unknown Riemann solver type: " + riemann_type);
        }
    }

    template<class Tvec>
    void to_json(nlohmann::json &j, const ReconstructConfig<Tvec> &p) {
        using T                 = ReconstructConfig<Tvec>;
        using PiecewiseConstant = typename T::PiecewiseConstant;
        using MUSCL             = typename T::MUSCL;
        using Limiter           = typename T::Limiter;

        if (std::get_if<PiecewiseConstant>(&p.config)) {
            j = {
                {"reconstruct_type", "piecewise_constant"},
            };
        } else if (const MUSCL *v = std::get_if<MUSCL>(&p.config)) {
            std::string limiter_str;
            switch (v->limiter) {
            case Limiter::VanLeer : limiter_str = "vanleer"; break;
            case Limiter::Minmod  : limiter_str = "minmod"; break;
            case Limiter::Superbee: limiter_str = "superbee"; break;
            case Limiter::MC      : limiter_str = "mc"; break;
            }
            j = {
                {"reconstruct_type", "muscl"},
                {"limiter", limiter_str},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    void from_json(const nlohmann::json &j, ReconstructConfig<Tvec> &p) {
        using T                 = ReconstructConfig<Tvec>;
        using PiecewiseConstant = typename T::PiecewiseConstant;
        using MUSCL             = typename T::MUSCL;
        using Limiter           = typename T::Limiter;

        if (!j.contains("reconstruct_type")) {
            shambase::throw_with_loc<std::runtime_error>(
                "no field reconstruct_type is found in this json");
        }

        std::string reconstruct_type;
        j.at("reconstruct_type").get_to(reconstruct_type);

        if (reconstruct_type == "piecewise_constant") {
            p.set(PiecewiseConstant{});
        } else if (reconstruct_type == "muscl") {
            std::string limiter_str;
            j.at("limiter").get_to(limiter_str);

            Limiter limiter;
            if (limiter_str == "vanleer") {
                limiter = Limiter::VanLeer;
            } else if (limiter_str == "minmod") {
                limiter = Limiter::Minmod;
            } else if (limiter_str == "superbee") {
                limiter = Limiter::Superbee;
            } else if (limiter_str == "mc") {
                limiter = Limiter::MC;
            } else {
                shambase::throw_unimplemented("Unknown limiter type: " + limiter_str);
            }

            p.set(MUSCL{limiter});
        } else {
            shambase::throw_unimplemented("Unknown reconstruction type: " + reconstruct_type);
        }
    }

    template<class Tvec>
    void to_json(nlohmann::json &j, const ForceFormulationConfig<Tvec> &p) {
        using T            = ForceFormulationConfig<Tvec>;
        using ChaWhitworth = typename T::ChaWhitworth;
        using InutsukaV2   = typename T::InutsukaV2;

        if (std::get_if<ChaWhitworth>(&p.config)) {
            j = {
                {"force_formulation", "cha_whitworth"},
            };
        } else if (std::get_if<InutsukaV2>(&p.config)) {
            j = {
                {"force_formulation", "inutsuka_v2"},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    void from_json(const nlohmann::json &j, ForceFormulationConfig<Tvec> &p) {
        using T            = ForceFormulationConfig<Tvec>;
        using ChaWhitworth = typename T::ChaWhitworth;
        using InutsukaV2   = typename T::InutsukaV2;

        if (!j.contains("force_formulation")) {
            shambase::throw_with_loc<std::runtime_error>(
                "no field force_formulation is found in this json");
        }

        std::string force_formulation;
        j.at("force_formulation").get_to(force_formulation);

        if (force_formulation == "cha_whitworth") {
            p.set(ChaWhitworth{});
        } else if (force_formulation == "inutsuka_v2") {
            p.set(InutsukaV2{});
        } else {
            shambase::throw_unimplemented("Unknown force formulation type: " + force_formulation);
        }
    }

    template void to_json<f64_3>(nlohmann::json &j, const RiemannConfig<f64_3> &p);
    template void from_json<f64_3>(const nlohmann::json &j, RiemannConfig<f64_3> &p);
    template void to_json<f64_3>(nlohmann::json &j, const ReconstructConfig<f64_3> &p);
    template void from_json<f64_3>(const nlohmann::json &j, ReconstructConfig<f64_3> &p);
    template void to_json<f64_3>(nlohmann::json &j, const ForceFormulationConfig<f64_3> &p);
    template void from_json<f64_3>(const nlohmann::json &j, ForceFormulationConfig<f64_3> &p);

} // namespace shammodels::gsph
