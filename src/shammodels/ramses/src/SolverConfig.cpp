// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SolverConfig.cpp
 * @author Anass Serhani (anass.serhani@cnrs.fr) --no git blame--
 * @author Benoit Commercon (benoit.commercon@ens-lyon.fr) --no git blame--
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Noé Brucy (noe.brucy@ens-lyon.fr) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief
 *
 */

#include "shambase/type_name_info.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/config/enum_DragSolverMode.hpp"
#include "shammodels/ramses/config/enum_DustRiemannSolverMode.hpp"
#include "shammodels/ramses/config/enum_GravityMode.hpp"
#include "shammodels/ramses/config/enum_RiemannSolverMode.hpp"
#include "shammodels/ramses/config/enum_SlopeMode.hpp"
#include "shamrock/io/json_print_diff.hpp"
#include "shamrock/io/json_std_optional.hpp"
#include "shamrock/io/json_utils.hpp"
#include "shamrock/io/units_json.hpp"
#include <nlohmann/json.hpp>

namespace shammodels::basegodunov {

    void to_json(nlohmann::json &j, const SlopeMode &e) {
        switch (e) {
        case SlopeMode::None       : j = "none"; break;
        case SlopeMode::VanLeer_f  : j = "vanleer_f"; break;
        case SlopeMode::VanLeer_std: j = "vanleer_std"; break;
        case SlopeMode::VanLeer_sym: j = "vanleer_sym"; break;
        case SlopeMode::Minmod     : j = "minmod"; break;
        default:
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid SlopeMode value: " + std::to_string(e));
        }
    }

    void from_json(const nlohmann::json &j, SlopeMode &e) {
        const std::string s = j.get<std::string>();
        if (s == "none") {
            e = SlopeMode::None;
        } else if (s == "vanleer_f") {
            e = SlopeMode::VanLeer_f;
        } else if (s == "vanleer_std") {
            e = SlopeMode::VanLeer_std;
        } else if (s == "vanleer_sym") {
            e = SlopeMode::VanLeer_sym;
        } else if (s == "minmod") {
            e = SlopeMode::Minmod;
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid SlopeMode value: " + s);
        }
    }

    void to_json(nlohmann::json &j, const RiemannSolverMode &e) {
        switch (e) {
        case RiemannSolverMode::Rusanov: j = "rusanov"; break;
        case RiemannSolverMode::HLL    : j = "hll"; break;
        case RiemannSolverMode::HLLC   : j = "hllc"; break;
        default:
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid RiemannSolverMode value: " + std::to_string(e));
        }
    }

    void from_json(const nlohmann::json &j, RiemannSolverMode &e) {
        const std::string s = j.get<std::string>();
        if (s == "rusanov") {
            e = RiemannSolverMode::Rusanov;
        } else if (s == "hll") {
            e = RiemannSolverMode::HLL;
        } else if (s == "hllc") {
            e = RiemannSolverMode::HLLC;
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid RiemannSolverMode value: " + s);
        }
    }

    void to_json(nlohmann::json &j, const GravityMode &e) {
        switch (e) {
        case GravityMode::NoGravity: j = "no_gravity"; break;
        case GravityMode::CG       : j = "cg"; break;
        case GravityMode::PCG      : j = "pcg"; break;
        case GravityMode::BICGSTAB : j = "bicgstab"; break;
        case GravityMode::MULTIGRID: j = "multigrid"; break;
        default:
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid GravityMode value: " + std::to_string(e));
        }
    }

    void from_json(const nlohmann::json &j, GravityMode &e) {
        const std::string s = j.get<std::string>();
        if (s == "no_gravity") {
            e = GravityMode::NoGravity;
        } else if (s == "cg") {
            e = GravityMode::CG;
        } else if (s == "pcg") {
            e = GravityMode::PCG;
        } else if (s == "bicgstab") {
            e = GravityMode::BICGSTAB;
        } else if (s == "multigrid") {
            e = GravityMode::MULTIGRID;
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid GravityMode value: " + s);
        }
    }

    void to_json(nlohmann::json &j, const DustRiemannSolverMode &e) {
        switch (e) {
        case DustRiemannSolverMode::NoDust: j = "no_dust"; break;
        case DustRiemannSolverMode::DHLL  : j = "dhll"; break;
        case DustRiemannSolverMode::HB    : j = "hb"; break;
        default:
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid DustRiemannSolverMode value: " + std::to_string(e));
        }
    }

    void from_json(const nlohmann::json &j, DustRiemannSolverMode &e) {
        const std::string s = j.get<std::string>();
        if (s == "no_dust") {
            e = DustRiemannSolverMode::NoDust;
        } else if (s == "dhll") {
            e = DustRiemannSolverMode::DHLL;
        } else if (s == "hb") {
            e = DustRiemannSolverMode::HB;
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid DustRiemannSolverMode value: " + s);
        }
    }

    void to_json(nlohmann::json &j, const DragSolverMode &e) {
        switch (e) {
        case DragSolverMode::NoDrag: j = "no_drag"; break;
        case DragSolverMode::IRK1  : j = "irk1"; break;
        case DragSolverMode::IRK2  : j = "irk2"; break;
        case DragSolverMode::EXPO  : j = "expo"; break;
        default:
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid DragSolverMode value: " + std::to_string(e));
        }
    }

    void from_json(const nlohmann::json &j, DragSolverMode &e) {
        const std::string s = j.get<std::string>();
        if (s == "no_drag") {
            e = DragSolverMode::NoDrag;
        } else if (s == "irk1") {
            e = DragSolverMode::IRK1;
        } else if (s == "irk2") {
            e = DragSolverMode::IRK2;
        } else if (s == "expo") {
            e = DragSolverMode::EXPO;
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid DragSolverMode value: " + s);
        }
    }

    void to_json(nlohmann::json &j, const BCConfig::GhostType &e) {
        switch (e) {
        case BCConfig::GhostType::Periodic  : j = "periodic"; break;
        case BCConfig::GhostType::Reflective: j = "reflective"; break;
        case BCConfig::GhostType::Outflow   : j = "outflow"; break;
        default:
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid BCConfig::GhostType value: " + std::to_string(static_cast<int>(e)));
        }
    }

    void from_json(const nlohmann::json &j, BCConfig::GhostType &e) {
        const std::string type = j.get<std::string>();
        if (type == "periodic") {
            e = BCConfig::GhostType::Periodic;
        } else if (type == "reflective") {
            e = BCConfig::GhostType::Reflective;
        } else if (type == "outflow") {
            e = BCConfig::GhostType::Outflow;
        } else {
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid BCConfig::GhostType value: " + type);
        }
    }

    void to_json(nlohmann::json &j, const BCConfig &p) {
        j = nlohmann::json{
            {"ghost_type_x", p.ghost_type_x},
            {"ghost_type_y", p.ghost_type_y},
            {"ghost_type_z", p.ghost_type_z}};
    }

    void from_json(const nlohmann::json &j, BCConfig &p) {
        j.at("ghost_type_x").get_to(p.ghost_type_x);
        j.at("ghost_type_y").get_to(p.ghost_type_y);
        j.at("ghost_type_z").get_to(p.ghost_type_z);
    }

    void to_json(nlohmann::json &j, const DragConfig &p) {
        j = nlohmann::json{
            {"drag_solver", p.drag_solver_config},
            {"alphas", p.alphas},
            {"enable_frictional_heating", p.enable_frictional_heating}};
    }

    void from_json(const nlohmann::json &j, DragConfig &p) {
        j.at("drag_solver").get_to(p.drag_solver_config);
        j.at("alphas").get_to(p.alphas);
        j.at("enable_frictional_heating").get_to(p.enable_frictional_heating);
    }

    template<class Tvec, class TgridVec>
    void amr_config_to_json(nlohmann::json &j, const AMRMode<Tvec, TgridVec> &p) {
        using AMR = AMRMode<Tvec, TgridVec>;

        if (std::holds_alternative<typename AMR::None>(p.config)) {
            j = {{"type", "none"}};
        } else if (const auto *cfg = std::get_if<typename AMR::DensityBased>(&p.config)) {
            j = {{"type", "density_based"}, {"crit_mass", cfg->crit_mass}};
        } else if (const auto *cfg = std::get_if<typename AMR::PseudoGradientBased>(&p.config)) {
            j
                = {{"type", "pseudo_gradient_based"},
                   {"error_min", cfg->error_min},
                   {"error_max", cfg->error_max}};
        } else if (const auto *cfg = std::get_if<typename AMR::JeansLengthBased>(&p.config)) {
            j = {{"type", "jeans_length_based"}, {"N_J", cfg->N_J}, {"T_0", cfg->T_0}};
        } else if (const auto *cfg = std::get_if<typename AMR::ShearBased>(&p.config)) {
            j = {{"type", "shear_based"}, {"threshold", cfg->threshold}};
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec, class TgridVec>
    void amr_config_from_json(const nlohmann::json &j, AMRMode<Tvec, TgridVec> &p) {
        using Tscal = shambase::VecComponent<Tvec>;

        const std::string type = j.at("type").get<std::string>();
        if (type == "none") {
            p.set_refine_none();
        } else if (type == "density_based") {
            p.set_refine_density_based(j.at("crit_mass").get<Tscal>());
        } else if (type == "pseudo_gradient_based") {
            p.set_refine_pseudo_gradient_based(
                j.at("error_min").get<Tscal>(), j.at("error_max").get<Tscal>());
        } else if (type == "jeans_length_based") {
            p.set_refine_jeans_length_based(j.at("N_J").get<u32>(), j.at("T_0").get<Tscal>());
        } else if (type == "shear_based") {
            p.set_refine_shear_based(j.at("threshold").get<Tscal>());
        } else {
            shambase::throw_with_loc<std::runtime_error>("Invalid AMR mode type: " + type);
        }
    }

    template<class Tvec, class TgridVec>
    void to_json(nlohmann::json &j, const AMRMode<Tvec, TgridVec> &p) {
        nlohmann::json config_j;
        amr_config_to_json(config_j, p);
        j = nlohmann::json{{"old_amr", p.old_amr}, {"config", config_j}};
    }

    template<class Tvec, class TgridVec>
    void from_json(const nlohmann::json &j, AMRMode<Tvec, TgridVec> &p) {
        j.at("old_amr").get_to(p.old_amr);
        amr_config_from_json(j.at("config"), p);
    }

    template<class Tvec, class TgridVec>
    void SolverConfig<Tvec, TgridVec>::set_layout(shamrock::patch::PatchDataLayerLayout &pdl) {
        pdl.add_field<TgridVec>("cell_min", 1);
        pdl.add_field<TgridVec>("cell_max", 1);
        pdl.add_field<Tscal>("rho", AMRBlock::block_size);
        pdl.add_field<Tvec>("rhovel", AMRBlock::block_size);
        pdl.add_field<Tscal>("rhoetot", AMRBlock::block_size);

        if (is_dust_on()) {
            u32 ndust = dust_config.ndust;
            pdl.add_field<Tscal>("rho_dust", (ndust * AMRBlock::block_size));
            pdl.add_field<Tvec>("rhovel_dust", (ndust * AMRBlock::block_size));
        }

        if (is_gravity_on()) {
            pdl.add_field<Tscal>("phi", AMRBlock::block_size);
        }

        if (is_gas_passive_scalar_on()) {
            u32 npscal_gas = npscal_gas_config.npscal_gas;
            pdl.add_field<Tscal>("rho_gas_pscal", (npscal_gas * AMRBlock::block_size));
        }
    }

    template<class Tvec, class TgridVec>
    void to_json(nlohmann::json &j, const SolverConfig<Tvec, TgridVec> &p) {

        j = nlohmann::json{
            {"type_id", shambase::get_type_name<Tvec>()},
            {"scheduler_config", p.scheduler_conf},
            {"courant_safety_factor", p.Csafe},
            {"dust_riemann_solver", p.dust_config.dust_riemann_config},
            {"ndust", p.dust_config.ndust},
            {"eos_gamma", p.eos_gamma},
            {"face_half_time_interpolation", p.face_half_time_interpolation},
            {"gravity_solver", p.gravity_config.gravity_mode},
            {"analytical_gravity", p.gravity_config.analytical_gravity},
            {"gravity_tol", p.gravity_config.tol},
            {"grid_coord_to_pos_fact", p.grid_coord_to_pos_fact},
            {"hydro_riemann_solver", p.riemann_config},
            {"passive_scalar_mode", p.npscal_gas_config.npscal_gas},
            {"slope_limiter", p.slope_config},
            {"unit_sys", p.unit_sys},
            {"drag_config", p.drag_config},
            {"bc_config", p.bc_config},
            {"amr_mode", p.amr_mode}};
    }

    template<class Tvec, class TgridVec>
    void from_json(const nlohmann::json &j, SolverConfig<Tvec, TgridVec> &p) {
        using T = SolverConfig<Tvec, TgridVec>;

        if (j.contains("type_id")) {

            std::string type_id = j.at("type_id").get<std::string>();

            if (type_id != shambase::get_type_name<Tvec>()) {
                shambase::throw_with_loc<std::runtime_error>(
                    "Invalid type to deserialize, wanted " + shambase::get_type_name<Tvec>()
                    + " but got " + type_id);
            }
        }

        bool has_used_defaults  = false;
        bool has_updated_config = false;

        auto _get_to_if_contains = [&](const std::string &key, auto &value) {
            shamrock::get_to_if_contains(j, key, value, has_used_defaults);
        };

        _get_to_if_contains("scheduler_config", p.scheduler_conf);

        // actual data stored in the json
        _get_to_if_contains("courant_safety_factor", p.Csafe);
        _get_to_if_contains("dust_riemann_solver", p.dust_config.dust_riemann_config);
        _get_to_if_contains("ndust", p.dust_config.ndust);
        _get_to_if_contains("eos_gamma", p.eos_gamma);
        _get_to_if_contains("face_half_time_interpolation", p.face_half_time_interpolation);
        _get_to_if_contains("gravity_solver", p.gravity_config.gravity_mode);
        _get_to_if_contains("analytical_gravity", p.gravity_config.analytical_gravity);
        _get_to_if_contains("gravity_tol", p.gravity_config.tol);
        _get_to_if_contains("grid_coord_to_pos_fact", p.grid_coord_to_pos_fact);
        _get_to_if_contains("hydro_riemann_solver", p.riemann_config);
        _get_to_if_contains("passive_scalar_mode", p.npscal_gas_config.npscal_gas);
        _get_to_if_contains("slope_limiter", p.slope_config);
        _get_to_if_contains("unit_sys", p.unit_sys);
        _get_to_if_contains("drag_config", p.drag_config);
        _get_to_if_contains("bc_config", p.bc_config);
        _get_to_if_contains("amr_mode", p.amr_mode);

        if (has_used_defaults || has_updated_config) {
            if (shamcomm::world_rank() == 0) {
                logger::info_ln(
                    "Ramses::SolverConfig",
                    shamrock::log_json_changes(p, j, has_used_defaults, has_updated_config));
            }
        }
    }

    template void to_json<f64_3, i64_3>(nlohmann::json &j, const AMRMode<f64_3, i64_3> &p);
    template void from_json<f64_3, i64_3>(const nlohmann::json &j, AMRMode<f64_3, i64_3> &p);
    template void to_json<f64_3, i64_3>(nlohmann::json &j, const SolverConfig<f64_3, i64_3> &p);
    template void from_json<f64_3, i64_3>(const nlohmann::json &j, SolverConfig<f64_3, i64_3> &p);

} // namespace shammodels::basegodunov

template class shammodels::basegodunov::SolverConfig<f64_3, i64_3>;
