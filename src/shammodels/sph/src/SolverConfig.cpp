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
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/type_name_info.hpp"
#include "shambackends/type_convert_json.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shamrock/io/json_std_optional.hpp"
#include "shamrock/io/json_utils.hpp"
#include "shamrock/io/units_json.hpp"
#include <nlohmann/json.hpp>

namespace shammodels::sph {

    template<class Tvec, template<class> class SPHKernel>
    void SolverConfig<Tvec, SPHKernel>::set_layout(shamrock::patch::PatchDataLayerLayout &pdl) {
        pdl.add_field<Tvec>("xyz", 1);
        pdl.add_field<Tvec>("vxyz", 1);
        pdl.add_field<Tvec>("axyz", 1);
        pdl.add_field<Tvec>("axyz_ext", 1);
        pdl.add_field<Tscal>("hpart", 1);

        if (track_particles_id) {
            pdl.add_field<u64>("part_id", 1);
        }

        if (has_field_uint()) {
            pdl.add_field<Tscal>("uint", 1);
            pdl.add_field<Tscal>("duint", 1);
        }

        if (has_field_alphaAV()) {
            pdl.add_field<Tscal>("alpha_AV", 1);
        }

        if (has_field_divv()) {
            pdl.add_field<Tscal>("divv", 1);
        }

        if (has_field_dtdivv()) {
            pdl.add_field<Tscal>("dtdivv", 1);
        }

        if (has_field_curlv()) {
            pdl.add_field<Tvec>("curlv", 1);
        }

        if (has_field_soundspeed()) {

            // this should not be needed idealy, but we need the pressure on the ghosts and
            // we don't want to communicate it as it can be recomputed from the other fields
            // hence we copy the soundspeed at the end of the step to a field in the patchdata
            pdl.add_field<Tscal>("soundspeed", 1);
        }

        if (has_field_B_on_rho()) {

            pdl.add_field<Tvec>("B/rho", 1);
            pdl.add_field<Tvec>("dB/rho", 1);
            pdl.add_field<Tscal>("drho/dt", 1);
        }

        if (has_field_psi_on_ch()) {
            pdl.add_field<Tscal>("psi/ch", 1);
            pdl.add_field<Tscal>("dpsi/ch", 1);
        }
        if (has_field_divB()) {
            pdl.add_field<Tscal>("divB", 1);
        }

        if (has_field_curlB()) {
            pdl.add_field<Tvec>("curlB", 1);
        }

        if (dust_config.has_epsilon_field()) {
            u32 ndust = dust_config.get_dust_nvar();
            pdl.add_field<Tscal>("epsilon", ndust);
            pdl.add_field<Tscal>("dtepsilon", ndust);
        }

        if (dust_config.has_deltav_field()) {
            u32 ndust = dust_config.get_dust_nvar();
            pdl.add_field<Tvec>("deltav", ndust);
            pdl.add_field<Tvec>("dtdeltav", ndust);
        }

        if (compute_luminosity) {
            pdl.add_field<Tscal>("luminosity", 1);
        }

        if (do_MHD_debug()) {
            pdl.add_field<Tvec>("gas_pressure", 1);
            pdl.add_field<Tvec>("mag_pressure", 1);
            pdl.add_field<Tvec>("mag_tension", 1);
            pdl.add_field<Tvec>("tensile_corr", 1);

            pdl.add_field<Tscal>("psi_propag", 1);
            pdl.add_field<Tscal>("psi_diff", 1);
            pdl.add_field<Tscal>("psi_cons", 1);
            pdl.add_field<Tscal>("u_mhd", 1);
        }

        if (should_save_dt_to_fields()) {
            pdl.add_field<Tscal>("dt_part", 1);
        }

        if (dust_config.has_s_j_field()) {
            u32 ndust = dust_config.get_dust_nvar();
            pdl.add_field<Tscal>("s_j", ndust);
            pdl.add_field<Tscal>("ds_j_dt", ndust);
            pdl.add_field<Tvec>("delta_v", ndust);
        }
    }

    template<class Tvec, template<class> class SPHKernel>
    void SolverConfig<Tvec, SPHKernel>::set_ghost_layout(
        shamrock::patch::PatchDataLayerLayout &ghost_layout) {

        ghost_layout.add_field<Tscal>("hpart", 1);
        ghost_layout.add_field<Tscal>("uint", 1);
        ghost_layout.add_field<Tvec>("vxyz", 1);

        if (has_axyz_in_ghost()) {
            ghost_layout.add_field<Tvec>("axyz", 1);
        }
        ghost_layout.add_field<Tscal>("omega", 1);

        if (ghost_has_soundspeed()) {
            ghost_layout.add_field<Tscal>("soundspeed", 1);
        }

        if (has_field_B_on_rho()) {
            ghost_layout.add_field<Tvec>("B/rho", 1);
        }

        if (has_field_psi_on_ch()) {
            ghost_layout.add_field<Tscal>("psi/ch", 1);
        }

        if (has_field_curlB()) {
            ghost_layout.add_field<Tvec>("curlB", 1);
        }

        if (dust_config.has_epsilon_field()) {
            u32 ndust = dust_config.get_dust_nvar();
            ghost_layout.add_field<Tscal>("epsilon", ndust);
        }

        if (dust_config.has_deltav_field()) {
            u32 ndust = dust_config.get_dust_nvar();
            ghost_layout.add_field<Tvec>("deltav", ndust);
        }

        if (dust_config.has_s_j_field()) {
            u32 ndust = dust_config.get_dust_nvar();
            ghost_layout.add_field<Tscal>("s_j", ndust);
        }
    }

    template<class Tvec, template<class> class SPHKernel>
    void SolverConfig<Tvec, SPHKernel>::print_status() {
        if (shamcomm::world_rank() != 0) {
            return;
        }
        logger::raw_ln("----- SPH Solver configuration -----");
        logger::raw_ln(nlohmann::json{*this}.dump(4));
        logger::raw_ln("------------------------------------");
    }

    template<class Tvec>
    void DustConfig<Tvec>::mode_to_json(nlohmann::json &j) const {
        if (const None *cfg = std::get_if<None>(&current_mode)) {
            j = {{"type", "none"}};
        } else if (const MonofluidTVA *cfg = std::get_if<MonofluidTVA>(&current_mode)) {
            j
                = {{"type", "monofluid_tva"},
                   {"ndust", cfg->ndust},
                   {"pure_diffusion_mode", cfg->pure_diffusion_mode},
                   {"C_1_fluid", cfg->C_1_fluid},
                   {"C_drift", cfg->C_drift},
                   {"cfl_density_threshold", cfg->cfl_density_threshold},
                   {"ensure_s_j_positivity", cfg->ensure_s_j_positivity},
                   {"smooth_s_positivity_limiter", cfg->smooth_s_positivity_limiter},
                   {"dust_corrected_av", cfg->dust_corrected_av}};
        } else if (const MonofluidComplete *cfg = std::get_if<MonofluidComplete>(&current_mode)) {
            j = {{"type", "monofluid_complete"}, {"ndust", cfg->ndust}};
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    void DustConfig<Tvec>::mode_from_json(const nlohmann::json &j) {
        const std::string type = j.at("type").get<std::string>();
        if (type == "none") {
            set_none();
        } else if (type == "monofluid_tva") {
            set_monofluid_tva(
                j.at("ndust").get<u32>(),
                j.at("pure_diffusion_mode").get<bool>(),
                j.at("C_1_fluid").get<Tscal>(),
                j.at("C_drift").get<Tscal>(),
                j.at("cfl_density_threshold").get<Tscal>(),
                j.at("ensure_s_j_positivity").get<bool>(),
                j.value("smooth_s_positivity_limiter", false),
                j.value("dust_corrected_av", false));
        } else if (type == "monofluid_complete") {
            set_monofluid_complete(j.at("ndust").get<u32>());
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    void DustConfig<Tvec>::drag_mode_to_json(nlohmann::json &j) const {
        if (std::holds_alternative<None>(dust_drag_mode)) {
            j = {{"type", "none"}};
        } else if (
            const ConstantStoppingTimes *cfg
            = std::get_if<ConstantStoppingTimes>(&dust_drag_mode)) {
            j = {{"type", "constant_stopping_times"}, {"stopping_times", cfg->stopping_times}};
        } else if (const EpsteinDrag *cfg = std::get_if<EpsteinDrag>(&dust_drag_mode)) {
            j
                = {{"type", "epstein_drag"},
                   {"gamma", cfg->gamma},
                   {"grains_sizes", cfg->grains_sizes},
                   {"grains_densities", cfg->grains_densities}};
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    void DustConfig<Tvec>::drag_mode_from_json(const nlohmann::json &j) {
        if (j.at("type").get<std::string>() == "none") {
            dust_drag_mode = None{};
        } else if (j.at("type").get<std::string>() == "constant_stopping_times") {
            dust_drag_mode
                = ConstantStoppingTimes{j.at("stopping_times").get<std::vector<Tscal>>()};
        } else if (j.at("type").get<std::string>() == "epstein_drag") {
            dust_drag_mode = EpsteinDrag{
                j.at("gamma").get<Tscal>(),
                j.at("grains_sizes").get<std::vector<Tscal>>(),
                j.at("grains_densities").get<std::vector<Tscal>>()};
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tscal>
    void to_json(nlohmann::json &j, const CFLConfig<Tscal> &p) {
        j = nlohmann::json{
            {"cfl_cour", p.cfl_cour},
            {"cfl_force", p.cfl_force},
            {"cfl_multiplier_stiffness", p.cfl_multiplier_stiffness},
            {"eta_sink", p.eta_sink}};
    }

    template<class Tscal>
    void from_json(const nlohmann::json &j, CFLConfig<Tscal> &p) {
        j.at("cfl_cour").get_to<Tscal>(p.cfl_cour);
        j.at("cfl_force").get_to<Tscal>(p.cfl_force);
        j.at("cfl_multiplier_stiffness").get_to<Tscal>(p.cfl_multiplier_stiffness);

        if (j.contains("eta_sink")) {
            j.at("eta_sink").get_to<Tscal>(p.eta_sink);
        } else {
            ON_RANK_0(shamlog_warn_ln(
                "SPHConfig", "eta_sink not found when deserializing, defaulting to", p.eta_sink));
        }
    }

    template<class Tvec>
    void to_json(nlohmann::json &j, const ParticleKillingConfig<Tvec> &p) {
        j = nlohmann::json::array();
        for (const auto &kill : p.kill_list) {
            if (std::holds_alternative<typename ParticleKillingConfig<Tvec>::Sphere>(kill)) {
                const auto &sphere = std::get<typename ParticleKillingConfig<Tvec>::Sphere>(kill);
                j.push_back(
                    {{"type", "sphere"}, {"center", sphere.center}, {"radius", sphere.radius}});
            }
        }
    }

    template<class Tvec>
    void from_json(const nlohmann::json &j, ParticleKillingConfig<Tvec> &p) {
        p.kill_list.clear();
        for (const auto &item : j) {
            std::string type = item.at("type").get<std::string>();
            if (type == "sphere") {
                typename ParticleKillingConfig<Tvec>::Sphere sphere;
                item.at("center").get_to(sphere.center);
                item.at("radius").get_to(sphere.radius);
                p.kill_list.push_back(sphere);
            }
        }
    }

    void to_json(nlohmann::json &j, const SmoothingLengthConfig &p) {
        if (const SmoothingLengthConfig::DensityBased *conf
            = std::get_if<SmoothingLengthConfig::DensityBased>(&p.config)) {
            j = {
                {"type", "density_based"},
            };

        } else if (
            const SmoothingLengthConfig::DensityBasedNeighLim *conf
            = std::get_if<SmoothingLengthConfig::DensityBasedNeighLim>(&p.config)) {

            j = {
                {"type", "density_based_neigh_lim"},
                {"max_neigh_count", conf->max_neigh_count},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    void from_json(const nlohmann::json &j, SmoothingLengthConfig &p) {
        if (j.at("type").get<std::string>() == "density_based") {
            p.config = SmoothingLengthConfig::DensityBased{};
        } else if (j.at("type").get<std::string>() == "density_based_neigh_lim") {
            p.config
                = SmoothingLengthConfig::DensityBasedNeighLim{j.at("max_neigh_count").get<u32>()};
        } else {
            shambase::throw_unimplemented();
        }
    }

    void to_json(nlohmann::json &j, const SelfGravConfig &p) {
        if (const SelfGravConfig::SFMM *conf = std::get_if<SelfGravConfig::SFMM>(&p.config)) {
            j = {
                {"type", "sfmm"},
                {"order", conf->order},
                {"opening_angle", conf->opening_angle},
                {"reduction_level", conf->reduction_level},
                {"leaf_lowering", conf->leaf_lowering},
            };
        } else if (const SelfGravConfig::FMM *conf = std::get_if<SelfGravConfig::FMM>(&p.config)) {
            j = {
                {"type", "fmm"},
                {"order", conf->order},
                {"opening_angle", conf->opening_angle},
                {"reduction_level", conf->reduction_level},
            };
        } else if (const SelfGravConfig::MM *conf = std::get_if<SelfGravConfig::MM>(&p.config)) {
            j = {
                {"type", "mm"},
                {"order", conf->order},
                {"opening_angle", conf->opening_angle},
                {"reduction_level", conf->reduction_level},
            };
        } else if (
            const SelfGravConfig::Direct *conf = std::get_if<SelfGravConfig::Direct>(&p.config)) {
            j = {
                {"type", "direct"},
                {"reference_mode", conf->reference_mode},
            };
        } else if (
            const SelfGravConfig::None *conf = std::get_if<SelfGravConfig::None>(&p.config)) {
            j = {
                {"type", "none"},
            };
        }

        if (const SelfGravConfig::SofteningPlummer *conf
            = std::get_if<SelfGravConfig::SofteningPlummer>(&p.softening_mode)) {
            j["softening_mode"]   = "plummer";
            j["softening_length"] = conf->epsilon;
        } else {
            shambase::throw_unimplemented();
        }
    }

    void from_json(const nlohmann::json &j, SelfGravConfig &p) {
        if (j.at("type").get<std::string>() == "sfmm") {
            p.config = SelfGravConfig::SFMM{
                .order           = j.at("order").get<u32>(),
                .opening_angle   = j.at("opening_angle").get<f64>(),
                .leaf_lowering   = j.at("leaf_lowering").get<bool>(),
                .reduction_level = j.at("reduction_level").get<u32>()};
        } else if (j.at("type").get<std::string>() == "fmm") {
            p.config = SelfGravConfig::FMM{
                .order           = j.at("order").get<u32>(),
                .opening_angle   = j.at("opening_angle").get<f64>(),
                .reduction_level = j.at("reduction_level").get<u32>()};
        } else if (j.at("type").get<std::string>() == "mm") {
            p.config = SelfGravConfig::MM{
                .order           = j.at("order").get<u32>(),
                .opening_angle   = j.at("opening_angle").get<f64>(),
                .reduction_level = j.at("reduction_level").get<u32>()};
        } else if (j.at("type").get<std::string>() == "direct") {
            p.config = SelfGravConfig::Direct{j.at("reference_mode").get<bool>()};
        } else if (j.at("type").get<std::string>() == "none") {
            p.config = SelfGravConfig::None{};
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid self gravity type: " + j.at("type").get<std::string>());
        }

        if (j.contains("softening_mode")) {
            std::string softening_mode = j.at("softening_mode").get<std::string>();
            if (softening_mode == "plummer") {
                p.softening_mode
                    = SelfGravConfig::SofteningPlummer{j.at("softening_length").get<f64>()};
            } else {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "Invalid softening mode: " + softening_mode);
            }
        }
    }

    template<class Tvec>
    void to_json(nlohmann::json &j, const DustConfig<Tvec> &p) {
        j = {};

        p.mode_to_json(j["mode"]);
        p.drag_mode_to_json(j["drag_mode"]);
        j["ballabio_ts_limiter"] = p.ballabio_ts_limiter;
    }

    template<class Tvec>
    void from_json(const nlohmann::json &j, DustConfig<Tvec> &p) {
        p.mode_from_json(j.at("mode"));
        p.drag_mode_from_json(j.at("drag_mode"));
        p.ballabio_ts_limiter = j.value("ballabio_ts_limiter", false);
    }

    template<class Tvec, template<class> class SPHKernel>
    void to_json(nlohmann::json &j, const SolverConfig<Tvec, SPHKernel> &p) {
        using T       = SolverConfig<Tvec, SPHKernel>;
        using Tkernel = typename T::Kernel;

        std::string kernel_id = shambase::get_type_name<Tkernel>();
        std::string type_id   = shambase::get_type_name<Tvec>();

        j = nlohmann::json{
            {"kernel_id", kernel_id},
            {"type_id", type_id},
            {"scheduler_config", p.scheduler_conf},
            {"gpart_mass", p.gpart_mass},
            {"cfl_config", p.cfl_config},
            {"unit_sys", p.unit_sys},
            {"show_cfl_detail", p.show_cfl_detail},
            {"mhd_config", p.mhd_config},
            {"dust_config", p.dust_config},
            {"self_grav_config", p.self_grav_config},
            {"tree_reduction_level", p.tree_reduction_level},
            {"use_two_stage_search", p.use_two_stage_search},
            {"show_neigh_stats", p.show_neigh_stats},
            {"combined_dtdiv_divcurlv_compute", p.combined_dtdiv_divcurlv_compute},
            {"htol_up_coarse_cycle", p.htol_up_coarse_cycle},
            {"htol_up_fine_cycle", p.htol_up_fine_cycle},
            {"epsilon_h", p.epsilon_h},
            {"smoothing_length_config", p.smoothing_length_config},
            {"h_iter_per_subcycles", p.h_iter_per_subcycles},
            {"h_max_subcycles_count", p.h_max_subcycles_count},
            {"enable_particle_reordering", p.enable_particle_reordering},
            {"particle_reordering_step_freq", p.particle_reordering_step_freq},
            {"save_dt_to_fields", p.save_dt_to_fields},
            {"show_ghost_zone_graph", p.show_ghost_zone_graph},
            {"eos_config", p.eos_config},
            {"artif_viscosity", p.artif_viscosity},
            {"boundary_config", p.boundary_config},
            {"ext_force_config", p.ext_force_config},
            {"do_debug_dump", p.do_debug_dump},
            {"debug_dump_filename", p.debug_dump_filename},
            {"particle_killing", p.particle_killing},
        };
    }

    template<class Tvec, template<class> class SPHKernel>
    void from_json(const nlohmann::json &j, SolverConfig<Tvec, SPHKernel> &p) {
        using T       = SolverConfig<Tvec, SPHKernel>;
        using Tkernel = typename T::Kernel;

        if (j.contains("kernel_id")) {

            std::string kernel_id = j.at("kernel_id").get<std::string>();

            if (kernel_id != shambase::get_type_name<Tkernel>()) {
                shambase::throw_with_loc<std::runtime_error>(
                    "Invalid type to deserialize, wanted " + shambase::get_type_name<Tvec>()
                    + " but got " + kernel_id);
            }
        }

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

        auto _get_to_if_contains_fallbacks = [&](const std::string &key,
                                                 auto &value,
                                                 std::initializer_list<const char *> fallbacks) {
            shamrock::get_to_if_contains_fallbacks(
                j, key, value, fallbacks, has_used_defaults, has_updated_config);
        };

        _get_to_if_contains("scheduler_config", p.scheduler_conf);
        _get_to_if_contains("gpart_mass", p.gpart_mass);
        _get_to_if_contains("cfl_config", p.cfl_config);
        _get_to_if_contains("unit_sys", p.unit_sys);
        _get_to_if_contains("show_cfl_detail", p.show_cfl_detail);
        _get_to_if_contains("mhd_config", p.mhd_config);
        _get_to_if_contains("dust_config", p.dust_config);
        _get_to_if_contains("self_grav_config", p.self_grav_config);
        _get_to_if_contains("tree_reduction_level", p.tree_reduction_level);
        _get_to_if_contains("use_two_stage_search", p.use_two_stage_search);
        _get_to_if_contains("show_neigh_stats", p.show_neigh_stats);
        _get_to_if_contains("combined_dtdiv_divcurlv_compute", p.combined_dtdiv_divcurlv_compute);
        _get_to_if_contains_fallbacks(
            "htol_up_coarse_cycle", p.htol_up_coarse_cycle, {"htol_up_tol"});
        _get_to_if_contains_fallbacks("htol_up_fine_cycle", p.htol_up_fine_cycle, {"htol_up_iter"});
        _get_to_if_contains("epsilon_h", p.epsilon_h);
        _get_to_if_contains("smoothing_length_config", p.smoothing_length_config);
        _get_to_if_contains("h_iter_per_subcycles", p.h_iter_per_subcycles);
        _get_to_if_contains("h_max_subcycles_count", p.h_max_subcycles_count);
        _get_to_if_contains("enable_particle_reordering", p.enable_particle_reordering);
        _get_to_if_contains("particle_reordering_step_freq", p.particle_reordering_step_freq);
        _get_to_if_contains("save_dt_to_fields", p.save_dt_to_fields);
        _get_to_if_contains("show_ghost_zone_graph", p.show_ghost_zone_graph);
        _get_to_if_contains("eos_config", p.eos_config);
        _get_to_if_contains("artif_viscosity", p.artif_viscosity);
        _get_to_if_contains("boundary_config", p.boundary_config);
        _get_to_if_contains("ext_force_config", p.ext_force_config);
        _get_to_if_contains("do_debug_dump", p.do_debug_dump);
        _get_to_if_contains("debug_dump_filename", p.debug_dump_filename);
        _get_to_if_contains("particle_killing", p.particle_killing);

        if (has_used_defaults || has_updated_config) {
            if (shamcomm::world_rank() == 0) {
                logger::info_ln(
                    "SPH::SolverConfig",
                    shamrock::log_json_changes(p, j, has_used_defaults, has_updated_config));
            }
        }
    }

} // namespace shammodels::sph

using namespace shammath;

template class shammodels::sph::SolverConfig<f64_3, M4>;
template class shammodels::sph::SolverConfig<f64_3, M6>;
template class shammodels::sph::SolverConfig<f64_3, M8>;

template class shammodels::sph::SolverConfig<f64_3, C2>;
template class shammodels::sph::SolverConfig<f64_3, C4>;
template class shammodels::sph::SolverConfig<f64_3, C6>;

template void shammodels::sph::to_json<f64>(
    nlohmann::json &j, const shammodels::sph::CFLConfig<f64> &p);
template void shammodels::sph::from_json<f64>(
    const nlohmann::json &j, shammodels::sph::CFLConfig<f64> &p);

template void shammodels::sph::to_json<f64_3>(
    nlohmann::json &j, const shammodels::sph::ParticleKillingConfig<f64_3> &p);
template void shammodels::sph::from_json<f64_3>(
    const nlohmann::json &j, shammodels::sph::ParticleKillingConfig<f64_3> &p);

template void shammodels::sph::to_json<f64_3>(
    nlohmann::json &j, const shammodels::sph::DustConfig<f64_3> &p);
template void shammodels::sph::from_json<f64_3>(
    const nlohmann::json &j, shammodels::sph::DustConfig<f64_3> &p);

#define SHAM_INSTANTIATE_SPH_SOLVERCONFIG_JSON(Kern)                                               \
    template void shammodels::sph::to_json<f64_3, Kern>(                                           \
        nlohmann::json & j, const shammodels::sph::SolverConfig<f64_3, Kern> &p);                  \
    template void shammodels::sph::from_json<f64_3, Kern>(                                         \
        const nlohmann::json &j, shammodels::sph::SolverConfig<f64_3, Kern> &p);

SHAM_INSTANTIATE_SPH_SOLVERCONFIG_JSON(M4)
SHAM_INSTANTIATE_SPH_SOLVERCONFIG_JSON(M6)
SHAM_INSTANTIATE_SPH_SOLVERCONFIG_JSON(M8)
SHAM_INSTANTIATE_SPH_SOLVERCONFIG_JSON(C2)
SHAM_INSTANTIATE_SPH_SOLVERCONFIG_JSON(C4)
SHAM_INSTANTIATE_SPH_SOLVERCONFIG_JSON(C6)

#undef SHAM_INSTANTIATE_SPH_SOLVERCONFIG_JSON
