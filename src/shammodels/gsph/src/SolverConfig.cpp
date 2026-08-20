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
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Implementation of GSPH solver configuration methods
 */

#include "shambase/type_name_info.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/gsph/config/FieldNames.hpp"
#include "shamrock/io/json_utils.hpp"
#include "shamrock/io/units_json.hpp"
#include <nlohmann/json.hpp>

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::SolverConfig<Tvec, SPHKernel>::set_layout(
    shamrock::patch::PatchDataLayerLayout &pdl) {

    // Position
    pdl.add_field<Tvec>(names::common::xyz, 1);

    // Velocity
    pdl.add_field<Tvec>(names::newtonian::vxyz, 1);

    // Acceleration
    pdl.add_field<Tvec>(names::newtonian::axyz, 1);

    // Smoothing length
    pdl.add_field<Tscal>(names::common::hpart, 1);

    // Internal energy (for adiabatic EOS)
    if (has_field_uint()) {
        pdl.add_field<Tscal>(names::newtonian::uint, 1);
        pdl.add_field<Tscal>(names::newtonian::duint, 1);
    }

    // Thermodynamic fields - stored in patchdata for persistence across restarts
    // These are computed during EOS step and copied to patchdata
    pdl.add_field<Tscal>(names::newtonian::density, 1);
    pdl.add_field<Tscal>(names::newtonian::pressure, 1);
    pdl.add_field<Tscal>(names::newtonian::soundspeed, 1);
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::gsph::SolverConfig<Tvec, SPHKernel>::set_ghost_layout(
    shamrock::patch::PatchDataLayerLayout &ghost_layout) {

    // Velocity (needed for Riemann solver)
    ghost_layout.add_field<Tvec>(names::newtonian::vxyz, 1);

    // Smoothing length
    ghost_layout.add_field<Tscal>(names::common::hpart, 1);

    // Omega (grad-h correction)
    ghost_layout.add_field<Tscal>(names::newtonian::omega, 1);

    // Density (computed via SPH summation)
    ghost_layout.add_field<Tscal>(names::newtonian::density, 1);

    // Internal energy (for adiabatic EOS)
    if (has_field_uint()) {
        ghost_layout.add_field<Tscal>(names::newtonian::uint, 1);
    }
}

namespace shammodels::gsph {

    template<class Tscal>
    void to_json(nlohmann::json &j, const CFLConfig<Tscal> &p) {
        j = nlohmann::json{
            {"cfl_cour", p.cfl_cour},
            {"cfl_force", p.cfl_force},
        };
    }

    template<class Tscal>
    void from_json(const nlohmann::json &j, CFLConfig<Tscal> &p) {
        j.at("cfl_cour").get_to(p.cfl_cour);
        j.at("cfl_force").get_to(p.cfl_force);
    }

    template<class Tvec, template<class> class SPHKernel>
    void to_json(nlohmann::json &j, const SolverConfig<Tvec, SPHKernel> &p) {
        using T       = SolverConfig<Tvec, SPHKernel>;
        using Tkernel = typename T::Kernel;

        std::string kernel_id = shambase::get_type_name<Tkernel>();
        std::string type_id   = shambase::get_type_name<Tvec>();

        j = nlohmann::json{
            {"solver_type", "gsph"},
            {"kernel_id", kernel_id},
            {"type_id", type_id},
            {"scheduler_config", p.scheduler_conf},
            {"gpart_mass", p.gpart_mass},
            {"cfl_config", p.cfl_config},
            {"unit_sys", p.unit_sys},
            {"riemann_config", p.riemann_config},
            {"reconstruct_config", p.reconstruct_config},
            {"force_formulation_config", p.force_formulation_config},
            {"eos_config", p.eos_config},
            {"boundary_config", p.boundary_config},
            {"tree_reduction_level", p.tree_reduction_level},
            {"use_two_stage_search", p.use_two_stage_search},
            {"htol_up_coarse_cycle", p.htol_up_coarse_cycle},
            {"htol_up_fine_cycle", p.htol_up_fine_cycle},
            {"epsilon_h", p.epsilon_h},
            {"h_iter_per_subcycles", p.h_iter_per_subcycles},
            {"h_max_subcycles_count", p.h_max_subcycles_count},
        };
    }

    template<class Tvec, template<class> class SPHKernel>
    void from_json(const nlohmann::json &j, SolverConfig<Tvec, SPHKernel> &p) {
        using T       = SolverConfig<Tvec, SPHKernel>;
        using Tkernel = typename T::Kernel;

        std::string kernel_id = j.at("kernel_id").get<std::string>();
        if (kernel_id != shambase::get_type_name<Tkernel>()) {
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid kernel type: expected " + shambase::get_type_name<Tkernel>() + " but got "
                + kernel_id);
        }

        std::string type_id = j.at("type_id").get<std::string>();
        if (type_id != shambase::get_type_name<Tvec>()) {
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid vector type: expected " + shambase::get_type_name<Tvec>() + " but got "
                + type_id);
        }

        bool has_used_defaults  = false;
        bool has_updated_config = false;

        auto _get_to_if_contains = [&](const std::string &key, auto &value) {
            shamrock::get_to_if_contains(j, key, value, has_used_defaults);
        };

        _get_to_if_contains("scheduler_config", p.scheduler_conf);
        _get_to_if_contains("gpart_mass", p.gpart_mass);
        _get_to_if_contains("cfl_config", p.cfl_config);
        _get_to_if_contains("unit_sys", p.unit_sys);
        _get_to_if_contains("riemann_config", p.riemann_config);
        _get_to_if_contains("reconstruct_config", p.reconstruct_config);
        _get_to_if_contains("force_formulation_config", p.force_formulation_config);
        _get_to_if_contains("eos_config", p.eos_config);
        _get_to_if_contains("boundary_config", p.boundary_config);
        _get_to_if_contains("tree_reduction_level", p.tree_reduction_level);
        _get_to_if_contains("use_two_stage_search", p.use_two_stage_search);
        _get_to_if_contains("htol_up_coarse_cycle", p.htol_up_coarse_cycle);
        _get_to_if_contains("htol_up_fine_cycle", p.htol_up_fine_cycle);
        _get_to_if_contains("epsilon_h", p.epsilon_h);
        _get_to_if_contains("h_iter_per_subcycles", p.h_iter_per_subcycles);
        _get_to_if_contains("h_max_subcycles_count", p.h_max_subcycles_count);

        if (has_used_defaults || has_updated_config) {
            if (shamcomm::world_rank() == 0) {
                logger::info_ln(
                    "GSPH::SolverConfig",
                    shamrock::log_json_changes(p, j, has_used_defaults, has_updated_config));
            }
        }
    }

} // namespace shammodels::gsph

// Explicit template instantiations
using namespace shammath;
template class shammodels::gsph::SolverConfig<f64_3, M4>;
template class shammodels::gsph::SolverConfig<f64_3, M6>;
template class shammodels::gsph::SolverConfig<f64_3, M8>;
template class shammodels::gsph::SolverConfig<f64_3, C2>;
template class shammodels::gsph::SolverConfig<f64_3, C4>;
template class shammodels::gsph::SolverConfig<f64_3, C6>;

template void shammodels::gsph::to_json<f64>(
    nlohmann::json &j, const shammodels::gsph::CFLConfig<f64> &p);
template void shammodels::gsph::from_json<f64>(
    const nlohmann::json &j, shammodels::gsph::CFLConfig<f64> &p);

#define SHAM_INSTANTIATE_GSPH_SOLVERCONFIG_JSON(Kern)                                              \
    template void shammodels::gsph::to_json<f64_3, Kern>(                                          \
        nlohmann::json & j, const shammodels::gsph::SolverConfig<f64_3, Kern> &p);                 \
    template void shammodels::gsph::from_json<f64_3, Kern>(                                        \
        const nlohmann::json &j, shammodels::gsph::SolverConfig<f64_3, Kern> &p);

SHAM_INSTANTIATE_GSPH_SOLVERCONFIG_JSON(M4)
SHAM_INSTANTIATE_GSPH_SOLVERCONFIG_JSON(M6)
SHAM_INSTANTIATE_GSPH_SOLVERCONFIG_JSON(M8)
SHAM_INSTANTIATE_GSPH_SOLVERCONFIG_JSON(C2)
SHAM_INSTANTIATE_GSPH_SOLVERCONFIG_JSON(C4)
SHAM_INSTANTIATE_GSPH_SOLVERCONFIG_JSON(C6)

#undef SHAM_INSTANTIATE_GSPH_SOLVERCONFIG_JSON
