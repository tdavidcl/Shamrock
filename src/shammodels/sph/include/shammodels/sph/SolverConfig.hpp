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
 * @file SolverConfig.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "config/AVConfig.hpp"
#include "config/BCConfig.hpp"
#include "nlohmann/json_fwd.hpp"
#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/type_traits.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/common/EOSConfig.hpp"
#include "shammodels/common/ExtForceConfig.hpp"
#include "shammodels/sph/config/MHDConfig.hpp"
#include "shamrock/experimental_features.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/RadixTree.hpp"
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
#include <stdexcept>
#include <variant>
#include <vector>

namespace shammodels::sph {

    /**
     * @brief The configuration for a sph solver
     *
     * @tparam Tvec the type of the vector used to represent the particles
     * @tparam SPHKernel the type of the SPH kernel
     */
    template<class Tvec, template<class> class SPHKernel>
    struct SolverConfig;

    /**
     * @brief The configuration for the CFL condition
     *
     * @tparam Tscal the type of the scalar used to represent the quantities
     */
    template<class Tscal>
    struct CFLConfig {

        /**
         * @brief The CFL condition for the courant factor
         */
        Tscal cfl_cour;

        /**
         * @brief The CFL condition for the force
         */
        Tscal cfl_force;

        /**
         * @brief The CFL multiplier stiffness
         */
        Tscal cfl_multiplier_stiffness = 2;

        /// eta sink to control the sink integrator
        Tscal eta_sink = 0.05;
    };

    template<class Tvec>
    struct ParticleKillingConfig {
        using Tscal = shambase::VecComponent<Tvec>;
        struct Sphere {
            Tvec center;
            Tscal radius;
        };

        using kill_t = std::variant<Sphere>;

        std::vector<kill_t> kill_list;

        inline void add_kill_sphere(const Tvec &center, Tscal radius) {
            kill_list.push_back(Sphere{center, radius});
        }
    };

    template<class Tscal>
    struct DustConfig {

        struct None {};

        struct MonofluidTVA {
            u32 ndust;
            bool pure_diffusion_mode = false;

            Tscal C_1_fluid             = 0.1;
            Tscal C_drift               = 1.0;
            Tscal cfl_density_threshold = shambase::get_epsilon<Tscal>();

            bool ensure_s_j_positivity = true;

            bool smooth_s_positivity_limiter = false;

            // use the corrected q_AV from Hutchison 2018 & Price Laibe 15
            bool dust_corrected_av = false;
        };

        struct MonofluidComplete {
            u32 ndust;
        };

        /// Variant type to store the EOS configuration
        using Variant = std::variant<None, MonofluidTVA, MonofluidComplete>;

        Variant current_mode = None{};

        inline void set_none() { current_mode = None{}; }
        inline void set_monofluid_tva(
            u32 nvar,
            bool pure_diffusion_mode         = false,
            Tscal C_1_fluid                  = 0.1,
            Tscal C_drift                    = 1.0,
            Tscal cfl_density_threshold      = shambase::get_epsilon<Tscal>(),
            bool ensure_s_j_positivity       = true,
            bool smooth_s_positivity_limiter = false,
            bool dust_corrected_av           = false) {
            current_mode = MonofluidTVA{
                nvar,
                pure_diffusion_mode,
                C_1_fluid,
                C_drift,
                cfl_density_threshold,
                ensure_s_j_positivity,
                smooth_s_positivity_limiter,
                dust_corrected_av};
        }
        inline void set_monofluid_complete(u32 nvar) { current_mode = MonofluidComplete{nvar}; }

        inline bool is_none() { return std::holds_alternative<None>(current_mode); }
        inline bool is_monofluid_tva() { return bool(std::get_if<MonofluidTVA>(&current_mode)); }
        inline bool is_monofluid_complete() {
            return bool(std::get_if<MonofluidComplete>(&current_mode));
        }

        inline MonofluidTVA &get_monofluid_tva() {
            return shambase::get_check_ref(std::get_if<MonofluidTVA>(&current_mode));
        }

        void mode_to_json(nlohmann::json &j) const;

        void mode_from_json(const nlohmann::json &j);

        inline bool has_s_j_field() {
            return is_monofluid_tva(); // S_j = sqrt(\rho \epsilon_j)
        }

        inline bool should_use_dust_av() {
            if (!is_monofluid_tva()) {
                return false;
            }
            return get_monofluid_tva().dust_corrected_av;
        }

        inline bool has_epsilon_field() {
            return bool(std::get_if<MonofluidComplete>(&current_mode));
        }

        inline bool has_deltav_field() {
            return bool(std::get_if<MonofluidComplete>(&current_mode));
        }

        inline u32 get_dust_nvar() {
            if (None *cfg = std::get_if<None>(&current_mode)) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "Querying a dust nvar with no dust as config is ... discutable ...");
                return 0;
            } else if (MonofluidTVA *cfg = std::get_if<MonofluidTVA>(&current_mode)) {
                return cfg->ndust;
            } else if (MonofluidComplete *cfg = std::get_if<MonofluidComplete>(&current_mode)) {
                return cfg->ndust;
            } else {
                shambase::throw_unimplemented("How did you get here ???");
            }
            return 0;
        }

        struct ConstantStoppingTimes {
            std::vector<Tscal> stopping_times;
        };

        struct EpsteinDrag {
            static constexpr bool supersonic_correction = false;
            Tscal gamma;
            std::vector<Tscal> grains_sizes;
            std::vector<Tscal> grains_densities;
        };

        std::variant<None, ConstantStoppingTimes, EpsteinDrag> dust_drag_mode = None{};

        bool ballabio_ts_limiter = false;

        void drag_mode_to_json(nlohmann::json &j) const;

        void drag_mode_from_json(const nlohmann::json &j);

        inline void set_drag_constant(ConstantStoppingTimes in) { dust_drag_mode = std::move(in); }

        inline void set_drag_epstein(EpsteinDrag in) { dust_drag_mode = std::move(in); }

        inline void check_config() {
            bool is_not_none = !is_none();
            if (is_not_none) {

                if (!shamrock::are_experimental_features_allowed()) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Dust config != None is experimental");
                } else {
                    ON_RANK_0(
                        logger::warn_ln(
                            "SPH::config",
                            "Dust config != None is work in progress, use it at your own risk"));
                }

                if (std::holds_alternative<None>(dust_drag_mode)) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "you must select a drag mode for the dust if the dust is on !");
                } else if (
                    ConstantStoppingTimes *cfg
                    = std::get_if<ConstantStoppingTimes>(&dust_drag_mode)) {
                    if (get_dust_nvar() != cfg->stopping_times.size()) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "stopping_times size does not match the number of dust bins");
                    }
                } else if (EpsteinDrag *cfg = std::get_if<EpsteinDrag>(&dust_drag_mode)) {
                    if (get_dust_nvar() != cfg->grains_densities.size()) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "grains_densities size does not match the number of dust bins");
                    }

                    if (get_dust_nvar() != cfg->grains_sizes.size()) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "grains_sizes size does not match the number of dust bins");
                    }
                }
            }
        }
    };

    struct SmoothingLengthConfig {
        struct DensityBased {};
        struct DensityBasedNeighLim {
            u32 max_neigh_count = 500;
        };

        using mode = std::variant<DensityBased, DensityBasedNeighLim>;

        mode config = DensityBased{};

        void set_density_based() { config = DensityBased{}; }
        void set_density_based_neigh_lim(u32 max_neigh_count) {
            config = DensityBasedNeighLim{max_neigh_count};
        }

        bool is_density_based_neigh_lim() const {
            return std::holds_alternative<DensityBasedNeighLim>(config);
        }
    };

    struct SelfGravConfig {

        struct SFMM {
            u32 order;
            f64 opening_angle;
            bool leaf_lowering;
            u32 reduction_level;
        };

        struct FMM {
            u32 order;
            f64 opening_angle;
            u32 reduction_level;
        };

        struct MM {
            u32 order;
            f64 opening_angle;
            u32 reduction_level;
        };

        struct Direct {
            bool reference_mode = false;
        };

        struct None {};

        using mode = std::variant<SFMM, FMM, MM, Direct, None>;

        mode config = None{};

        void set_none() { config = None{}; }
        void set_direct(bool reference_mode = false) { config = Direct{reference_mode}; }
        void set_mm(u32 mm_order, f64 opening_angle, u32 reduction_level) {
            config = MM{
                .order           = mm_order,
                .opening_angle   = opening_angle,
                .reduction_level = reduction_level};
        }
        void set_fmm(u32 order, f64 opening_angle, u32 reduction_level) {
            config = FMM{
                .order = order, .opening_angle = opening_angle, .reduction_level = reduction_level};
        }
        void set_sfmm(u32 order, f64 opening_angle, bool leaf_lowering, u32 reduction_level) {
            config = SFMM{
                .order           = order,
                .opening_angle   = opening_angle,
                .leaf_lowering   = leaf_lowering,
                .reduction_level = reduction_level};
        }

        bool is_none() const { return std::holds_alternative<None>(config); }
        bool is_direct() const { return std::holds_alternative<Direct>(config); }
        bool is_mm() const { return std::holds_alternative<MM>(config); }
        bool is_fmm() const { return std::holds_alternative<FMM>(config); }
        bool is_sfmm() const { return std::holds_alternative<SFMM>(config); }

        bool is_sg_on() const { return !is_none(); }
        bool is_sg_off() const { return is_none(); }

        struct SofteningPlummer {
            f64 epsilon;
        };

        using mode_soft          = std::variant<SofteningPlummer>;
        mode_soft softening_mode = SofteningPlummer{1e-9};

        void set_softening_plummer(f64 epsilon) { softening_mode = SofteningPlummer{epsilon}; }
        void set_softening_none() { set_softening_plummer(0.); }

        bool is_softening_plummer() const {
            return std::holds_alternative<SofteningPlummer>(softening_mode);
        }
    };

} // namespace shammodels::sph

template<class Tvec, template<class> class SPHKernel>
struct shammodels::sph::SolverConfig {

    /// The type of the scalar used to represent the quantities
    using Tscal = shambase::VecComponent<Tvec>;
    /// The dimension of the problem
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
    /// The type of the kernel used for the SPH interactions
    using Kernel = SPHKernel<Tscal>;
    /// The type of the Morton code for the tree
    using u_morton = u32;

    using RTree = shamtree::CompressedLeafBVH<u_morton, Tvec, 3>;

    /// The radius of the sph kernel
    static constexpr Tscal Rkern = Kernel::Rkern;

    Tscal gpart_mass; ///< The mass of each gas particle

    bool track_particles_id = false;

    inline void set_particle_tracking(bool state) { track_particles_id = state; }

    PatchSchedulerConfig scheduler_conf = {};

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Units Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// The unit system of the simulation
    std::optional<shamunits::UnitSystem<Tscal>> unit_sys = {};

    /// Set the unit system of the simulation
    inline void set_units(shamunits::UnitSystem<Tscal> new_sys) { unit_sys = new_sys; }

    /// Retrieves the value of the constant G based on the unit system.
    inline Tscal get_constant_G() {
        if (!unit_sys) {
            ON_RANK_0(logger::warn_ln("sph::Config", "the unit system is not set"));
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.G();
        } else {
            return shamunits::Constants<Tscal>{*unit_sys}.G();
        }
    }

    /// Retrieves the value of the constant c based on the unit system.
    inline Tscal get_constant_c() {
        if (!unit_sys) {
            ON_RANK_0(logger::warn_ln("sph::Config", "the unit system is not set"));
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.c();
        } else {
            return shamunits::Constants<Tscal>{*unit_sys}.c();
        }
    }

    /// Retrieves the value of the constant mu_0 based on the unit system.
    inline Tscal get_constant_mu_0() {
        if (!unit_sys) {
            ON_RANK_0(logger::warn_ln("sph::Config", "the unit system is not set"));
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.mu_0();
        } else {
            return shamunits::Constants<Tscal>{*unit_sys}.mu_0();
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Units Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Particle killing config
    //////////////////////////////////////////////////////////////////////////////////////////////

    ParticleKillingConfig<Tvec> particle_killing;

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Particle killing config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // CFL Configuration (config)
    //////////////////////////////////////////////////////////////////////////////////////////////

    CFLConfig<Tscal> cfl_config; ///< The configuration for the CFL condition

    /// Set the CFL multiplier for the stiffness
    inline void set_cfl_mult_stiffness(Tscal cstiff) {
        cfl_config.cfl_multiplier_stiffness = cstiff;
    }

    /// Get the CFL multiplier for the stiffness
    inline Tscal get_cfl_mult_stiffness() { return cfl_config.cfl_multiplier_stiffness; }

    bool show_cfl_detail = false;

    //////////////////////////////////////////////////////////////////////////////////////////////
    // CFL Configuration (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // MHD Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using MHDConfig      = MHDConfig<Tvec>;
    MHDConfig mhd_config = {};

    /// disable MHD in the SPH solver
    inline void set_noMHD() {
        using Tmp = typename MHDConfig::None;
        mhd_config.set(Tmp{});
    }

    /// Enable the ideal MHD hydro solver
    inline void set_IdealMHD(typename MHDConfig::IdealMHD_constrained_hyper_para v) {
        mhd_config.set(v);
    }

    inline void set_NonIdealMHD(typename MHDConfig::NonIdealMHD v) { mhd_config.set(v); }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // MHD Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Dust config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using DustConfig       = DustConfig<Tscal>;
    DustConfig dust_config = {};

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Dust config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Self gravity config
    //////////////////////////////////////////////////////////////////////////////////////////////

    SelfGravConfig self_grav_config = SelfGravConfig{};

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Self gravity config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Tree config
    //////////////////////////////////////////////////////////////////////////////////////////////

    u32 tree_reduction_level  = 3;    ///< Reduction level to be used in the tree build
    bool use_two_stage_search = true; ///< Use two stage neighbors search (see shamrock paper)

    /// Setter for the tree reduction level
    inline void set_tree_reduction_level(u32 level) { tree_reduction_level = level; }
    /// Setter for the two stage search
    inline void set_two_stage_search(bool enable) { use_two_stage_search = enable; }

    bool show_neigh_stats = false;
    inline void set_show_neigh_stats(bool enable) { show_neigh_stats = enable; }
    //////////////////////////////////////////////////////////////////////////////////////////////
    // Tree config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver behavior config
    //////////////////////////////////////////////////////////////////////////////////////////////

    bool combined_dtdiv_divcurlv_compute = false; ///< Use the combined dtdivv and divcurlv compute
    /// Factor applied to the smoothing length for neighbors search (and ghost zone size)
    /// @note This value must be larger or equal to htol_up_fine_cycle
    Tscal htol_up_coarse_cycle = 1.1;
    /// Maximum factor of the smoothing length evolution per subcycles
    Tscal htol_up_fine_cycle  = 1.1;
    Tscal epsilon_h           = 1e-6; ///< Convergence criteria for the smoothing length
    u32 h_iter_per_subcycles  = 50;   ///< Maximum number of iterations per subcycle
    u32 h_max_subcycles_count = 100;  ///< Maximum number of subcycles before solver crash

    SmoothingLengthConfig smoothing_length_config;

    inline void set_smoothing_length_density_based() {
        smoothing_length_config.set_density_based();
    }
    inline void set_smoothing_length_density_based_neigh_lim(u32 max_neigh_count) {
        smoothing_length_config.set_density_based_neigh_lim(max_neigh_count);
    }

    bool enable_particle_reordering = false;
    inline void set_enable_particle_reordering(bool enable) { enable_particle_reordering = enable; }
    u64 particle_reordering_step_freq = 1000;
    inline void set_particle_reordering_step_freq(u64 freq) {
        if (freq == 0) {
            shambase::throw_with_loc<std::invalid_argument>(
                "particle_reordering_step_freq cannot be zero");
        }
        particle_reordering_step_freq = freq;
    }

    bool save_dt_to_fields = false;
    inline void set_save_dt_to_fields(bool enable) { save_dt_to_fields = enable; }
    inline bool should_save_dt_to_fields() const { return save_dt_to_fields; }

    bool show_ghost_zone_graph = false;
    inline void set_show_ghost_zone_graph(bool enable) { show_ghost_zone_graph = enable; }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver behavior config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // EOS Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// Alias to EOSConfig type
    using EOSConfig = shammodels::EOSConfig<Tvec>;

    /// EOS configuration
    EOSConfig eos_config;

    /// Check if the EOS is a locally isothermal equation of state
    inline bool is_eos_locally_isothermal() {
        using T = typename EOSConfig::LocallyIsothermal;
        return bool(std::get_if<T>(&eos_config.config));
    }

    /// Check if the EOS is an adiabatic equation of state
    inline bool is_eos_adiabatic() {
        using T = typename EOSConfig::Adiabatic;
        return bool(std::get_if<T>(&eos_config.config));
    }

    /// Check if the EOS is a polytropic equation of state
    inline bool is_eos_polytropic() {
        using T = typename EOSConfig::Polytropic;
        return bool(std::get_if<T>(&eos_config.config));
    }

    /// Check if the EOS is an isothermal equation of state
    inline bool is_eos_isothermal() {
        using T = typename EOSConfig::Isothermal;
        return bool(std::get_if<T>(&eos_config.config));
    }

    /// Check if the EOS is a Fermi equation of state
    inline bool is_eos_fermi() {
        using T = typename EOSConfig::Fermi;
        return bool(std::get_if<T>(&eos_config.config));
    }

    /**
     * @brief Set the EOS configuration to an isothermal equation of state
     *
     * @param cs The isothermal index
     */
    inline void set_eos_isothermal(Tscal cs) { eos_config.set_isothermal(cs); }

    /**
     * @brief Set the EOS configuration to an adiabatic equation of state
     *
     * @param gamma The adiabatic index
     */
    inline void set_eos_adiabatic(Tscal gamma) { eos_config.set_adiabatic(gamma); }

    /**
     * @brief Set the EOS configuration to an polytropic equation of state
     *
     * @param gamma The adiabatic index
     */
    inline void set_eos_polytropic(Tscal K, Tscal gamma) { eos_config.set_polytropic(K, gamma); }

    /**
     * @brief Set the EOS configuration to a locally isothermal equation of state
     */
    inline void set_eos_locally_isothermal() { eos_config.set_locally_isothermal(); }

    /**
     * @brief Set the EOS configuration to a locally isothermal equation of state from Lodato
     * Price 2007
     *
     * @param cs0 Soundspeed at the reference radius
     * @param q Power exponent of the soundspeed profile
     * @param r0 Reference radius
     */
    inline void set_eos_locally_isothermalLP07(Tscal cs0, Tscal q, Tscal r0) {
        eos_config.set_locally_isothermalLP07(cs0, q, r0);
    }

    /**
     * @brief Set the EOS configuration to a locally isothermal equation of state fromFarris 2014
     *
     * @param cs0 Soundspeed at the reference radius
     * @param q Power exponent of the soundspeed profile
     * @param r0 Reference radius
     */
    inline void set_eos_locally_isothermalFA2014(Tscal h_over_r) {
        eos_config.set_locally_isothermalFA2014(h_over_r);
    }

    /**
     * @brief Set the EOS configuration to a locally isothermal equation of state from Farris 2014
     * extended to q != 1/2
     *
     * @param cs0 Soundspeed at the reference radius
     * @param q Power exponent of the soundspeed profile
     * @param r0 Reference radius
     * @param n_sinks Number of sinks to consider for the equation of state
     */
    inline void set_eos_locally_isothermalFA2014_extended(
        Tscal cs0, Tscal q, Tscal r0, u32 n_sinks) {
        eos_config.set_locally_isothermalFA2014_extended(cs0, q, r0, n_sinks);
    }

    /**
     * @brief Set the EOS configuration to a Fermi equation of state
     *
     * @param mu_e The mean molecular weight
     */
    inline void set_eos_fermi(Tscal mu_e) { eos_config.set_fermi(mu_e); }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // EOS Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Artificial viscosity Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Configuration for the Artificial Viscosity (AV)
     *
     * @details This struct contains the information needed to configure the Artificial Viscosity
     * in the SPH algorithm. It is a variant of two possible types of artificial viscosity:
     * - None: no AV
     * - Constant: AV with a constant value
     * - VaryingMM97: AV with a varying value, using the Monaghan & Gingold 1997 prescription
     * - VaryingCD10: AV with a varying value, using the Cullen & Dehnen 2010 prescription
     * - ConstantDisc: AV with a constant value, but only in the disc plane
     */
    using AVConfig = AVConfig<Tvec>;

    /// Configuration for the Artificial Viscosity (AV)
    AVConfig artif_viscosity;

    /**
     * @brief Set the artificial viscosity configuration to None
     */
    inline void set_artif_viscosity_None() {
        using Tmp = typename AVConfig::None;
        artif_viscosity.set(Tmp{});
    }

    /**
     * @brief Set the artificial viscosity configuration to a constant value
     *
     * @param v Constant value of the artificial viscosity
     */
    inline void set_artif_viscosity_Constant(typename AVConfig::Constant v) {
        artif_viscosity.set(v);
    }

    /**
     * @brief Set the artificial viscosity configuration to a varying value using
     * the prescription of Monaghan & Gingold 1997
     *
     * @param v Configuration of the artificial viscosity (alpha, beta, etc.)
     */
    inline void set_artif_viscosity_VaryingMM97(typename AVConfig::VaryingMM97 v) {
        artif_viscosity.set(v);
    }

    /**
     * @brief Set the artificial viscosity configuration to a varying value using
     * the prescription of Cullen & Dehnen 2010
     *
     * @param v Configuration of the artificial viscosity (alpha, beta, etc.)
     */
    inline void set_artif_viscosity_VaryingCD10(typename AVConfig::VaryingCD10 v) {
        artif_viscosity.set(v);
    }

    /**
     * @brief Set the artificial viscosity configuration to a constant value in the disc plane.
     * @param v Configuration of the artificial viscosity (alpha, beta, etc.)
     */
    inline void set_artif_viscosity_ConstantDisc(typename AVConfig::ConstantDisc v) {
        artif_viscosity.set(v);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Artificial viscosity Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Boundary Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Configuration of the boundary conditions
     */
    using BCConfig = BCConfig<Tvec>;

    /**
     * @brief Boundary condition configuration
     *
     * See the documentation of the `BCConfig` struct for more informations.
     */
    BCConfig boundary_config;

    /**
     * @brief Set the boundary condition to free boundary
     */
    inline void set_boundary_free() { boundary_config.set_free(); }

    /**
     * @brief Set the boundary condition to periodic boundary
     */
    inline void set_boundary_periodic() { boundary_config.set_periodic(); }

    /**
     * @brief Set the boundary condition to shearing periodic boundary
     *
     * The particles are periodic in all directions, but with a shear in the direction
     * given by `shear_dir` and a period of `speed`.
     *
     * @param[in] shear_base The base of the scalar product to define the number of shearing
     * periodicity to be applied
     * @param[in] shear_dir The direction of the shear
     * @param[in] speed The speed of the shear
     */
    inline void set_boundary_shearing_periodic(i32_3 shear_base, i32_3 shear_dir, Tscal speed) {
        boundary_config.set_shearing_periodic(shear_base, shear_dir, speed);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Boundary Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Ext force Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief External force configuration
     *
     * This configuration is used to define the external forces that are applied to the
     * particles in the simulation.
     *
     * The external forces are defined by a variant of different types of forces
     * (i.e., point mass, Lense-Thirring, etc.). The user can add different types
     * of forces using the functions `add_ext_force_point_mass`, `add_ext_force_lense_thirring`,
     * etc.
     */
    using ExtForceConfig = shammodels::ExtForceConfig<Tvec>;

    /**
     * @brief External force configuration
     */
    ExtForceConfig ext_force_config{};

    /**
     * @brief Add a point mass external force
     *
     * @param[in] central_mass The mass of the central object
     * @param[in] Racc The accretion radius of the central object
     */
    inline void add_ext_force_point_mass(Tscal central_mass, Tscal Racc) {
        ext_force_config.add_point_mass(central_mass, Racc);
    }

    /**
     * @brief Add a post-newtonian Paczynski-Wiita potential
     *
     * @param[in] central_mass The mass of the central object
     * @param[in] Racc The accretion radius of the central object
     */
    inline void add_ext_force_paczynski_wiita(Tscal central_mass, Tvec central_pos, Tscal Racc) {
        ext_force_config.add_paczynski_wiita(central_mass, central_pos, Racc);
    }

    /**
     * @brief Add a Lense-Thirring external force
     *
     * @param[in] central_mass The mass of the central object
     * @param[in] Racc The accretion radius of the central object
     * @param[in] a_spin The spin of the central object
     * @param[in] dir_spin The direction of the spin of the central object
     */
    inline void add_ext_force_lense_thirring(
        Tscal central_mass, Tscal Racc, Tscal a_spin, Tvec dir_spin) {
        ext_force_config.add_lense_thirring(central_mass, Racc, a_spin, dir_spin);
    }

    /**
     * @brief Add a shearing box external force
     *
     * @param[in] Omega_0 The angular frequency of the shear
     * @param[in] eta The shear rate
     * @param[in] q The power-law index of the shear
     */
    inline void add_ext_force_shearing_box(Tscal Omega_0, Tscal eta, Tscal q) {
        ext_force_config.add_shearing_box(Omega_0, eta, q);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Ext force Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Debug dump config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Whether to dump debug information to file
    bool do_debug_dump = false;

    /// @brief The filename to dump debug information in
    std::string debug_dump_filename = "";

    /// @brief Set whether to dump debug information to file
    ///
    /// @param[in] _do_debug_dump Whether to dump debug information to file
    /// @param[in] _debug_dump_filename The filename to dump debug information to
    inline void set_debug_dump(bool _do_debug_dump, std::string _debug_dump_filename) {
        this->do_debug_dump       = _do_debug_dump;
        this->debug_dump_filename = _debug_dump_filename;
    }

    /// @brief Whether to add debug fields to the pdl.
    inline constexpr bool do_MHD_debug() { return false; }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Debug dump config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Whether the ghost cells have a sound speed (i.e. the eos is locally isothermal)
    inline bool ghost_has_soundspeed() { return is_eos_locally_isothermal(); }

    /// @brief Whether the solver has a field for the particle's uint
    ///
    /// @note for now, this is always true as
    inline bool has_field_uint() {
        // no barotropic for now
        return true;
    }

    /// @brief Whether the solver has a field for alpha AV
    inline bool has_field_alphaAV() { return artif_viscosity.has_alphaAV_field(); }

    /// @brief Whether the solver has a field for divv
    inline bool has_field_divv() { return artif_viscosity.has_alphaAV_field(); }

    /// @brief Whether the solver has a field for dt divv
    inline bool has_field_dtdivv() { return artif_viscosity.has_dtdivv_field(); }

    /// @brief Whether the solver has a field for curlv
    inline bool has_field_curlv() { return artif_viscosity.has_curlv_field() && (dim == 3); }

    /// @brief Whether the solver has a field for ax, ay, az in ghost cells
    inline bool has_axyz_in_ghost() { return has_field_dtdivv(); }

    /// @brief Whether the solver has a field for sound speed
    inline bool has_field_soundspeed() {
        return artif_viscosity.has_field_soundspeed() || is_eos_locally_isothermal();
    }

    /// @brief Whether the solver has a field for B_on_rho
    inline bool has_field_B_on_rho() { return mhd_config.has_B_field() && (dim == 3); }

    /// @brief Whether the solver has a field for psi_on_ch
    inline bool has_field_psi_on_ch() { return mhd_config.has_psi_field(); }

    /// @brief Whether the solver has a field for divB
    inline bool has_field_divB() { return mhd_config.has_divB_field(); }

    /// @brief Whether the solver has a field for curlB
    inline bool has_field_curlB() { return mhd_config.has_curlB_field() && (dim == 3); }

    /// @brief Whether the solver has a field for dt divB
    inline bool has_field_dtdivB() { return mhd_config.has_dtdivB_field(); }

    /// @brief Whether to store luminosity
    bool compute_luminosity = false;
    inline void use_luminosity(bool enable) { compute_luminosity = enable; }

    /// Print the current status of the solver config
    void print_status();

    inline void check_config() {
        dust_config.check_config();

        if (track_particles_id && false /*particle injection when added*/) {
            shamrock::experimental_feature_check(
                "particle injection is not yet compatible with particle id tracking");
        }

        if (track_particles_id) {
            shamrock::experimental_feature_check("Particle tracking is experimental");
        }

        if (!self_grav_config.is_none()) {
            shamrock::experimental_feature_check(
                "Self gravity is experimental, please enable experimental features to use it");
        }
    }

    void set_layout(shamrock::patch::PatchDataLayerLayout &pdl);
    void set_ghost_layout(shamrock::patch::PatchDataLayerLayout &ghost_layout);
};

namespace shammodels::sph {

    template<class Tscal>
    void to_json(nlohmann::json &j, const CFLConfig<Tscal> &p);

    template<class Tscal>
    void from_json(const nlohmann::json &j, CFLConfig<Tscal> &p);

    template<class Tvec>
    void to_json(nlohmann::json &j, const ParticleKillingConfig<Tvec> &p);

    template<class Tvec>
    void from_json(const nlohmann::json &j, ParticleKillingConfig<Tvec> &p);

    void to_json(nlohmann::json &j, const SmoothingLengthConfig &p);
    void from_json(const nlohmann::json &j, SmoothingLengthConfig &p);

    void to_json(nlohmann::json &j, const SelfGravConfig &p);
    void from_json(const nlohmann::json &j, SelfGravConfig &p);

    template<class Tvec>
    void to_json(nlohmann::json &j, const DustConfig<Tvec> &p);

    template<class Tvec>
    void from_json(const nlohmann::json &j, DustConfig<Tvec> &p);

    template<class Tvec, template<class> class SPHKernel>
    void to_json(nlohmann::json &j, const SolverConfig<Tvec, SPHKernel> &p);

    template<class Tvec, template<class> class SPHKernel>
    void from_json(const nlohmann::json &j, SolverConfig<Tvec, SPHKernel> &p);

} // namespace shammodels::sph
