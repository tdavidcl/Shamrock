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
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Benoit Commercon (benoit.commercon@ens-lyon.fr)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Noé Brucy (noe.brucy@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/overloaded.hpp"
#include "shambase/string.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/riemann_common.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/ramses/config/CoolingConfig.hpp"
#include "shammodels/ramses/config/ExtForceConfig.hpp"
#include "shammodels/ramses/config/enum_AMRInterpMode.hpp"
#include "shammodels/ramses/config/enum_DragSolverMode.hpp"
#include "shammodels/ramses/config/enum_DustRiemannSolverMode.hpp"
#include "shammodels/ramses/config/enum_GravityMode.hpp"
#include "shammodels/ramses/config/enum_RiemannSolverMode.hpp"
#include "shammodels/ramses/config/enum_SlopeMode.hpp"
#include "shamrock/experimental_features.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include <nlohmann/json.hpp>
#include <shamrock/io/json_std_optional.hpp>
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
#include <shamunits/constant_table.hpp>
#include <stdexcept>
#include <variant>

namespace shammodels::basegodunov {

    /**
     * @brief alphas is the dust collision rate (the inverse of the stopping time)
     */
    struct DragConfig {
        DragSolverMode drag_solver_config = NoDrag;
        std::vector<f64> alphas;
        bool enable_frictional_heating
            = false; // 0 to turn off and 1 when all dissipation is deposited to the gas
    };

    struct DustConfig {
        DustRiemannSolverMode dust_riemann_config = NoDust;
        u32 ndust                                 = 0;

        inline bool is_dust_on() {
            if (dust_riemann_config != NoDust) {

                if (ndust == 0) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "Dust is on with ndust == 0");
                }
                return true;
            }
            return false;
        }
    };
    /**
     * @brief Npscal_gas is the number of gas passive scalars
     */
    struct PassiveScalarGasConfig {
        u32 npscal_gas = 0;

        inline bool is_gas_passive_scalar_on() { return npscal_gas > 0; }
    };

    template<class Tvec>
    struct GravityConfig {
        using Tscal              = shambase::VecComponent<Tvec>;
        GravityMode gravity_mode = NoGravity;
        /// Convergence tolerance of the Poisson solver
        Tscal tol = 1e-6;
        /// Tolerance of the BiCGSTAB lucky breakdown test (restarts the solver below it)
        Tscal tol_hp_bk = 1e-6;
        /// Maximum number of iterations of the Poisson solver
        u32 Niter_max = 100;
        inline Tscal get_tolerance() { return tol; }
        inline bool is_gravity_on() { return gravity_mode != NoGravity; }
    };

    template<class Tvec, class TgridVec>
    struct AMRMode {

        using Tscal = shambase::VecComponent<Tvec>;

        struct None {};

        struct DensityBased {
            Tscal crit_mass;
        };

        struct PseudoGradientBased {
            Tscal error_min;
            Tscal error_max;
        };

        /**
         * @brief Refine the blocks where the Jeans length is resolved by less than N_J cells
         *
         * A block is refined if one of its cells has \f$ N_J \Delta x > \lambda_J =
         * \sqrt{\pi c_s^2 / (G \rho)} \f$, with the sound speed of the equation of state (of
         * the barotrope with the InstantBarotropic cooling). It is derefined if the coarse cells of
         * size
         * \f$ 2 \Delta x \f$ would still resolve the Jeans length of all its cells by more than
         * \f$ \sqrt{2} N_J \f$ cells (hysteresis margin).
         */
        struct JeansLengthBased {
            u32 N_J = 4;
        };

        struct ShearBased {
            Tscal threshold;
        };

        using mode
            = std::variant<None, DensityBased, PseudoGradientBased, JeansLengthBased, ShearBased>;

        mode config = None{};

        bool old_amr = true;

        void set_refine_none() { config = None{}; }
        void set_refine_density_based(Tscal crit_mass) { config = DensityBased{crit_mass}; }
        void set_refine_pseudo_gradient_based(Tscal error_min, Tscal error_max) {
            config = PseudoGradientBased{error_min, error_max};
        }

        void set_refine_jeans_length_based(u32 N_J) { config = JeansLengthBased{N_J}; }

        void set_refine_shear_based(Tscal thresh) { config = ShearBased{thresh}; }

        bool need_level_zero_compute() { return !old_amr; }
        bool need_amr_level_compute() { return !old_amr; }
    };

    /**
     * @brief Equation of state of the gas, each alternative maps to a shammath FluidStateSpec
     *
     * The solver supports the Adiabatic alternative only for now. The RAMSES-like reset of the
     * internal energy onto a barotrope is a cooling, see CoolingConfig::InstantBarotropic.
     */
    template<class Tvec>
    struct EOSConfig {
        using Tscal = shambase::VecComponent<Tvec>;

        /// Ideal gas equation of state, see shammath::FluidStateAdiabatic
        struct Adiabatic {
            Tscal gamma = 5. / 3.;

            inline shammath::FluidStateAdiabatic<Tvec> get_spec() const {
                return shammath::FluidStateAdiabatic<Tvec>{.m_gamma = gamma};
            }
        };

        /// Isothermal below rho_crit and adiabatic above, see shammath::FluidStateBarotropic
        struct Barotropic {
            Tscal rho_crit;
            Tscal cs0;
            Tscal gamma = 5. / 3.;

            inline shammath::FluidStateBarotropic<Tvec> get_spec() const {
                return shammath::FluidStateBarotropic<Tvec>(rho_crit, cs0, gamma);
            }
        };

        using Variant = std::variant<Adiabatic, Barotropic>;

        Variant config = Adiabatic{};

        inline void set_adiabatic(Tscal gamma) { config = Adiabatic{gamma}; }
        inline void set_barotropic(Tscal rho_crit, Tscal cs0, Tscal gamma) {
            config = Barotropic{rho_crit, cs0, gamma};
        }

        /// True if the hydro step is the adiabatic one
        inline bool is_hydro_adiabatic() const { return std::holds_alternative<Adiabatic>(config); }
    };

    struct BCConfig {
        enum class GhostType { Periodic = 0, Reflective = 1, Outflow = 2, ShearingPeriodic = 3 };

        GhostType ghost_type_x = GhostType::Periodic;
        GhostType ghost_type_y = GhostType::Periodic;
        GhostType ghost_type_z = GhostType::Periodic;

        /**
         * @brief Parameters of the shearing-periodic boundary
         *
         * Same convention as the SPH solver: the periodic image of the box shifted by
         * \f$ +L \f$ along `shear_base_axis` is displaced by \f$ +v_{\rm shear} t \f$ along
         * `shear_dir_axis`, and its velocity by \f$ +v_{\rm shear} \f$ along that axis. For a
         * shearing box with background flow \f$ v_y = -q \Omega x \f$, this is
         * `shear_base_axis = 0`, `shear_dir_axis = 1` and \f$ v_{\rm shear} = -q \Omega L_x \f$.
         */
        struct ShearingPeriodicInfo {
            u32 shear_base_axis = 0; ///< axis whose two faces are shearing periodic
            u32 shear_dir_axis  = 1; ///< axis along which the periodic images slide
            f64 shear_speed     = 0; ///< velocity of the image at +L along shear_dir_axis
        };

        /// Only used if one of the axes is ShearingPeriodic
        ShearingPeriodicInfo shearing{};

        GhostType get_x() const { return ghost_type_x; }
        GhostType get_y() const { return ghost_type_y; }
        GhostType get_z() const { return ghost_type_z; }

        /// Ghost type of the axis `axis` (0 = x, 1 = y, 2 = z)
        GhostType get(u32 axis) const {
            switch (axis) {
            case 0 : return ghost_type_x;
            case 1 : return ghost_type_y;
            case 2 : return ghost_type_z;
            default: break;
            }
            shambase::throw_with_loc<std::invalid_argument>(
                "Invalid axis index: " + std::to_string(axis));
            return {};
        }

        void set_x(GhostType ghost_type) { ghost_type_x = ghost_type; }
        void set_y(GhostType ghost_type) { ghost_type_y = ghost_type; }
        void set_z(GhostType ghost_type) { ghost_type_z = ghost_type; }

        /// Set the ghost type of the axis `axis` (0 = x, 1 = y, 2 = z)
        void set(u32 axis, GhostType ghost_type) {
            switch (axis) {
            case 0 : ghost_type_x = ghost_type; return;
            case 1 : ghost_type_y = ghost_type; return;
            case 2 : ghost_type_z = ghost_type; return;
            default: break;
            }
            shambase::throw_with_loc<std::invalid_argument>(
                "Invalid axis index: " + std::to_string(axis));
        }

        /// True if one of the axes uses the shearing-periodic boundary
        bool is_shearing_periodic() const {
            return ghost_type_x == GhostType::ShearingPeriodic
                   || ghost_type_y == GhostType::ShearingPeriodic
                   || ghost_type_z == GhostType::ShearingPeriodic;
        }

        /**
         * @brief Make the faces normal to `shear_base` shearing periodic
         *
         * `shear_base` and `shear_dir` must be distinct unit axis vectors (e.g. (1,0,0) and
         * (0,1,0)), matching the arguments of the SPH solver's `set_boundary_shearing_periodic`.
         * The `shear_dir` axis must be periodic, which is checked in `check_config`.
         */
        void set_shearing_periodic(i32_3 shear_base, i32_3 shear_dir, f64 shear_speed) {
            auto get_axis = [](i32_3 v, const char *name) -> u32 {
                u32 axis  = 0;
                u32 count = 0;
                for (u32 i = 0; i < 3; i++) {
                    if (v[i] == 1) {
                        axis = i;
                        count++;
                    } else if (v[i] != 0) {
                        count = 2; // invalid value, reported below
                    }
                }
                if (count != 1) {
                    shambase::throw_with_loc<std::invalid_argument>(sham::format(
                        "{} must be a unit axis vector such as (1,0,0), got ({},{},{})",
                        name,
                        v[0],
                        v[1],
                        v[2]));
                }
                return axis;
            };

            u32 base_axis = get_axis(shear_base, "shear_base");
            u32 dir_axis  = get_axis(shear_dir, "shear_dir");

            if (base_axis == dir_axis) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "shear_base and shear_dir must be different axes");
            }

            // only one shearing-periodic axis at a time
            for (u32 i = 0; i < 3; i++) {
                if (get(i) == GhostType::ShearingPeriodic) {
                    set(i, GhostType::Periodic);
                }
            }

            set(base_axis, GhostType::ShearingPeriodic);
            shearing = {base_axis, dir_axis, shear_speed};
        }
    };

    template<class Tvec, class TgridVec>
    struct SolverConfig;

}; // namespace shammodels::basegodunov

template<class Tvec, class TgridVec>
struct shammodels::basegodunov::SolverConfig {

    using Tscal = shambase::VecComponent<Tvec>;

    Tscal grid_coord_to_pos_fact = 1;

    static constexpr u32 NsideBlockPow = 1;
    using AMRBlock                     = amr::AMRBlock<Tvec, TgridVec, NsideBlockPow>;

    /// Equation of state of the gas
    EOSConfig<Tvec> eos_config{};

    /// Set an adiabatic equation of state (kept for backward compatibility)
    inline void set_eos_gamma(Tscal gamma) { eos_config.set_adiabatic(gamma); }

    /// Adiabatic index of the hydro step, throws if the hydro step is not adiabatic
    inline Tscal get_eos_gamma() const {
        using Adiabatic = typename EOSConfig<Tvec>::Adiabatic;
        if (const Adiabatic *cfg = std::get_if<Adiabatic>(&eos_config.config)) {
            return cfg->gamma;
        }
        shambase::throw_with_loc<std::invalid_argument>(
            "The Ramses solver only supports an adiabatic hydro step (Adiabatic equation of "
            "state) for now");
        return {};
    }

    /// Cooling of the gas
    CoolingConfig<Tvec> cooling_config{};

    inline bool is_cooling_on() const { return cooling_config.is_cooling_on(); }

    RiemannSolverMode riemann_config  = HLL;
    SlopeMode slope_config            = VanLeer_sym;
    bool face_half_time_interpolation = true;
    AMRInterpMode amr_interp_mode     = FIRST_ORDER;

    /// Set the gas Riemann solver
    inline void set_riemann_solver(RiemannSolverMode mode) { riemann_config = mode; }

    inline bool should_compute_rho_mean() { return is_gravity_on() && is_boundary_periodic(); }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Dust config
    //////////////////////////////////////////////////////////////////////////////////////////////

    DustConfig dust_config{};
    DragConfig drag_config{};

    inline bool is_dust_on() { return dust_config.is_dust_on(); }
    // get alpha values from user
    // alphas is the dust collision rate (the inverse of the stopping time)
    inline void set_alphas_static(f32 alpha_values) {
        StackEntry stack_lock{};
        drag_config.alphas.push_back(alpha_values);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Dust config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    BCConfig bc_config{};

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Gas passive scalars config
    //////////////////////////////////////////////////////////////////////////////////////////////

    PassiveScalarGasConfig npscal_gas_config{};

    inline bool is_gas_passive_scalar_on() { return npscal_gas_config.is_gas_passive_scalar_on(); }
    //////////////////////////////////////////////////////////////////////////////////////////////
    // Gas passive scalars config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Gravity config
    //////////////////////////////////////////////////////////////////////////////////////////////
    /// The unit system of the config, or the default one (SI) if it is not set
    inline shamunits::UnitSystem<Tscal> get_unit_sys() {
        if (!unit_sys) {
            ON_RANK_0(logger::warn_ln("amr::Config", "the unit system is not set"));
            return shamunits::UnitSystem<Tscal>{};
        } else {
            return *unit_sys;
        }
    }
    inline Tscal get_constant_G() { return get_unit_sys().template get<shamunits::constants::G>(); }
    inline Tscal get_constant_4piG() {
        return get_unit_sys().template get<4 * shamunits::pi<double> * shamunits::constants::G>();
    }
    inline bool is_boundary_periodic() { return true; }
    GravityConfig<Tvec> gravity_config{};
    inline Tscal get_grav_tol() { return gravity_config.get_tolerance(); }
    inline Tscal get_happy_bk_grav_tol() { return gravity_config.tol_hp_bk; }
    inline bool is_gravity_on() { return gravity_config.is_gravity_on(); }
    inline bool is_coordinate_field_required() { return ext_force_config.is_coordinates_needed(); }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Gravity config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // External forces config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using ExtForceConfig = shammodels::basegodunov::ExtForceConfig<Tvec>;

    /// External forces applied to the gas
    ExtForceConfig ext_force_config{};

    inline bool has_ext_forces() { return ext_force_config.has_ext_forces(); }

    inline bool has_shearing_box_forces() { return ext_force_config.has_shearing_box(); }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // External forces config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// AMR refinement mode
    AMRMode<Tvec, TgridVec> amr_mode = {};

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Units Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// The unit system of the simulation
    std::optional<shamunits::UnitSystem<Tscal>> unit_sys = {};

    /// Set the unit system of the simulation
    inline void set_units(shamunits::UnitSystem<Tscal> new_sys) { unit_sys = new_sys; }
    //////////////////////////////////////////////////////////////////////////////////////////////
    // Units Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    PatchSchedulerConfig scheduler_conf = {};

    //////////////////////////////////////////////////////////////////////////////////////////////
    // CFL Configuration (config)
    //////////////////////////////////////////////////////////////////////////////////////////////

    Tscal Csafe = 0.9;

    //////////////////////////////////////////////////////////////////////////////////////////////
    // CFL Configuration (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    inline void check_config() {
        if (grid_coord_to_pos_fact <= 0) {
            shambase::throw_with_loc<std::runtime_error>(
                sham::format("grid_coord_to_pos_fact must be > 0, got {}", grid_coord_to_pos_fact));
        }

        if (is_dust_on()) {
            ON_RANK_0(logger::warn_ln("Ramses::SolverConfig", "Dust is experimental"));
        }

        if (is_gravity_on()) {
            if (gravity_config.gravity_mode != CG && gravity_config.gravity_mode != BICGSTAB) {
                shambase::throw_with_loc<std::invalid_argument>(sham::format(
                    "self-gravity: only the cg and bicgstab Poisson solvers are implemented, got "
                    "gravity mode {}",
                    nlohmann::json(gravity_config.gravity_mode).dump()));
            }

            // self-gravity is applied through the external forces, its acceleration at t^{n+1}
            // only depending on the density at t^{n+1} (see Solver::init_solver_graph)
            if (ext_force_config.coupling != ExtForceCoupling::Trapezoidal) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "self-gravity requires the trapezoidal coupling of the external forces");
            }

            // Combinations that were not validated with this self-gravity scheme (only gas
            // self-gravity was). The external forces path would apply the self-gravity
            // acceleration to the dust as well, and the drag integrator is not coupled to it.
            if (is_dust_on()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "self-gravity with dust is not supported yet");
            }
            if (drag_config.drag_solver_config != NoDrag) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "self-gravity with drag is not supported yet");
            }

            ON_RANK_0(logger::warn_ln("Ramses::SolverConfig", "Self gravity is experimental"));
            u32 mode = gravity_config.gravity_mode;

            shamrock::experimental_feature_check(
                sham::format(
                    "self gravity mode is not enabled but gravity mode is set to {} (> 0 whith 0 "
                    "== "
                    "NoGravity mode)",
                    mode));
        }

        using EOSAdiabatic = typename EOSConfig<Tvec>::Adiabatic;
        if (const EOSAdiabatic *cfg = std::get_if<EOSAdiabatic>(&eos_config.config)) {
            if (!(cfg->gamma > 1.0)) {
                shambase::throw_with_loc<std::invalid_argument>(
                    sham::format("Gamma must be > 1, currently Gamma = {}", cfg->gamma));
            }
        } else {
            shambase::throw_with_loc<std::invalid_argument>(
                "The barotropic equation of state is not supported by the Ramses solver yet");
        }

        cooling_config.check_config();

        if (is_gas_passive_scalar_on()) {
            ON_RANK_0(logger::warn_ln("Ramses::SolverConfig", "Passive scalars are experimental"));
            shamrock::experimental_feature_check(
                sham::format(
                    "gas passive scalars mode is not enabled but gas passive scalars mode is set "
                    "to {}"
                    "> 0",
                    npscal_gas_config.npscal_gas));
        }

        if (!amr_mode.old_amr) {
            shamrock::experimental_feature_check("new AMR is experimental");
        }

        using AMRNone         = typename AMRMode<Tvec, TgridVec>::None;
        using AMRDensityBased = typename AMRMode<Tvec, TgridVec>::DensityBased;
        if (amr_mode.old_amr
            && !(
                std::holds_alternative<AMRNone>(amr_mode.config)
                || std::holds_alternative<AMRDensityBased>(amr_mode.config))) {
            shambase::throw_with_loc<std::invalid_argument>(
                "the old AMR only implements the None and DensityBased refinement criteria, the "
                "selected one is only implemented in the new AMR (cfg.set_amr_mode_old(False))");
        }

        using AMRJeansLengthBased = typename AMRMode<Tvec, TgridVec>::JeansLengthBased;
        if (const AMRJeansLengthBased *cfg = std::get_if<AMRJeansLengthBased>(&amr_mode.config)) {
            if (cfg->N_J == 0) {
                shambase::throw_with_loc<std::invalid_argument>("N_J must be > 0");
            }
        }

        check_config_shearing_periodic();
        check_config_shearing_box_forces();
    }

    /// Checks of the shearing box forces (no-op if they are not used)
    inline void check_config_shearing_box_forces() {
        if (!has_shearing_box_forces()) {
            return;
        }

        // the forces use x as the radial and y as the azimuthal axis
        if (bc_config.is_shearing_periodic()) {
            const auto &shear = bc_config.shearing;
            if (shear.shear_base_axis != 0 || shear.shear_dir_axis != 1) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "the shearing box forces require the shearing-periodic boundary to be set "
                    "with shear_base = (1,0,0) and shear_dir = (0,1,0)");
            }
        }

        ON_RANK_0(
            logger::warn_ln("Ramses::SolverConfig", "The shearing box forces are experimental"));
    }

    /// Checks of the shearing-periodic boundary (no-op if it is not used)
    inline void check_config_shearing_periodic() {
        using GhostType = BCConfig::GhostType;

        if (!bc_config.is_shearing_periodic()) {
            return;
        }

        const auto &shear = bc_config.shearing;

        u32 shearing_axis_count = 0;
        for (u32 i = 0; i < 3; i++) {
            shearing_axis_count += (bc_config.get(i) == GhostType::ShearingPeriodic);
        }

        if (shearing_axis_count != 1 || shear.shear_base_axis > 2 || shear.shear_dir_axis > 2
            || shear.shear_base_axis == shear.shear_dir_axis
            || bc_config.get(shear.shear_base_axis) != GhostType::ShearingPeriodic) {
            shambase::throw_with_loc<std::invalid_argument>(
                "the shearing-periodic boundary must be set with set_boundary_shearing_periodic "
                "(exactly one shearing-periodic axis, distinct from the shear direction)");
        }

        if (bc_config.get(shear.shear_dir_axis) != GhostType::Periodic) {
            shambase::throw_with_loc<std::invalid_argument>(
                "the shearing-periodic boundary requires the shear direction axis to be "
                "periodic");
        }

        // The current implementation interpolates the ghost cells assuming that the cells on the
        // shearing faces all have the same size (see InterpolateShearGhosts).
        using AMRNone = typename AMRMode<Tvec, TgridVec>::None;
        if (!std::holds_alternative<AMRNone>(amr_mode.config)) {
            shambase::throw_with_loc<std::invalid_argument>(
                "the shearing-periodic boundary does not support mesh refinement yet");
        }

        // The Poisson solvers exchange the ghosts of phi without the shearing interpolation
        if (is_gravity_on()) {
            shambase::throw_with_loc<std::invalid_argument>(
                "the shearing-periodic boundary does not support self-gravity yet");
        }

        ON_RANK_0(
            logger::warn_ln(
                "Ramses::SolverConfig",
                "The shearing-periodic boundary is experimental (no flux correction across the "
                "shearing faces, so the mass is only conserved to truncation error)"));
    }

    void set_layout(shamrock::patch::PatchDataLayerLayout &pdl);
};

namespace shammodels::basegodunov {

    inline void to_json(nlohmann::json &j, const BCConfig::GhostType &e) {
        switch (e) {
        case BCConfig::GhostType::Periodic        : j = "periodic"; break;
        case BCConfig::GhostType::Reflective      : j = "reflective"; break;
        case BCConfig::GhostType::Outflow         : j = "outflow"; break;
        case BCConfig::GhostType::ShearingPeriodic: j = "shearing_periodic"; break;
        default:
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid BCConfig::GhostType value: " + std::to_string(static_cast<int>(e)));
        }
    }

    inline void from_json(const nlohmann::json &j, BCConfig::GhostType &e) {
        const std::string type = j.get<std::string>();
        if (type == "periodic") {
            e = BCConfig::GhostType::Periodic;
        } else if (type == "reflective") {
            e = BCConfig::GhostType::Reflective;
        } else if (type == "outflow") {
            e = BCConfig::GhostType::Outflow;
        } else if (type == "shearing_periodic") {
            e = BCConfig::GhostType::ShearingPeriodic;
        } else {
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid BCConfig::GhostType value: " + type);
        }
    }

    inline void to_json(nlohmann::json &j, const BCConfig::ShearingPeriodicInfo &p) {
        j = nlohmann::json{
            {"shear_base_axis", p.shear_base_axis},
            {"shear_dir_axis", p.shear_dir_axis},
            {"shear_speed", p.shear_speed}};
    }

    inline void from_json(const nlohmann::json &j, BCConfig::ShearingPeriodicInfo &p) {
        j.at("shear_base_axis").get_to(p.shear_base_axis);
        j.at("shear_dir_axis").get_to(p.shear_dir_axis);
        j.at("shear_speed").get_to(p.shear_speed);
    }

    inline void to_json(nlohmann::json &j, const BCConfig &p) {
        j = nlohmann::json{
            {"ghost_type_x", p.ghost_type_x},
            {"ghost_type_y", p.ghost_type_y},
            {"ghost_type_z", p.ghost_type_z},
            {"shearing", p.shearing}};
    }

    inline void from_json(const nlohmann::json &j, BCConfig &p) {
        j.at("ghost_type_x").get_to(p.ghost_type_x);
        j.at("ghost_type_y").get_to(p.ghost_type_y);
        j.at("ghost_type_z").get_to(p.ghost_type_z);
        // absent from configs written before the shearing-periodic boundary existed
        if (j.contains("shearing")) {
            j.at("shearing").get_to(p.shearing);
        }
    }

    inline void to_json(nlohmann::json &j, const DragConfig &p) {
        j = nlohmann::json{
            {"drag_solver", p.drag_solver_config},
            {"alphas", p.alphas},
            {"enable_frictional_heating", p.enable_frictional_heating}};
    }

    inline void from_json(const nlohmann::json &j, DragConfig &p) {
        j.at("drag_solver").get_to(p.drag_solver_config);
        j.at("alphas").get_to(p.alphas);
        j.at("enable_frictional_heating").get_to(p.enable_frictional_heating);
    }

    template<class Tvec, class TgridVec>
    inline void amr_config_to_json(nlohmann::json &j, const AMRMode<Tvec, TgridVec> &p) {
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
            j = {{"type", "jeans_length_based"}, {"N_J", cfg->N_J}};
        } else if (const auto *cfg = std::get_if<typename AMR::ShearBased>(&p.config)) {
            j = {{"type", "shear_based"}, {"threshold", cfg->threshold}};
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec, class TgridVec>
    inline void amr_config_from_json(const nlohmann::json &j, AMRMode<Tvec, TgridVec> &p) {
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
            p.set_refine_jeans_length_based(j.at("N_J").get<u32>());
        } else if (type == "shear_based") {
            p.set_refine_shear_based(j.at("threshold").get<Tscal>());
        } else {
            shambase::throw_with_loc<std::runtime_error>("Invalid AMR mode type: " + type);
        }
    }

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const EOSConfig<Tvec> &p) {
        using EOS = EOSConfig<Tvec>;

        std::visit(
            shambase::overloaded{
                [&](const typename EOS::Adiabatic &cfg) {
                    j = {{"type", "adiabatic"}, {"gamma", cfg.gamma}};
                },
                [&](const typename EOS::Barotropic &cfg) {
                    j
                        = {{"type", "barotropic"},
                           {"rho_crit", cfg.rho_crit},
                           {"cs0", cfg.cs0},
                           {"gamma", cfg.gamma}};
                },
            },
            p.config);
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, EOSConfig<Tvec> &p) {
        using Tscal = shambase::VecComponent<Tvec>;

        const std::string type = j.at("type").get<std::string>();
        if (type == "adiabatic") {
            p.set_adiabatic(j.at("gamma").get<Tscal>());
        } else if (type == "barotropic") {
            p.set_barotropic(
                j.at("rho_crit").get<Tscal>(),
                j.at("cs0").get<Tscal>(),
                j.at("gamma").get<Tscal>());
        } else if (type == "reset_barotropic") {
            // legacy EOS, now an adiabatic EOS + CoolingConfig::InstantBarotropic, only
            // converted when deserializing a whole SolverConfig
            shambase::throw_with_loc<std::runtime_error>(
                "The reset_barotropic EOS was replaced by an adiabatic EOS with the "
                "instant_barotropic cooling, it can only be converted when deserializing a "
                "SolverConfig");
        } else {
            shambase::throw_with_loc<std::runtime_error>("Invalid EOS type: " + type);
        }
    }

    template<class Tvec, class TgridVec>
    inline void to_json(nlohmann::json &j, const AMRMode<Tvec, TgridVec> &p) {
        nlohmann::json config_j;
        amr_config_to_json(config_j, p);
        j = nlohmann::json{{"old_amr", p.old_amr}, {"config", config_j}};
    }

    template<class Tvec, class TgridVec>
    inline void from_json(const nlohmann::json &j, AMRMode<Tvec, TgridVec> &p) {
        j.at("old_amr").get_to(p.old_amr);
        amr_config_from_json(j.at("config"), p);
    }

    /**
     * @brief Serialize a SolverConfig to a JSON object
     *
     * @param[out] j  The JSON object to write to
     * @param[in] p  The SolverConfig to serialize
     */
    template<class Tvec, class TgridVec>
    void to_json(nlohmann::json &j, const SolverConfig<Tvec, TgridVec> &p);
    /**
     * @brief Deserializes a SolverConfig object from a JSON object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The SolverConfig object to populate.
     */
    template<class Tvec, class TgridVec>
    void from_json(const nlohmann::json &j, SolverConfig<Tvec, TgridVec> &p);

} // namespace shammodels::basegodunov
