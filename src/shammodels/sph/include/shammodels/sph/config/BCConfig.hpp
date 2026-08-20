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
 * @file BCConfig.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "nlohmann/json_fwd.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <variant>

namespace shammodels::sph {

    /**
     * @brief Boundary conditions configuration
     *
     * This struct is used to configure the boundary conditions of a simulation.
     *
     * @tparam Tvec The vector type used for the simulation.
     */
    template<class Tvec>
    struct BCConfig;

} // namespace shammodels::sph

template<class Tvec>
struct shammodels::sph::BCConfig {

    /// Type of the components of the vector of coordinates
    using Tscal = shambase::VecComponent<Tvec>;
    /// Number of dimensions of the problem
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    /**
     * @brief Free boundary condition
     *
     * The box will be expanded if a particle is outside of it.
     */
    struct Free {
        /**
         * @brief The tolerance for the box expansion
         *
         * If a particle is outside of the box, the box will be expanded to the new range with an
         * added margin factor of expand_tolerance
         */
        Tscal expand_tolerance = 1.2;
    };

    /**
     * @brief Periodic boundary condition
     */
    struct Periodic {};

    /**
     * @brief Shearing periodic boundary condition
     * @todo use a bib entry instead
     * @see https://ui.adsabs.harvard.edu/abs/2010ApJS..189..142S/abstract
     */

    struct ShearingPeriodic {
        /**
         * @brief The base of the scalar product to define the number of shearing periodicity to be
         * applied
         */
        i32_3 shear_base;

        /**
         * @brief The direction of the shear
         */
        i32_3 shear_dir;

        /**
         * @brief The speed of the shear
         */
        Tscal shear_speed;
    };

    /// Variant of all types of artificial viscosity possible
    using Variant = std::variant<Free, Periodic, ShearingPeriodic>;

    /// The actual configuration (default to free boundaries)
    Variant config = Free{};

    /// Set the boundary condition to free boundaries
    inline void set_free() { config = Free{}; }

    /// Set the boundary condition to periodic boundaries
    inline void set_periodic() { config = Periodic{}; }

    /**
     * @brief Set the boundary condition to shearing periodic boundaries
     *
     * @param shear_base The base of the scalar product to define the number of shearing periodicity
     * to be applied
     * @param shear_dir The direction of the shear
     * @param speed The speed of the shear
     */
    inline void set_shearing_periodic(i32_3 shear_base, i32_3 shear_dir, Tscal speed) {
        config = ShearingPeriodic{shear_base, shear_dir, speed};
    }

    /**
     * @brief Prints the current boundary condition configuration to the logger.
     *
     * The function logs the type of boundary condition (free, periodic, or shearing periodic)
     * and the relevant parameters for the shearing periodic case.
     */
    inline void print_status() {
        logger::raw_ln("--- Bondaries config");

        if (Free *v = std::get_if<Free>(&config)) {
            logger::raw_ln("  Config Type : Free boundaries");
        } else if (Periodic *v = std::get_if<Periodic>(&config)) {
            logger::raw_ln("  Config Type : Periodic boundaries");
        } else if (ShearingPeriodic *v = std::get_if<ShearingPeriodic>(&config)) {
            logger::raw_ln("  Config Type : ShearingPeriodic (Stone 2010)");
            logger::raw_ln("  shear_base   =", v->shear_base);
            logger::raw_ln("  shear_dir   =", v->shear_dir);
            logger::raw_ln("  shear_speed =", v->shear_speed);
        } else {
            shambase::throw_unimplemented();
        }

        logger::raw_ln("--- Bondaries config config (deduced)");

        logger::raw_ln("-------------");
    }
};

namespace shammodels::sph {

    /**
     * @brief Serialize a BCConfig to a JSON object
     *
     * @param[out] j  The JSON object to write to
     * @param[in] p  The BCConfig to serialize
     */
    template<class Tvec>
    void to_json(nlohmann::json &j, const BCConfig<Tvec> &p);

    template<class Tvec>
    void from_json(const nlohmann::json &j, BCConfig<Tvec> &p);

} // namespace shammodels::sph
