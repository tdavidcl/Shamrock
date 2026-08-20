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
 * @file ExtForceConfig.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "nlohmann/json_fwd.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <type_traits>
#include <string>
#include <variant>

namespace shammodels {

    template<class Tvec>
    struct ExtForceVariant {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        struct PointMass {
            Tscal central_mass;
            Tscal Racc;
        };

        struct PN_PW {
            Tscal central_mass;
            Tvec central_pos;
            Tscal Racc;
        };

        struct LenseThirring {
            Tscal central_mass;
            Tscal Racc;
            Tscal a_spin;
            Tvec dir_spin;
        };

        /**
         * \brief Shearing box forces as in athena
         * \cite Stone2010_shear_box
         * \f[
         *  \mathbf{f} = 2\Omega_0 \left(  q \Omega_0 x +  v_y \right) \basevec{x} -2\Omega_0 v_x
         * \basevec{y} - \Omega_0^2 z \basevec{z}  \f] Shear speed : \f[ \omega = q \Omega_0 L_x \f]
         */
        struct ShearingBoxForce {
            i32_3 shear_base = {1, 0, 0};
            i32_3 shear_dir  = {0, 1, 0};

            Tscal Omega_0;
            Tscal eta;
            Tscal q;

            inline Tscal shear_speed(Tscal box_length) { return q * Omega_0 * box_length; }

            ShearingBoxForce() = default;
            ShearingBoxForce(Tscal Omega_0, Tscal eta, Tscal q)
                : Omega_0(Omega_0), eta(eta), q(q) {};
            ShearingBoxForce(i32_3 shear_base, i32_3 shear_dir, Tscal Omega_0, Tscal eta, Tscal q)
                : shear_base(shear_base), shear_dir(shear_dir), Omega_0(Omega_0), eta(eta), q(q) {};
        };

        /// f = -GMy / sqrt(R0^2 + y^2)
        struct VerticalDiscPotential {
            Tscal central_mass;
            Tscal R0;
        };

        /// f = -eta v
        struct VelocityDissipation {
            Tscal eta;
        };

        using VariantForce = std::variant<
            PointMass,
            PN_PW,
            LenseThirring,
            ShearingBoxForce,
            VerticalDiscPotential,
            VelocityDissipation>;
        VariantForce val;
    };

    template<class Tvec>
    struct ExtForceConfig {

        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using PointMass             = typename ExtForceVariant<Tvec>::PointMass;
        using PN_PW                 = typename ExtForceVariant<Tvec>::PN_PW;
        using LenseThirring         = typename ExtForceVariant<Tvec>::LenseThirring;
        using ShearingBoxForce      = typename ExtForceVariant<Tvec>::ShearingBoxForce;
        using VerticalDiscPotential = typename ExtForceVariant<Tvec>::VerticalDiscPotential;
        using VelocityDissipation   = typename ExtForceVariant<Tvec>::VelocityDissipation;

        std::vector<ExtForceVariant<Tvec>> ext_forces;

        inline void add_point_mass(Tscal central_mass, Tscal Racc) {
            ext_forces.push_back(ExtForceVariant<Tvec>{PointMass{central_mass, Racc}});
        }

        inline void add_paczynski_wiita(Tscal central_mass, Tvec central_pos, Tscal Racc) {
            ext_forces.push_back(ExtForceVariant<Tvec>{PN_PW{central_mass, central_pos, Racc}});
        }

        inline void add_lense_thirring(
            Tscal central_mass, Tscal Racc, Tscal a_spin, Tvec dir_spin) {
            if (sham::abs(sycl::length(dir_spin) - 1) > 1e-8) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "the sping direction should be a unit vector");
            }
            ext_forces.push_back(
                ExtForceVariant<Tvec>{LenseThirring{central_mass, Racc, a_spin, dir_spin}});
        }

        /**
         * @brief
         * \todo add check for norm of shear vectors
         */
        inline void add_shearing_box(Tscal Omega_0, Tscal eta, Tscal q) {

            ext_forces.push_back(ExtForceVariant<Tvec>{ShearingBoxForce{Omega_0, eta, q}});
        }

        inline void add_vertical_disc_potential(Tscal central_mass, Tscal R0) {
            ext_forces.push_back(ExtForceVariant<Tvec>{VerticalDiscPotential{central_mass, R0}});
        }

        inline void add_velocity_dissipation(Tscal eta) {
            ext_forces.push_back(ExtForceVariant<Tvec>{VelocityDissipation{eta}});
        }
    };

} // namespace shammodels

namespace shammodels {
    template<class Tvec>
    void to_json(nlohmann::json &j, const ExtForceVariant<Tvec> &p);

    template<class Tvec>
    void from_json(const nlohmann::json &j, ExtForceVariant<Tvec> &p);

    template<class Tvec>
    void to_json(nlohmann::json &j, const ExtForceConfig<Tvec> &p);

    template<class Tvec>
    void from_json(const nlohmann::json &j, ExtForceConfig<Tvec> &p);
} // namespace shammodels
