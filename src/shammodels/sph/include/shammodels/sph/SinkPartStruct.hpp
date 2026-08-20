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
 * @file SinkPartStruct.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "nlohmann/json_fwd.hpp"
#include "shambackends/vec.hpp"
namespace shammodels::sph {

    template<class Tvec>
    struct SinkParticle {

        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        Tvec pos;
        Tvec velocity;
        Tvec sph_acceleration;
        Tvec ext_acceleration;
        Tscal mass;
        Tvec angular_momentum;
        Tscal accretion_radius;
    };

    template<class Tvec>
    void to_json(nlohmann::json &j, const SinkParticle<Tvec> &p);

    template<class Tvec>
    void from_json(const nlohmann::json &j, SinkParticle<Tvec> &p);

} // namespace shammodels::sph
