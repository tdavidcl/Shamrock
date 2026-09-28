// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file enum_NeighCacheStrategy.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Neighbour cache build strategy json deserialization
 */

#include "shammodels/common/config/enum_NeighCacheStrategy.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"

namespace shammodels {

    void get_to_neigh_cache_strategy(
        const nlohmann::json &j,
        NeighCacheStrategy &value,
        const std::string &log_ctx,
        bool &has_used_defaults,
        bool &has_updated_config) {

        if (j.contains(neigh_cache_strategy_json_key)) {
            j.at(neigh_cache_strategy_json_key).get_to(value);
            return;
        }

        if (j.contains(neigh_cache_strategy_legacy_json_key)) {
            value = neigh_cache_strategy_from_two_stage_search(
                j.at(neigh_cache_strategy_legacy_json_key).template get<bool>());
            has_updated_config = true;
            if (shamcomm::world_rank() == 0) {
                shamcomm::logs::warn_ln(
                    log_ctx,
                    "Updating old key [" + std::string(neigh_cache_strategy_legacy_json_key)
                        + "] to new key [" + std::string(neigh_cache_strategy_json_key)
                        + "] in from_json");
            }
            return;
        }

        has_used_defaults = true;
    }

} // namespace shammodels
