// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file gpu_core_timeline.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief json dump of the GPU core timeline profiler
 *
 */

#include "shambackends/gpu_core_timeline.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <vector>

NLOHMANN_JSON_NAMESPACE_BEGIN
template<>
struct adl_serializer<sham::TimelineEvent> {
    static void to_json(json &j, const sham::TimelineEvent &e) {
        j
            = {{"start", e.start},
               {"first_end", e.first_end},
               {"last_end", e.last_end},
               {"color", e.color},
               {"lane", e.lane}};
    }
};
NLOHMANN_JSON_NAMESPACE_END

namespace sham {

    void gpu_core_timeline_profilier::dump_to_file(const std::string &filename) {

        u32 sz = event_count.get_val_at_idx(0);

        std::cout << "dumping to " << filename << " size = " << sz << std::endl;

        std::vector<TimelineEvent> events = this->events.copy_to_stdvec_idx_range(0, sz);

        u64 base_clock = get_base_clock_value();

        for (auto &t : events) {
            t.start -= base_clock;
            t.first_end -= base_clock;
            t.last_end -= base_clock;
        }

        std::ofstream file(filename);
        file << nlohmann::json(events).dump(4) << std::endl;
    }

} // namespace sham
