// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file EventList.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/EventList.hpp"
#include "shamcomm/logs.hpp"

void sham::EventList::add_events(std::vector<sycl::event> &e) {
    events.insert(events.end(), e.begin(), e.end());
    consumed = false;
}

void sham::EventList::add_events(sham::EventList &e) {
    events.insert(events.end(), e.events.begin(), e.events.end());
    consumed   = false;
    e.consumed = true;
}

sham::EventList::~EventList() noexcept(false) {
    if (!consumed && !events.empty()) {
        std::string log_str = sham::format(
            "EventList destroyed without being consumed :\n    -> creation : {}",
            loc_build.format_one_line());

        for (auto &e : events) {
            e.wait();
        }
        throw shambase::make_except_with_loc<std::runtime_error>(log_str);
    }
}
