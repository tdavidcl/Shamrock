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
 * @file ProfilePane.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Profile pane: live flamegraph of the solver step on rank 0.
 *
 */

#include "sham/gui/DemoSimulation.hpp"
#include "sham/gui/ui.hpp"
#include <vector>

namespace sham::gui {

    // Hover a scope for its time and share, click to zoom, click a dimmed parent (or Reset zoom)
    // to go back out; Live pauses the snapshot, which otherwise follows the running mean at 2 Hz.
    struct ProfilePane {
        bool live = true;         // follow the solver (false: paused)
        int focus = 0;            // index of the zoomed-in scope
        std::vector<double> snap; // per-scope times shown (ms)
        double next_refresh = 0;

        void refresh(const DemoSimulation &sim, double t, bool force, bool shown);
        void draw(SDL *dl, double x, double y, double w, double h, const DemoSimulation &sim);
    };

} // namespace sham::gui
