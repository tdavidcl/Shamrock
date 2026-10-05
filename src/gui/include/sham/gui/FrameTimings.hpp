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
 * @file FrameTimings.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Per-frame CPU timings of shamrock_gui, printed as JSON by --bench.
 *
 */

#include <map>
#include <string>
#include <vector>

namespace sham::gui {

    /// Per-frame CPU timings for --bench, in seconds: "update" (data update), "ui" (building the
    /// UI) and "frame" (start of one frame to the start of the next).
    struct FrameTimings {
        std::map<std::string, std::vector<double>> timings{
            {"update", {}}, {"ui", {}}, {"frame", {}}};
        double prev_frame_start = -1; ///< start of the previous frame, -1 before the first one
        double t0               = 0;  ///< start of the current frame
        double t1               = 0;  ///< end of the data update of the current frame

        /// Wall clock in seconds, independent of GuiClock (virtual 60 fps clock in headless runs).
        static double wall();

        /// Mark the start of a frame; records the "frame" sample from the second frame on.
        void begin_frame();

        /// Mark the end of the data update; records the "update" sample.
        void mark_update();

        /// Mark the end of the UI build; records the "ui" sample.
        void mark_ui();

        /// Print the mean, median and p95 of each timing in ms, skipping the first `warmup`
        /// samples, as one line: BENCH {"impl": "cpp", "update": {...}, "ui": {...}, "frame":
        /// {...}}
        void print(int warmup) const;
    };

} // namespace sham::gui
