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
 * @file ui.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief UI scale state and logical -> physical pixel conversion. Drawing in logical pixels goes
 * through ScaledDrawList.hpp.
 *
 */

#include "imgui.h"

namespace sham::gui {

    inline ImVec2 V(double x, double y) { return ImVec2(float(x), float(y)); }

    /// Global UI scale. The app is laid out in *logical* pixels (the mockup's 1x sizes); these
    /// helpers convert to physical pixels when drawing, placing widgets and reading the mouse.
    struct UI {
        /// current scale (physical pixels per logical pixel)
        static inline double scale = 1.0;

        /// main viewport position (scaling is done around it)
        static inline ImVec2 origin{0, 0};
    };

    /// logical point -> physical point
    inline ImVec2 P(ImVec2 p) {
        const float s = float(UI::scale);
        return ImVec2(
            UI::origin.x + (p.x - UI::origin.x) * s, UI::origin.y + (p.y - UI::origin.y) * s);
    }

    /// logical length -> physical length
    inline float S(double v) { return float(v * UI::scale); }

} // namespace sham::gui
