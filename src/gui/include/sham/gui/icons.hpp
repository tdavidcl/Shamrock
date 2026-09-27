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
 * @file icons.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Vector icons of the top bar and pane headers, drawn in logical pixels.
 *
 */

#include "sham/gui/style.hpp"
#include "sham/gui/ui.hpp"
#include <cmath>

namespace sham::gui {

    // ============================================================================
    //  Icons
    // ============================================================================
    inline void icon_play(SDL *dl, double cx, double cy, ImU32 col, double s = 7.0) {
        dl->AddTriangleFilled(V(cx - s * 0.6, cy - s), V(cx + s, cy), V(cx - s * 0.6, cy + s), col);
    }
    inline void icon_pause(SDL *dl, double cx, double cy, ImU32 col) {
        dl->AddLine(V(cx - 3.5, cy - 6), V(cx - 3.5, cy + 6), col, 2.0f);
        dl->AddLine(V(cx + 3.5, cy - 6), V(cx + 3.5, cy + 6), col, 2.0f);
    }
    inline void icon_step(SDL *dl, double cx, double cy, ImU32 col) {
        dl->AddTriangle(V(cx - 5, cy - 6), V(cx + 3, cy), V(cx - 5, cy + 6), col, 1.8f);
        dl->AddLine(V(cx + 5.5, cy - 6), V(cx + 5.5, cy + 6), col, 2.0f);
    }
    inline void icon_stop(SDL *dl, double cx, double cy, ImU32 col) {
        dl->AddRect(V(cx - 5, cy - 5), V(cx + 5, cy + 5), col, 1.0f, 0, 1.8f);
    }
    inline void icon_viewer(SDL *dl, double cx, double cy, ImU32 col) {
        dl->AddRect(V(cx - 7, cy - 6), V(cx + 7, cy + 6), col, 2.0f, 0, 1.5f);
        dl->AddCircle(V(cx, cy), 3.2f, col, 0, 1.5f);
    }
    inline void icon_graph(SDL *dl, double cx, double cy, ImU32 col) {
        dl->AddRect(V(cx - 8, cy - 6), V(cx - 2.5, cy - 1.5), col, 1.2f, 0, 1.4f);
        dl->AddRect(V(cx + 2.5, cy + 1.5), V(cx + 8, cy + 6), col, 1.2f, 0, 1.4f);
        dl->AddBezierCubic(
            V(cx - 2.5, cy - 3.8),
            V(cx + 1, cy - 3.8),
            V(cx - 1, cy + 3.8),
            V(cx + 2.5, cy + 3.8),
            col,
            1.4f);
    }
    inline void polyline3(SDL *dl, ImVec2 a, ImVec2 b, ImVec2 c, ImU32 col, float t) {
        ImVec2 p[3] = {a, b, c};
        dl->AddPolyline(p, 3, col, 0, t);
    }
    inline void icon_flame(
        SDL *dl, double cx, double cy, ImU32 col) { // three stacked bars: a flamegraph
        dl->AddRectFilled(V(cx - 8, cy + 3), V(cx + 8, cy + 6.5), col, 1.0f);
        dl->AddRectFilled(V(cx - 8, cy - 1.5), V(cx + 3, cy + 2), col, 1.0f);
        dl->AddRectFilled(V(cx - 8, cy - 6), V(cx - 2, cy - 2.5), col, 1.0f);
    }
    inline void icon_code(SDL *dl, double cx, double cy, ImU32 col) {
        polyline3(dl, V(cx - 3.5, cy - 5), V(cx - 8, cy), V(cx - 3.5, cy + 5), col, 1.5f);
        polyline3(dl, V(cx + 3.5, cy - 5), V(cx + 8, cy), V(cx + 3.5, cy + 5), col, 1.5f);
        dl->AddLine(V(cx + 1.5, cy - 6), V(cx - 1.5, cy + 6), col, 1.5f);
    }
    inline void icon_layout(SDL *dl, double cx, double cy, ImU32 col, int lay) {
        const double x0 = cx - 9, y0 = cy - 7, x1 = cx + 9, y1 = cy + 7;
        auto L = [&](double ax, double ay, double bx, double by) {
            dl->AddLine(V(ax, ay), V(bx, by), col, 1.3f);
        };
        if (lay == 0) { // stack: one pane in front of the others
            dl->AddRect(V(x0 + 4, y0), V(x1, y1 - 4), col, 1.5f, 0, 1.1f);
            dl->AddRectFilled(V(x0, y0 + 4), V(x1 - 4, y1), C::BUTTON, 1.5f);
            dl->AddRect(V(x0, y0 + 4), V(x1 - 4, y1), col, 1.5f, 0, 1.3f);
            return;
        }
        dl->AddRect(V(x0, y0), V(x1, y1), col, 1.5f, 0, 1.3f);
        switch (lay) {
        case 1:
            L(x0 + 10, y0, x0 + 10, y1);
            L(x0 + 10, cy, x1, cy);
            break; // tall
        case 2:
            L(x0, y0 + 8, x1, y0 + 8);
            L(cx, y0 + 8, cx, y1);
            break; // fat
        case 3:
            L(cx, y0, cx, y1);
            L(x0, cy, x1, cy);
            break; // grid
        case 4:
            L(x0 + 6, y0, x0 + 6, y1);
            L(x0 + 12, y0, x0 + 12, y1);
            break; // horizontal
        case 5:
            L(x0, y0 + 4.7, x1, y0 + 4.7);
            L(x0, y0 + 9.3, x1, y0 + 9.3);
            break; // vertical
        default:
            L(x0 + 7, y0, x0 + 7, y1);
            L(x0 + 7, cy - 1, x1, cy - 1);
            L(x0 + 12.5, cy - 1, x0 + 12.5, y1); // splits
        }
    }
    inline void icon_chevron(SDL *dl, double cx, double cy, ImU32 col) {
        polyline3(dl, V(cx - 3.5, cy - 1.5), V(cx, cy + 2), V(cx + 3.5, cy - 1.5), col, 1.6f);
    }
    inline void icon_gear(SDL *dl, double cx, double cy, ImU32 col) {
        dl->AddCircle(V(cx, cy), 3.2f, col, 0, 1.5f);
        for (int k = 0; k < 8; ++k) {
            double a = k * PI / 4;
            dl->AddLine(
                V(cx + std::cos(a) * 5.5, cy + std::sin(a) * 5.5),
                V(cx + std::cos(a) * 8, cy + std::sin(a) * 8),
                col,
                1.5f);
        }
    }

} // namespace sham::gui
