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
 * @file ScaledDrawList.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief ImDrawList wrapper taking logical pixels (see ui.hpp), and text drawing through it.
 *
 */

#include "imgui.h"
#include "sham/gui/ui.hpp"
#include <string>
#include <vector>

namespace sham::gui {

    /// Same calls as ImDrawList, but taking logical coordinates and sizes.
    struct ScaledDrawList {
        ImDrawList *d = nullptr;
        void AddRectFilled(ImVec2 a, ImVec2 b, ImU32 c, float r = 0, ImDrawFlags f = 0) {
            d->AddRectFilled(P(a), P(b), c, S(r), f);
        }
        void AddRect(ImVec2 a, ImVec2 b, ImU32 c, float r = 0, ImDrawFlags f = 0, float th = 1) {
            d->AddRect(P(a), P(b), c, S(r), f, S(th));
        }
        void AddRectFilledMultiColor(ImVec2 a, ImVec2 b, ImU32 c1, ImU32 c2, ImU32 c3, ImU32 c4) {
            d->AddRectFilledMultiColor(P(a), P(b), c1, c2, c3, c4);
        }
        void AddLine(ImVec2 a, ImVec2 b, ImU32 c, float th = 1) {
            d->AddLine(P(a), P(b), c, S(th));
        }
        void AddCircle(ImVec2 c, float r, ImU32 col, int seg = 0, float th = 1) {
            d->AddCircle(P(c), S(r), col, seg, S(th));
        }
        void AddCircleFilled(ImVec2 c, float r, ImU32 col, int seg = 0) {
            d->AddCircleFilled(P(c), S(r), col, seg);
        }
        void AddTriangle(ImVec2 a, ImVec2 b, ImVec2 c, ImU32 col, float th = 1) {
            d->AddTriangle(P(a), P(b), P(c), col, S(th));
        }
        void AddTriangleFilled(ImVec2 a, ImVec2 b, ImVec2 c, ImU32 col) {
            d->AddTriangleFilled(P(a), P(b), P(c), col);
        }
        void AddConvexPolyFilled(const ImVec2 *pts, int n, ImU32 col) {
            auto q = scaled(pts, n);
            d->AddConvexPolyFilled(q.data(), n, col);
        }
        void AddPolyline(const ImVec2 *pts, int n, ImU32 col, ImDrawFlags f, float th) {
            auto q = scaled(pts, n);
            d->AddPolyline(q.data(), n, col, f, S(th));
        }
        void AddBezierCubic(
            ImVec2 a, ImVec2 b, ImVec2 c, ImVec2 e, ImU32 col, float th, int seg = 0) {
            d->AddBezierCubic(P(a), P(b), P(c), P(e), col, S(th), seg);
        }
        void AddText(ImFont *f, float size, ImVec2 pos, ImU32 col, const char *s) {
            d->AddText(f, S(size), P(pos), col, s);
        }
        void AddImage(ImTextureRef t, ImVec2 a, ImVec2 b) { d->AddImage(t, P(a), P(b)); }
        void AddImageRounded(
            ImTextureRef t, ImVec2 a, ImVec2 b, ImVec2 uv0, ImVec2 uv1, ImU32 col, float r) {
            d->AddImageRounded(t, P(a), P(b), uv0, uv1, col, S(r));
        }
        void AddImageQuad(ImTextureRef t, ImVec2 a, ImVec2 b, ImVec2 c, ImVec2 e) {
            d->AddImageQuad(t, P(a), P(b), P(c), P(e));
        }
        void PushClipRect(ImVec2 a, ImVec2 b, bool intersect = false) {
            d->PushClipRect(P(a), P(b), intersect);
        }
        void PopClipRect() { d->PopClipRect(); }

        private:
        static std::vector<ImVec2> scaled(const ImVec2 *pts, int n) {
            std::vector<ImVec2> q(n);
            for (int i = 0; i < n; ++i)
                q[i] = P(pts[i]);
            return q;
        }
    };

    // ============================================================================
    //  Free functions
    // ============================================================================

    /// Wrapper of the current window's draw list.
    inline ScaledDrawList window_draw_list() { return {ImGui::GetWindowDrawList()}; }

    /// Draw s with its top-left corner at (x, y) (logical pixels).
    inline void draw_text(
        ScaledDrawList &dl,
        ImFont *font,
        double size,
        double x,
        double y,
        ImU32 col,
        const std::string &s) {
        dl.AddText(font, float(size), V(x, y), col, s.c_str());
    }

    /// Draw s starting at x, vertically centred on cy (logical pixels).
    inline void draw_text_vc(
        ScaledDrawList &dl,
        ImFont *font,
        double size,
        double x,
        double cy,
        ImU32 col,
        const std::string &s) {
        dl.AddText(font, float(size), V(x, cy - size * 0.66), col, s.c_str());
    }

} // namespace sham::gui
