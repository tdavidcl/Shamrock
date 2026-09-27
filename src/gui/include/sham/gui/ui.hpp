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
 * @brief UI scale layer and small widgets: logical-pixel drawing (SDL), text measurement, buttons
 * and plots.
 *
 */

#include "imgui.h"
#include "sham/gui/font.hpp"
#include "sham/gui/style.hpp"
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <string>
#include <vector>

namespace sham::gui {

    inline Fonts g_fonts; // set by load_fonts() in main()
    inline constexpr double PI = 3.14159265358979323846;

    // ============================================================================
    //  Small drawing helpers
    // ============================================================================
    inline ImVec2 V(double x, double y) { return ImVec2(float(x), float(y)); }

    // Global UI scale. The app is laid out in *logical* pixels (the mockup's 1x sizes); these
    // helpers convert to physical pixels when drawing, placing widgets and reading the mouse.
    struct UI {
        static inline double scale = 1.0;
        static inline ImVec2 origin{0, 0}; // main viewport position (scaling is done around it)
        static constexpr double MIN = 0.5, MAX = 3.0, STEP = 0.1;
    };
    inline ImVec2 P(ImVec2 p) {
        const float s = float(UI::scale);
        return ImVec2(
            UI::origin.x + (p.x - UI::origin.x) * s, UI::origin.y + (p.y - UI::origin.y) * s);
    }
    inline float S(double v) { return float(v * UI::scale); }
    inline ImVec2 mouse_pos() {
        ImVec2 m = ImGui::GetIO().MousePos;
        return V(
            UI::origin.x + (m.x - UI::origin.x) / UI::scale,
            UI::origin.y + (m.y - UI::origin.y) / UI::scale);
    }
    inline ImVec2 mouse_delta() {
        ImVec2 d = ImGui::GetIO().MouseDelta;
        return V(d.x / UI::scale, d.y / UI::scale);
    }
    inline void set_cursor(ImVec2 p) { ImGui::SetCursorScreenPos(P(p)); }
    inline bool invisible_button(const char *id, ImVec2 size, ImGuiButtonFlags flags = 0) {
        return ImGui::InvisibleButton(
            id, V(std::max(size.x * UI::scale, 1.0), std::max(size.y * UI::scale, 1.0)), flags);
    }

    // Same calls as ImDrawList, but taking logical coordinates and sizes.
    struct SDL {
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
    inline SDL g_sdl[16];
    inline int g_sdl_next = 0;
    inline SDL *window_draw_list() { // one wrapper per window drawn in the frame (main window +
                                     // panes)
        SDL *s = &g_sdl[g_sdl_next++ % 16];
        s->d   = ImGui::GetWindowDrawList();
        return s;
    }

    struct TextKey {
        const ImFont *font;
        int size100;
        std::string s;
        int scale100;
        bool operator==(const TextKey &o) const {
            return font == o.font && size100 == o.size100 && s == o.s && scale100 == o.scale100;
        }
    };
    struct TextKeyHash {
        size_t operator()(const TextKey &k) const {
            return std::hash<std::string>()(k.s) ^ (std::hash<const void *>()(k.font) << 1)
                   ^ size_t(k.size100) * 31;
        }
    };
    inline std::unordered_map<TextKey, double, TextKeyHash> g_text_cache;

    inline double text_w(ImFont *font, double size, const std::string &s) {
        TextKey key{font, int(std::lround(size * 100)), s, int(std::lround(UI::scale * 100))};
        auto it = g_text_cache.find(key);
        if (it != g_text_cache.end())
            return it->second;
        ImGui::PushFont(font, float(size * UI::scale));
        double w = ImGui::CalcTextSize(s.c_str()).x / UI::scale;
        ImGui::PopFont();
        if (g_text_cache.size() > 4000)
            g_text_cache.clear();
        g_text_cache.emplace(std::move(key), w);
        return w;
    }

    inline void draw_text(
        SDL *dl, ImFont *font, double size, double x, double y, ImU32 col, const std::string &s) {
        dl->AddText(font, float(size), V(x, y), col, s.c_str());
    }
    inline void draw_text_vc(
        SDL *dl, ImFont *font, double size, double x, double cy, ImU32 col, const std::string &s) {
        dl->AddText(font, float(size), V(x, cy - size * 0.66), col, s.c_str());
    }

    inline void tooltip(const char *text) {
        ImGui::PushFont(g_fonts.sans, S(13));
        ImGui::SetItemTooltip("%s", text);
        ImGui::PopFont();
    }

    struct Hit {
        bool clicked, hovered;
    };
    // Optional observer of every clickable area (physical pixels); used by tools such as the demo
    // recorder.
    inline void (*g_on_hit)(const char *id, ImVec2 min, ImVec2 max) = nullptr;
    inline Hit hit(const char *id, double x, double y, double w, double h) {
        set_cursor(V(x, y));
        bool clicked = invisible_button(id, V(std::max(w, 1.0), std::max(h, 1.0)));
        bool hovered = ImGui::IsItemHovered();
        if (g_on_hit)
            g_on_hit(id, P(V(x, y)), P(V(x + w, y + h)));
        if (hovered)
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        return {clicked, hovered};
    }

    inline ImU32 lighten(ImU32 col, int amount = 14) {
        int r = col & 255, g = (col >> 8) & 255, b = (col >> 16) & 255, a = (col >> 24) & 255;
        return rgb_u32(
            std::min(r + amount, 255), std::min(g + amount, 255), std::min(b + amount, 255), a);
    }

    inline bool framed_button(
        SDL *dl,
        const char *id,
        double x,
        double y,
        double w,
        double h,
        ImU32 bg,
        const ImU32 *border,
        double radius = 5.0) {
        Hit r = hit(id, x, y, w, h);
        dl->AddRectFilled(V(x, y), V(x + w, y + h), r.hovered ? lighten(bg) : bg, float(radius));
        if (border)
            dl->AddRect(V(x + 0.5, y + 0.5), V(x + w - 0.5, y + h - 0.5), *border, float(radius));
        return r.clicked;
    }
    inline bool framed_button(
        SDL *dl,
        const char *id,
        double x,
        double y,
        double w,
        double h,
        ImU32 bg,
        ImU32 border,
        double radius = 5.0) {
        return framed_button(dl, id, x, y, w, h, bg, &border, radius);
    }

    struct Btn {
        bool clicked;
        double w;
    };
    inline Btn text_button(
        SDL *dl,
        const char *id,
        double x,
        double cy,
        const std::string &label,
        bool warm = false,
        bool dot  = false,
        double h  = 26.0) {
        ImFont *font      = g_fonts.sans;
        const double size = 12.0, pad = 10.0, dot_w = dot ? 12.0 : 0.0;
        double w = pad * 2 + dot_w + text_w(font, size, label);
        double y = cy - h / 2;
        ImU32 bg = warm ? C::WARM_BG : C::BUTTON, border = warm ? C::WARM_BORDER : C::BORDER,
              fg     = warm ? C::WARM_TEXT : C::TEXT_2;
        bool clicked = framed_button(dl, id, x, y, w, h, bg, border, 5.0);
        double tx    = x + pad;
        if (dot) {
            dl->AddCircleFilled(V(tx + 3, cy), 3.0f, C::ACCENT);
            tx += dot_w;
        }
        draw_text_vc(dl, font, size, tx, cy, fg, label);
        return {clicked, w};
    }

    inline void dashed_line(
        SDL *dl,
        double ax,
        double ay,
        double bx,
        double by,
        ImU32 col,
        double dash      = 4.0,
        double gap       = 4.0,
        double thickness = 1.0) {
        double length = std::hypot(bx - ax, by - ay);
        if (length < 1e-3)
            return;
        double ux = (bx - ax) / length, uy = (by - ay) / length, t = 0.0;
        while (t < length) {
            double t2 = std::min(t + dash, length);
            dl->AddLine(
                V(ax + ux * t, ay + uy * t), V(ax + ux * t2, ay + uy * t2), col, float(thickness));
            t += dash + gap;
        }
    }
    inline void draw_live_dot(SDL *dl, double cx, double cy, double r = 3.0) {
        dl->AddCircleFilled(V(cx, cy), float(r), C::ACCENT);
    }

    inline void sparkline(
        SDL *dl,
        double x0,
        double y0,
        double x1,
        double y1,
        const std::deque<double> &values,
        ImU32 col,
        double thickness = 1.6,
        double pad_frac  = 0.12) {
        const size_t n = values.size();
        if (n < 2)
            return;
        auto [mn, mx] = std::minmax_element(values.begin(), values.end());
        double lo = *mn, hi = *mx, span = hi > lo ? hi - lo : 1.0;
        lo -= span * pad_frac;
        hi += span * pad_frac;
        std::vector<ImVec2> pts(n);
        const double step = (x1 - x0) / double(n - 1);
        for (size_t i = 0; i < n; ++i) {
            double x = i == n - 1 ? x1 : x0 + double(i) * step;
            pts[i]   = V(x, y1 - (values[i] - lo) / (hi - lo) * (y1 - y0));
        }
        dl->AddPolyline(pts.data(), int(n), col, 0, float(thickness));
    }

    // Background and bottom divider of a pane header; returns its vertical centre.
    inline double pane_header(SDL *dl, double x, double y, double w) {
        dl->AddRectFilled(V(x, y), V(x + w, y + PANE_HDR), C::PANEL);
        dl->AddLine(V(x, y + PANE_HDR - 0.5), V(x + w, y + PANE_HDR - 0.5), C::DIVIDER);
        return y + PANE_HDR / 2;
    }

} // namespace sham::gui
