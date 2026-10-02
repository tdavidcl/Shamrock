// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ViewerPane.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Viewer pane: main view, 3D cube projection, colorbar, plots and subscriptions.
 *
 */

#include "sham/gui/ViewerPane.hpp"
#include "sham/gui/format.hpp"
#include "sham/gui/style.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <string>

namespace sham::gui {

    namespace {
        // Toggle chip of the pane header (field selection, tracers).
        Btn chip(
            SDL *dl,
            const char *id,
            double x,
            double cy,
            const std::string &label,
            bool on,
            double h = 26.0) {
            double w = 20 + text_w(g_fonts.mono, 12, label);
            Hit r    = hit(id, x, cy - h / 2, w, h);
            ImU32 bg = on ? C::ACCENT_BG : (r.hovered ? lighten(C::BUTTON) : C::BUTTON);
            dl->AddRectFilled(V(x, cy - h / 2), V(x + w, cy + h / 2), bg, 5);
            dl->AddRect(
                V(x + 0.5, cy - h / 2 + 0.5),
                V(x + w - 0.5, cy + h / 2 - 0.5),
                on ? C::ACCENT : C::BORDER,
                5);
            draw_text_vc(dl, g_fonts.mono, 12, x + 10, cy, on ? C::ACCENT_TEXT : C::TEXT_3, label);
            return {r.clicked, w};
        }
    } // namespace

    void ViewerPane::create_textures() { tex_main.create(512, 512); }

    double ViewerPane::refresh(const DemoSimulation &sim, double t, bool force) {
        if (!force && t < next_refresh) // main view: 10 Hz
            return 0;
        Image img = sim.slice(field, 512, 512, &main_lo, &main_hi);
        tex_main.upload(img.px);
        tracer_pts   = sim.tracers_xy(520);
        next_refresh = t + 0.1;
        return 512 * 512 * 10;
    }

    void ViewerPane::draw(
        SDL *dl,
        double x,
        double y,
        double w,
        double h,
        const DemoSimulation &sim,
        const SubscriptionSummary &subs) {
        double cy = pane_header(dl, x, y, w);
        double widths[3], total = 0;
        for (int k = 0; k < 3; ++k) {
            widths[k] = 20 + text_w(g_fonts.mono, 12, FIELDS[k].label);
            total += widths[k];
        }
        double tr_w = 20 + text_w(g_fonts.mono, 12, "tracers");
        total += 6 * 2 + 6 + 1 + 6 + tr_w;
        double cx = x + w - 12 - total;
        for (int k = 0; k < 3; ++k) {
            std::string id = std::string("##field_") + FIELDS[k].id;
            Btn b          = chip(dl, id.c_str(), cx, cy, FIELDS[k].label, field == k);
            if (b.clicked && field != k) {
                field = k;
                if (on_field_change)
                    on_field_change();
            }
            cx += widths[k] + 6;
        }
        dl->AddLine(V(cx, cy - 9), V(cx, cy + 9), C::BORDER);
        cx += 7;
        if (chip(dl, "##tracers", cx, cy, "tracers", show_tracers).clicked)
            show_tracers = !show_tracers;

        double bx = x + 16, by = y + PANE_HDR + 16, bw = w - 32, bh = h - PANE_HDR - 32;
        bool horizontal = w / std::max(h, 1.0) > 1.25;
        double img      = horizontal ? std::min(bh - 30, w * 0.55) : std::min(bw, bh - 300);
        img             = std::max(140.0, std::floor(img));
        view_image(dl, bx, by, img, sim);
        colorbar(dl, bx, by + img + 10, img);
        double sx, sy, sw;
        if (horizontal) {
            sx = bx + img + 24;
            sy = by;
            sw = bw - img - 24;
        } else {
            sx = bx;
            sy = by + img + 10 + 14 + 18;
            sw = bw;
        }
        if (sw > 120) {
            sy = plots(dl, sx, sy, sw, sim);
            subscriptions(dl, sx, sy + 18, sw, subs);
        }
    }

    void ViewerPane::view_image(SDL *dl, double x, double y, double s, const DemoSimulation &sim) {
        dl->AddRectFilled(V(x, y), V(x + s, y + s), rgba("#0c0d10"), 4);
        dl->PushClipRect(V(x, y), V(x + s, y + s), true);
        if (view3d) {
            draw_cube(dl, x, y, s, sim);
        } else {
            dl->AddImageRounded(
                tex_main.ref,
                V(x, y),
                V(x + s, y + s),
                ImVec2(0, 0),
                ImVec2(1, 1),
                rgba("#ffffff"),
                4);
            if (show_tracers)
                for (auto &p : tracer_pts) {
                    double qx = x + (p[0] + 0.5) * s, qy = y + (0.5 - p[1]) * s;
                    dl->AddRectFilled(V(qx, qy), V(qx + 2, qy + 2), rgba("#ffffff", 0.55));
                }
            dl->AddCircle(V(x + 0.648 * s, y + 0.585 * s), 4.5f, rgba("#ffffff"), 0, 1.5f);
        }

        const char *seg[2] = {"3D", "Slice"};
        double seg_w[2]
            = {20 + text_w(g_fonts.mono, 11, seg[0]), 20 + text_w(g_fonts.mono, 11, seg[1])};
        double tw = seg_w[0] + seg_w[1] + 2 * 3;
        // start allow utf-8
        std::string label = std::string("state.") + FIELDS[field].label + " · "
                            + (view3d ? "cut z=0.5" : "z=0.5") + " · 10 Hz";
        // end allow utf-8
        if (8 + 20 + text_w(g_fonts.mono, 11, label) + 8 + 12 + tw + 8 > s)
            label = std::string("state.") + FIELDS[field].label;
        double bw_ = 8 + 6 + 6 + text_w(g_fonts.mono, 11, label) + 8;
        dl->AddRectFilled(V(x + 8, y + 8), V(x + 8 + bw_, y + 28), C::CARD, 4);
        draw_live_dot(dl, x + 8 + 11, y + 18);
        draw_text_vc(dl, g_fonts.mono, 11, x + 8 + 20, y + 18, C::TEXT, label);

        double sx0 = x + s - 8 - tw;
        dl->AddRectFilled(V(sx0, y + 8), V(sx0 + tw, y + 36), C::CARD, 6);
        dl->AddRect(V(sx0 + 0.5, y + 8.5), V(sx0 + tw - 0.5, y + 35.5), C::BORDER, 6);
        double bx = sx0 + 2;
        for (int i = 0; i < 2; ++i) {
            bool is3d = i == 0, on = view3d == is3d;
            std::string id = std::string("##seg_") + seg[i];
            Hit h          = hit(id.c_str(), bx, y + 10, seg_w[i], 24);
            if (on)
                dl->AddRectFilled(V(bx, y + 10), V(bx + seg_w[i], y + 34), C::ACCENT_BG, 4);
            draw_text_vc(dl, g_fonts.mono, 11, bx + 10, y + 22, on ? C::ACCENT : C::MUTED, seg[i]);
            if (h.clicked)
                view3d = is3d;
            bx += seg_w[i] + 2;
        }

        double val = sim.sample(field, 0.648 - 0.5, 0.5 - 0.585);
        std::string probe
            = std::string(FIELDS[field].label) + " = " + fmt("%.3g", val) + "  (0.65, 0.41)";
        double pw = 16 + text_w(g_fonts.mono, 11, probe);
        dl->AddRectFilled(V(x + s - 8 - pw, y + s - 30), V(x + s - 8, y + s - 8), C::CARD, 4);
        dl->AddRect(V(x + s - 8 - pw, y + s - 30), V(x + s - 8, y + s - 8), C::NODE_BORDER, 4);
        draw_text_vc(dl, g_fonts.mono, 11, x + s - pw, y + s - 19, C::TEXT, probe);

        // orbit with left drag (3D), double-click resets the camera. Submitted after the
        // overlay buttons: with overlapping items the first one submitted takes the hover, so
        // the 3D / Slice switch must come first.
        set_cursor(V(x, y));
        invisible_button("##view_orbit", V(s, s));
        if (view3d && ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
            ImVec2 d = mouse_delta();
            yaw += d.x * 0.008;
            pitch = std::max(-1.35, std::min(1.35, pitch + d.y * 0.008));
        }
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
            yaw   = -38 * PI / 180;
            pitch = 24 * PI / 180;
        }
        dl->PopClipRect();
    }

    void ViewerPane::draw_cube(
        SDL *dl, double x, double y, double s, const DemoSimulation &sim) const {
        const double cyw = std::cos(yaw), syw = std::sin(yaw), cp = std::cos(pitch),
                     sp    = std::sin(pitch);
        const double scale = s * 0.56, ox = x + s / 2, oy = y + s * 0.51;
        using P3 = std::array<double, 3>;
        auto rot = [&](P3 p) {
            double px = p[0] * cyw + p[2] * syw, pz = -p[0] * syw + p[2] * cyw, py = p[1];
            return P3{px, py * cp - pz * sp, py * sp + pz * cp};
        };
        auto proj = [&](P3 p) {
            P3 r = rot(p);
            return V(ox + r[0] * scale, oy - r[1] * scale);
        };
        const double hh = 0.5;
        double ambient  = std::clamp(
            DemoSimulation::normalize<double>(
                field, sim.sample(field, 0.49, 0.49), main_lo, main_hi),
            0.0,
            1.0);
        P3 light{0.35, 0.8, 0.5};
        double ln = std::sqrt(light[0] * light[0] + light[1] * light[1] + light[2] * light[2]);
        for (double &c : light)
            c /= ln;
        struct Face {
            P3 n;
            std::array<P3, 4> c;
            bool textured;
        };
        const Face faces[6] = {
            {{0, 0, 1}, {{{-hh, hh, 0}, {hh, hh, 0}, {hh, -hh, 0}, {-hh, -hh, 0}}}, true},
            {{0, 0, -1}, {{{hh, hh, -hh}, {-hh, hh, -hh}, {-hh, -hh, -hh}, {hh, -hh, -hh}}}, false},
            {{1, 0, 0}, {{{hh, hh, 0}, {hh, hh, -hh}, {hh, -hh, -hh}, {hh, -hh, 0}}}, false},
            {{-1, 0, 0}, {{{-hh, hh, -hh}, {-hh, hh, 0}, {-hh, -hh, 0}, {-hh, -hh, -hh}}}, false},
            {{0, 1, 0}, {{{-hh, hh, -hh}, {hh, hh, -hh}, {hh, hh, 0}, {-hh, hh, 0}}}, false},
            {{0, -1, 0}, {{{-hh, -hh, 0}, {hh, -hh, 0}, {hh, -hh, -hh}, {-hh, -hh, -hh}}}, false},
        };
        for (const Face &f : faces) {
            P3 rn = rot(f.n);
            if (rn[2] <= 0)
                continue; // back-facing: convex box needs no sorting
            ImVec2 pts[4] = {proj(f.c[0]), proj(f.c[1]), proj(f.c[2]), proj(f.c[3])};
            if (f.textured) {
                dl->AddImageQuad(tex_main.ref, pts[0], pts[1], pts[2], pts[3]);
                dl->AddPolyline(pts, 4, rgba("#ffffff", 0.45), ImDrawFlags_Closed, 1.0f);
            } else {
                double shade
                    = 0.62
                      + 0.5 * std::max(0.0, rn[0] * light[0] + rn[1] * light[1] + rn[2] * light[2]);
                dl->AddConvexPolyFilled(pts, 4, lut_color(VIRIDIS, ambient, shade));
                dl->AddPolyline(pts, 4, rgba("#ffffff", 0.3), ImDrawFlags_Closed, 1.0f);
            }
        }
        if (rot({0, 0, 1})[2] > 0) {
            if (show_tracers) {
                ImU32 col = rgba("#ffffff", 0.55);
                for (auto &p : tracer_pts) {
                    ImVec2 q = proj({p[0], p[1], 0.0});
                    dl->AddRectFilled(V(q.x - 1.0, q.y - 1.0), V(q.x + 1.0, q.y + 1.0), col);
                }
            }
            dl->AddCircle(proj({0.148, -0.085, 0.0}), 4.5f, rgba("#ffffff"), 0, 1.5f);
        }
        ImU32 ghost = rgba("#ffffff", 0.24);
        P3 front[4] = {{-hh, hh, hh}, {hh, hh, hh}, {hh, -hh, hh}, {-hh, -hh, hh}};
        for (int i = 0; i < 4; ++i) {
            ImVec2 a = proj(front[i]), b = proj(front[(i + 1) % 4]),
                   c = proj({front[i][0], front[i][1], 0.0});
            dashed_line(dl, a.x, a.y, b.x, b.y, ghost);
            dashed_line(dl, a.x, a.y, c.x, c.y, ghost);
        }
        double gx = x + 30, gy = y + s - 22;
        struct Ax {
            P3 d;
            ImU32 col;
            const char *name;
        };
        const Ax axes[3]
            = {{{1, 0, 0}, rgba("#e8866a"), "x"},
               {{0, 1, 0}, rgba("#8cc084"), "y"},
               {{0, 0, 1}, rgba("#6f9be8"), "z"}};
        for (const Ax &a : axes) {
            P3 r      = rot(a.d);
            double ex = gx + r[0] * 14, ey = gy - r[1] * 14;
            dl->AddLine(V(gx, gy), V(ex, ey), a.col, 1.6f);
            draw_text_vc(dl, g_fonts.mono, 11, ex + r[0] * 5 - 3, ey - r[1] * 5, a.col, a.name);
        }
    }

    void ViewerPane::colorbar(SDL *dl, double x, double y, double w) const {
        auto f = [&](double v) {
            return FIELDS[field].log ? fmt("%.2g", v) : strip_zeros(fmt("%.2f", v));
        };
        std::string lo_s = f(main_lo), hi_s = f(main_hi), name = FIELDS[field].cb_label;
        double cy = y + 7;
        draw_text_vc(dl, g_fonts.mono, 11, x, cy, C::TEXT_3, lo_s);
        double bx0 = x + text_w(g_fonts.mono, 11, lo_s) + 10;
        double bx1
            = x + w - text_w(g_fonts.mono, 11, hi_s) - 10 - text_w(g_fonts.mono, 11, name) - 10;
        const int n = 24;
        for (int i = 0; i < n; ++i) {
            double a = bx0 + (bx1 - bx0) * i / n, b = bx0 + (bx1 - bx0) * (i + 1) / n;
            ImU32 ca = viridis_u32(double(i) / n), cb = viridis_u32(double(i + 1) / n);
            dl->AddRectFilledMultiColor(V(a, cy - 4), V(b + 0.5, cy + 4), ca, cb, cb, ca);
        }
        draw_text_vc(dl, g_fonts.mono, 11, bx1 + 10, cy, C::TEXT_3, hi_s);
        draw_text_vc(
            dl,
            g_fonts.mono,
            11,
            bx1 + 10 + text_w(g_fonts.mono, 11, hi_s) + 10,
            cy,
            C::TEXT,
            name);
    }

    double ViewerPane::plots(
        SDL *dl, double x, double y, double w, const DemoSimulation &sim) const {
        auto &hist = sim.history;
        struct Item {
            const char *title;
            std::string value;
            const std::deque<double> *s;
            ImU32 col;
            bool ref;
        };
        const Item items[3] = {
            {"Total energy",
             fmt("%.7f", hist.at("E_tot").back()),
             &hist.at("E_tot"),
             C::TEAL,
             true},
            {"Timestep dt", fmt("%.2e", hist.at("dt").back()), &hist.at("dt"), C::ACCENT, false},
            {"Speed",
             fmt("%.1f", hist.at("speed").back() / 1e9) + " Gcell/s",
             &hist.at("speed"),
             C::BLUE,
             false},
        };
        const double gap = 10.0, cw = (w - gap * 2) / 3;
        for (int i = 0; i < 3; ++i) {
            const Item &it = items[i];
            double cx      = x + i * (cw + gap);
            draw_text(dl, g_fonts.sans, 12, cx, y, C::TEXT_2, it.title);
            draw_text(dl, g_fonts.mono, 12, cx, y + 18, C::TEXT, it.value);
            double by0 = y + 38, by1 = y + 38 + 44;
            dl->AddRectFilled(V(cx, by0), V(cx + cw, by1), C::CANVAS, 4);
            for (double f : {0.25, 0.5, 0.75}) {
                double gy = by0 + (by1 - by0) * f;
                dl->AddLine(V(cx, gy), V(cx + cw, gy), rgba("#26282d"));
            }
            if (it.ref) {
                auto [mn, mx] = std::minmax_element(it.s->begin(), it.s->end());
                double span   = (*mx - *mn) != 0 ? *mx - *mn : 1.0;
                double ry     = by1 - (1.0 - (*mn - span * 0.12)) / (span * 1.24) * (by1 - by0);
                if (by0 < ry && ry < by1)
                    dashed_line(dl, cx, ry, cx + cw, ry, rgba("#4a4c52"), 3, 3);
            }
            sparkline(dl, cx + 1, by0 + 3, cx + cw - 1, by1 - 3, *it.s, it.col, 1.6);
        }
        return y + 82;
    }

    void ViewerPane::subscriptions(
        SDL *dl, double x, double y, double w, const SubscriptionSummary &subs) const {
        draw_text(dl, g_fonts.sans, 12, x, y, C::TEXT_2, "Preview subscriptions");
        std::string rate = fmt("%.1f", subs.bytes_per_s / 1e6) + " MB/s";
        draw_text(
            dl, g_fonts.mono, 11, x + w - text_w(g_fonts.mono, 11, rate), y + 1, C::MUTED, rate);
        bool on = subs.cards_on;
        struct R {
            std::string c[4];
            bool active;
        };
        const bool pf = subs.profile_on;
        // start allow utf-8
        const R rows[6] = {
            {{std::string("state.") + FIELDS[field].label, "main", "512²", "10 Hz"}, true},
            {{"state.ρ", "card", "128²", on ? "4 Hz" : "off"}, on},
            {{"tracers", "card", "128²", on ? "4 Hz" : "off"}, on},
            {{"diagnostics", "card", "series", on ? "1 Hz" : "off"}, on},
            {{"profile", "pane", "scopes", pf ? "2 Hz" : "off"}, pf},
            {{"mesh", "card", "—", "off"}, false},
        };
        // end allow utf-8
        const double col_x[4] = {0.0, 0.4, 0.6, 0.8}; // first column is the widest (edge names)
        double ry             = y + 22;
        for (const R &r : rows) {
            for (int i = 0; i < 4; ++i) {
                ImU32 col = r.active ? (i == 0 ? C::TEXT : C::TEXT_3) : C::DIM;
                draw_text(dl, g_fonts.mono, 11, x + col_x[i] * w, ry, col, r.c[i]);
            }
            ry += 17;
        }
    }

} // namespace sham::gui
