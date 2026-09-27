// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ProfilePane.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Profile pane: flamegraph layout, colours and interaction.
 *
 */

#include "sham/gui/ProfilePane.hpp"
#include "sham/gui/format.hpp"
#include "sham/gui/style.hpp"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>

namespace sham::gui {

    namespace {
        ImU32 cat_color(DemoSimulation::Cat c, const char *name) {
            uint32_t hsh = 2166136261u; // small per-scope shade variation, stable across frames
            for (const char *p = name; *p; ++p)
                hsh = (hsh ^ uint8_t(*p)) * 16777619u;
            double k   = 0.88 + 0.24 * double(hsh % 1000) / 1000.0;
            auto shade = [&](int r, int g, int b) {
                return rgb_u32(
                    std::min(255, int(r * k)),
                    std::min(255, int(g * k)),
                    std::min(255, int(b * k)));
            };
            switch (c) {
            case DemoSimulation::Cat::Gpu : return shade(0xd9, 0x8f, 0x3a);
            case DemoSimulation::Cat::Mpi : return shade(0x6f, 0x9b, 0xe8);
            case DemoSimulation::Cat::Host: return shade(0x9a, 0x96, 0x8e);
            case DemoSimulation::Cat::Io  : return shade(0x4f, 0xb3, 0xa9);
            default                       : return rgba("#3a3c42");
            }
        }
    } // namespace

    void ProfilePane::refresh(const DemoSimulation &sim, double t, bool force, bool shown) {
        if (shown && live && (force || t >= next_refresh || snap.empty())) { // 2 Hz
            snap.resize(sim.prof.size());
            for (size_t i = 0; i < sim.prof.size(); ++i)
                snap[i] = sim.prof[i].ema;
            next_refresh = t + 0.5;
        }
    }

    void ProfilePane::draw(
        SDL *dl, double x, double y, double w, double h, const DemoSimulation &sim) {
        const auto &prof = sim.prof;
        if (snap.size() != prof.size()) {
            snap.resize(prof.size());
            for (size_t i = 0; i < prof.size(); ++i)
                snap[i] = prof[i].ema;
        }
        double cy = pane_header(dl, x, y, w);
        // start allow utf-8
        std::string info = "rank 0 · running mean · " + fmt("%.2f", snap[0]) + " ms / step";
        // right: Live toggle, Reset zoom when zoomed
        std::string live_label = live ? "Live · 2 Hz" : "Paused";
        // end allow utf-8
        double live_w = 20 + (live ? 12 : 0) + text_w(g_fonts.sans, 12, live_label);
        double bx     = x + w - 12 - live_w;
        if (focus != 0) {
            double rw = 20 + text_w(g_fonts.sans, 12, "Reset zoom");
            if (text_button(dl, "##prof_reset", bx - 6 - rw, cy, "Reset zoom").clicked)
                focus = 0;
        }
        if (text_button(dl, "##prof_live", bx, cy, live_label, live, live).clicked)
            live = !live;
        double info_max
            = bx - 12 - (focus != 0 ? 20 + text_w(g_fonts.sans, 12, "Reset zoom") + 6 : 0);
        if (x + 12 + text_w(g_fonts.mono, 12, info) <= info_max)
            draw_text_vc(dl, g_fonts.mono, 12, x + 12, cy, C::MUTED, info);

        // frames: ancestors of the focused scope (dimmed, full width), the focus, then its
        // subtree
        const double fx = x + 12, fw = w - 24, gap = 1;
        double top = y + PANE_HDR + 10;
        std::vector<int> chain;
        for (int i = focus; i >= 0; i = prof[i].parent)
            chain.insert(chain.begin(), i);
        std::function<int(int)> sub_depth = [&](int i) {
            int d = 1;
            for (size_t c = 0; c < prof.size(); ++c)
                if (prof[c].parent == i && snap[c] > 1e-4)
                    d = std::max(d, 1 + sub_depth(int(c)));
            return d;
        };
        const int rows = int(chain.size()) - 1 + sub_depth(focus);
        // rows shrink (down to 14 px) before anything is cut; the legend goes first when space
        // is short
        double avail     = y + h - top - 8;
        bool show_legend = avail - 26 >= rows * (16 + gap);
        if (show_legend)
            avail -= 26;
        const double row_h  = std::clamp(avail / rows - gap, 14.0, 22.0);
        const double bottom = top + avail;
        int depth           = 0;
        auto frame_box      = [&](int i, double bx0, double bw, int d, bool dim) {
            double by0 = top + d * (row_h + gap);
            if (by0 + row_h > bottom + 0.5 || bw < 1.0)
                return;
            ImU32 fill     = dim ? rgba("#2a2b30") : cat_color(prof[i].cat, prof[i].name);
            std::string id = std::string("##prof_") + std::to_string(i) + (dim ? "a" : "");
            Hit hv         = hit(id.c_str(), bx0, by0, std::max(bw - 1, 1.0), row_h);
            dl->AddRectFilled(
                V(bx0, by0),
                V(bx0 + std::max(bw - 1, 1.0), by0 + row_h),
                hv.hovered ? lighten(fill, 22) : fill,
                2);
            if (hv.hovered)
                dl->AddRect(
                    V(bx0, by0), V(bx0 + std::max(bw - 1, 1.0), by0 + row_h), C::TEXT, 2, 0, 1.0f);
            bool dark_text    = !dim && prof[i].cat != DemoSimulation::Cat::Root;
            ImU32 tc          = dark_text ? rgba("#16140f") : C::TEXT_2;
            std::string ms    = fmt("%.2f ms", snap[i]);
            std::string label = prof[i].name;
            double tw = text_w(g_fonts.mono, 11, label), mw = text_w(g_fonts.mono, 11, ms);
            if (bw > tw + mw + 24 && row_h >= 16) {
                draw_text_vc(dl, g_fonts.mono, 11, bx0 + 6, by0 + row_h / 2, tc, label);
                draw_text_vc(dl, g_fonts.mono, 11, bx0 + bw - 7 - mw, by0 + row_h / 2, tc, ms);
            } else if (bw > tw + 12) {
                draw_text_vc(dl, g_fonts.mono, 11, bx0 + 6, by0 + row_h / 2, tc, label);
            } else if (bw > 28) {
                // start allow utf-8
                std::string cut
                    = label.substr(0, std::max<size_t>(1, size_t((bw - 16) / 6.6))) + "…";
                // end allow utf-8
                draw_text_vc(dl, g_fonts.mono, 11, bx0 + 5, by0 + row_h / 2, tc, cut);
            }
            if (hv.hovered) {
                double step     = std::max(snap[0], 1e-9);
                double par      = prof[i].parent >= 0 ? std::max(snap[prof[i].parent], 1e-9) : step;
                const char *cat = prof[i].cat == DemoSimulation::Cat::Gpu   ? "GPU kernel"
                                  : prof[i].cat == DemoSimulation::Cat::Mpi ? "MPI / communication"
                                  : prof[i].cat == DemoSimulation::Cat::Io  ? "I/O"
                                                                            : "host";
                // start allow utf-8
                tooltip((label + "\n" + fmt("%.3f ms", snap[i]) + "   "
                         + fmt("%.1f", 100 * snap[i] / step) + "% of step   "
                         + fmt("%.1f", 100 * snap[i] / par) + "% of parent\n" + cat
                         + (prof[i].leaf ? "" : "   ·   click to zoom"))
                            .c_str());
                // end allow utf-8
            }
            if (hv.clicked && (dim || !prof[i].leaf))
                focus = i; // zoom in, or back out via an ancestor
        };
        for (size_t k = 0; k + 1 < chain.size(); ++k)
            frame_box(chain[k], fx, fw, depth++, true);
        std::function<void(int, double, double, int)> draw_sub
            = [&](int i, double bx0, double bw, int d) {
                  frame_box(i, bx0, bw, d, false);
                  double total = std::max(snap[i], 1e-9), cx = bx0;
                  for (size_t c = 0; c < prof.size(); ++c) {
                      if (prof[c].parent != i || snap[c] <= 1e-4)
                          continue;
                      double cw = bw * snap[c] / total;
                      draw_sub(int(c), cx, cw, d + 1);
                      cx += cw;
                  }
              };
        draw_sub(focus, fx, fw, depth);

        // legend
        struct L {
            DemoSimulation::Cat c;
            const char *t;
        };
        const L legend[4]
            = {{DemoSimulation::Cat::Gpu, "GPU kernel"},
               {DemoSimulation::Cat::Mpi, "MPI / comm"},
               {DemoSimulation::Cat::Host, "host"},
               {DemoSimulation::Cat::Io, "I/O"}};
        if (!show_legend)
            return;
        double lx = x + 12, ly = y + h - 16;
        for (const L &l : legend) {
            dl->AddRectFilled(V(lx, ly - 5), V(lx + 10, ly + 5), cat_color(l.c, ""), 2);
            draw_text_vc(dl, g_fonts.sans, 11, lx + 16, ly, C::MUTED, l.t);
            lx += 16 + text_w(g_fonts.sans, 11, l.t) + 16;
        }
        std::string hint = "click a frame to zoom";
        double hw        = text_w(g_fonts.sans, 11, hint);
        if (lx + 20 + hw < x + w - 12)
            draw_text_vc(dl, g_fonts.sans, 11, x + w - 12 - hw, ly, C::DIM, hint);
    }

} // namespace sham::gui
