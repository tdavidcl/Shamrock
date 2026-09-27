// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GraphPane.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Graph pane: demo graph, canvas interaction and drawing.
 *
 */

#include "sham/gui/GraphPane.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace sham::gui {

    std::pair<std::vector<Node>, std::vector<Link>> build_demo_graph() {
        const ImU32 A = C::ACCENT, G = C::GRAY, B = C::BLUE, T = C::TEAL;
        auto param = [](const char *l, const char *v, bool hl = false) {
            return Row{RowKind::Param, l, v, 0, hl};
        };
        auto port = [](RowKind k, const char *l, ImU32 c) {
            return Row{k, l, "", c, false};
        };
        std::vector<Node> nodes;
        // start allow utf-8
        nodes.push_back(
            {"sedov",
             true,
             "Sedov init",
             20,
             150,
             160,
             C::INPUT,
             false,
             {param("E₀", "1.0"),
              param("n_tracers", "1.2M"),
              port(RowKind::Out, "mesh", G),
              port(RowKind::Out, "state", A),
              port(RowKind::Out, "tracers", B)}});
        // end allow utf-8
        nodes.push_back(
            {"hydro",
             true,
             "Hydro step",
             460,
             40,
             160,
             C::SOLVER,
             true,
             {port(RowKind::In, "mesh", G),
              port(RowKind::RW, "state", A),
              param("riemann", "hllc"),
              param("cfl", "0.40", true),
              port(RowKind::Out, "diagnostics", T)}});
        nodes.push_back(
            {"advect",
             true,
             "Advect tracers",
             460,
             300,
             160,
             C::SOLVER,
             true,
             {port(RowKind::In, "state", A),
              port(RowKind::RW, "tracers", B),
              param("scheme", "rk2")}});
        auto edge = [](const char *id,
                       double x,
                       double y,
                       ImU32 c,
                       const char *meta,
                       Preview p,
                       const char *footer) {
            Node n{id, false, id, x, y, 150};
            n.color   = c;
            n.meta    = meta;
            n.preview = p;
            n.footer  = footer;
            return n;
        };
        nodes.push_back(edge("mesh", 240, 20, G, "amr", Preview::None, ""));
        // start allow utf-8
        nodes.push_back(edge("state", 240, 130, A, "ρ v p", Preview::Slice, "ρ · z=0.5 · 128²"));
        nodes.push_back(edge("tracers", 240, 306, B, "1.2M", Preview::Tracers, "xy proj · 128²"));
        Node diag = edge("diag", 700, 60, T, "series", Preview::Series, "E_tot, dt, speed · 1 Hz");
        // end allow utf-8
        diag.title = "diagnostics";
        nodes.push_back(diag);
        std::vector<Link> links = {
            {"sedov", "mesh", "mesh", "in", G},
            {"sedov", "state", "state", "in", A},
            {"sedov", "tracers", "tracers", "in", B},
            {"mesh", "out", "hydro", "mesh", G},
            {"state", "out", "hydro", "state", A},
            {"state", "out", "advect", "state", A},
            {"tracers", "out", "advect", "tracers", B},
            {"hydro", "diagnostics", "diag", "in", T},
        };
        return {nodes, links};
    }

    Node *GraphView::by_id(const std::string &id) {
        for (auto &n : nodes)
            if (n.id == id)
                return &n;
        return nullptr;
    }

    std::array<double, 2> GraphView::to_screen(const double *origin, double wx, double wy) const {
        return {origin[0] + pan[0] + wx * zoom, origin[1] + pan[1] + wy * zoom};
    }

    std::array<double, 2> GraphView::port_world(const Node &n, const std::string &port) const {
        if (!n.compute)
            return {port == "in" ? n.x : n.x + n.w, n.y + 16};
        for (size_t i = 0; i < n.rows.size(); ++i) {
            const Row &r = n.rows[i];
            if (r.label == port && r.kind != RowKind::Param)
                return {r.kind == RowKind::Out ? n.x + n.w : n.x, n.y + 34 + 26.0 * i + 13};
        }
        return {n.x, n.y};
    }

    void GraphView::fit(double w, double h) {
        double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
        for (auto &n : nodes) {
            x0 = std::min(x0, n.x);
            y0 = std::min(y0, n.y);
            x1 = std::max(x1, n.x + n.w);
            y1 = std::max(y1, n.y + n.h());
        }
        x0 -= 20;
        y0 -= 20;
        x1 += 20;
        y1 += 36;
        zoom   = std::max(0.35, std::min({w / (x1 - x0), h / (y1 - y0), 1.3}));
        pan[0] = (w - (x1 - x0) * zoom) / 2 - x0 * zoom;
        pan[1] = std::max(4.0, (h - (y1 - y0) * zoom) / 2) - y0 * zoom;
    }

    void GraphView::draw_grid(SDL *dl, double x, double y, double w, double h) const {
        const double step = 20.0;
        auto pymod        = [](double a, double b) {
            double m = std::fmod(a, b);
            return m < 0 ? m + b : m;
        };
        const double ox = pymod(grid_offset[0], step), oy = pymod(grid_offset[1], step);
        for (double gx = x + ox; gx < x + w; gx += step)
            for (double gy = y + oy; gy < y + h; gy += step)
                dl->AddRectFilled(V(gx, gy), V(gx + 1.2, gy + 1.2), C::GRID_DOT);
    }

    void GraphView::interact(double x, double y, double w, double h, const double *origin) {
        set_cursor(V(x, y));
        invisible_button(
            "##graph_canvas",
            V(w, h),
            ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight
                | ImGuiButtonFlags_MouseButtonMiddle);
        bool hovered = ImGui::IsItemHovered(), active = ImGui::IsItemActive();
        ImGuiIO &io = ImGui::GetIO();
        ImVec2 m = mouse_pos(), md = mouse_delta();
        double mx = m.x, my = m.y;
        if (hovered && io.MouseWheel != 0.0f) {
            double old = zoom;
            zoom       = std::max(0.3, std::min(zoom * std::pow(1.12, double(io.MouseWheel)), 3.0));
            double wx = (mx - x - pan[0]) / old, wy = (my - y - pan[1]) / old;
            pan[0]    = mx - x - wx * zoom;
            pan[1]    = my - y - wy * zoom;
            user_view = true;
        }
        if (hovered && ImGui::IsMouseClicked(0)) {
            dragging.clear();
            selected.clear();
            for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
                auto s = to_screen(origin, it->x, it->y);
                if (s[0] <= mx && mx <= s[0] + it->w * zoom && s[1] <= my
                    && my <= s[1] + it->h() * zoom) {
                    selected = dragging = it->id;
                    break;
                }
            }
        }
        if (!dragging.empty() && ImGui::IsMouseDown(0)) {
            Node *n = by_id(dragging);
            n->x += md.x / zoom;
            n->y += md.y / zoom;
        }
        if (!ImGui::IsMouseDown(0))
            dragging.clear();
        if (active && (ImGui::IsMouseDragging(1) || ImGui::IsMouseDragging(2))) {
            pan[0] += md.x;
            pan[1] += md.y;
            grid_offset[0] += md.x;
            grid_offset[1] += md.y;
            user_view = true;
        }
        if (hovered && ImGui::IsMouseDoubleClicked(0) && selected.empty()) {
            user_view = false;
            fit(w, h);
        }
    }

    void GraphView::draw_link(SDL *dl, const double *origin, const Link &ln) {
        auto a   = port_world(*by_id(ln.src), ln.src_port);
        auto b   = port_world(*by_id(ln.dst), ln.dst_port);
        auto p1  = to_screen(origin, a[0], a[1]);
        auto p4  = to_screen(origin, b[0], b[1]);
        double k = std::max(std::abs(p4[0] - p1[0]) * 0.5, 30 * zoom);
        dl->AddBezierCubic(
            V(p1[0], p1[1]),
            V(p1[0] + k, p1[1]),
            V(p4[0] - k, p4[1]),
            V(p4[0], p4[1]),
            ln.color,
            2.0f);
    }

    void GraphView::port(SDL *dl, double cx, double cy, ImU32 color, bool diamond) const {
        double r_out = 5.0 * zoom, r_in = 3.2 * zoom;
        if (diamond) {
            double a = r_out * 1.25, b = r_in * 1.3;
            ImVec2 o[4] = {V(cx, cy - a), V(cx + a, cy), V(cx, cy + a), V(cx - a, cy)};
            ImVec2 i[4] = {V(cx, cy - b), V(cx + b, cy), V(cx, cy + b), V(cx - b, cy)};
            dl->AddConvexPolyFilled(o, 4, C::CANVAS);
            dl->AddConvexPolyFilled(i, 4, color);
        } else {
            dl->AddCircleFilled(V(cx, cy), float(r_out), C::CANVAS);
            dl->AddCircleFilled(V(cx, cy), float(r_in), color);
        }
    }

    void GraphView::draw_compute(SDL *dl, const double *origin, const Node &n) {
        const double z = zoom;
        auto p0        = to_screen(origin, n.x, n.y);
        double x0 = p0[0], y0 = p0[1], x1 = x0 + n.w * z, y1 = y0 + n.h() * z;
        dl->AddRectFilled(V(x0, y0), V(x1, y1), C::NODE, float(8 * z));
        dl->AddRectFilled(
            V(x0, y0), V(x1, y0 + 30 * z), n.style.bg, float(8 * z), ImDrawFlags_RoundCornersTop);
        dl->AddRect(
            V(x0, y0),
            V(x1, y1),
            selected == n.id ? C::ACCENT : C::NODE_BORDER,
            float(8 * z),
            0,
            1.0f);
        draw_text_vc(dl, g_fonts.medium, 13 * z, x0 + 12 * z, y0 + 15 * z, n.style.fg, n.title);
        if (n.gpu) {
            double cw = text_w(g_fonts.mono, 11 * z, "GPU") + 12 * z, cx = x1 - 12 * z - cw;
            dl->AddRectFilled(
                V(cx, y0 + 7 * z), V(cx + cw, y0 + 23 * z), n.style.chip, float(4 * z));
            draw_text_vc(dl, g_fonts.mono, 11 * z, cx + 6 * z, y0 + 15 * z, n.style.fg, "GPU");
        }
        for (size_t i = 0; i < n.rows.size(); ++i) {
            const Row &r = n.rows[i];
            double ry = y0 + (34 + 26.0 * i) * z, cy = ry + 13 * z;
            if (r.highlight)
                dl->AddRectFilled(V(x0 + 1, ry), V(x1 - 1, ry + 26 * z), C::ROW_HL);
            if (r.kind == RowKind::Param) {
                draw_text_vc(dl, g_fonts.sans, 12 * z, x0 + 12 * z, cy, C::ROW, r.label);
                double vw = text_w(g_fonts.mono, 12 * z, r.value);
                draw_text_vc(dl, g_fonts.mono, 12 * z, x1 - 12 * z - vw, cy, C::TEXT, r.value);
            } else if (r.kind == RowKind::Out) {
                double lw = text_w(g_fonts.sans, 12 * z, r.label);
                draw_text_vc(dl, g_fonts.sans, 12 * z, x1 - 12 * z - lw, cy, C::ROW, r.label);
                port(dl, x1, cy, r.color);
            } else {
                draw_text_vc(dl, g_fonts.sans, 12 * z, x0 + 12 * z, cy, C::ROW, r.label);
                port(dl, x0, cy, r.color, r.kind == RowKind::RW);
                if (r.kind == RowKind::RW) {
                    double tw = text_w(g_fonts.mono, 11 * z, "rw") + 10 * z, tx = x1 - 12 * z - tw;
                    dl->AddRectFilled(
                        V(tx, cy - 8 * z), V(tx + tw, cy + 8 * z), rgba("#2e3035"), float(3 * z));
                    draw_text_vc(dl, g_fonts.mono, 11 * z, tx + 5 * z, cy, C::TEXT_2, "rw");
                }
            }
        }
    }

    void GraphView::draw_legend(SDL *dl, double x, double y, double w, double h) const {
        const char *kinds[3]  = {"edge", "dot", "diamond"};
        const char *labels[3] = {"data edge", "read / write", "in-place (rw)"};
        ImFont *f             = g_fonts.sans;
        const double s        = 11.0;
        double widths[3], sum = 0;
        for (int i = 0; i < 3; ++i) {
            widths[i] = text_w(f, s, labels[i]) + 20;
            sum += widths[i];
        }
        double lw = sum + 16 * 2 + 20, lx = x + w - 12 - lw, ly = y + h - 10 - 26;
        dl->AddRectFilled(V(lx, ly), V(lx + lw, ly + 26), C::CANVAS, 6);
        dl->AddRect(V(lx, ly), V(lx + lw, ly + 26), rgba("#2a2c31"), 6);
        double cx = lx + 10, cy = ly + 13;
        for (int i = 0; i < 3; ++i) {
            if (!std::strcmp(kinds[i], "edge")) {
                dl->AddRectFilled(V(cx, cy - 5), V(cx + 14, cy + 5), C::CARD, 3);
                dl->AddRect(V(cx, cy - 5), V(cx + 14, cy + 5), rgba("#5a5c62"), 3);
            } else if (!std::strcmp(kinds[i], "dot")) {
                dl->AddCircleFilled(V(cx + 4, cy), 4, C::TEXT_3);
            } else {
                ImVec2 d[4] = {V(cx + 4, cy - 5), V(cx + 9, cy), V(cx + 4, cy + 5), V(cx - 1, cy)};
                dl->AddConvexPolyFilled(d, 4, C::TEXT_3);
            }
            draw_text_vc(dl, f, s, cx + 20, cy, C::MUTED, labels[i]);
            cx += widths[i] + 16;
        }
    }

    void GraphView::draw(double x, double y, double w, double h, const CardPreviews &cards) {
        SDL *dl                = window_draw_list();
        const double origin[2] = {x, y};
        if (!user_view && (std::abs(last_size[0] - w) > 0.5 || std::abs(last_size[1] - h) > 0.5))
            fit(w, h);
        last_size[0] = w;
        last_size[1] = h;
        dl->AddRectFilled(V(x, y), V(x + w, y + h), C::CANVAS);
        draw_grid(dl, x, y, w, h);
        interact(x, y, w, h, origin);
        dl->PushClipRect(V(x, y), V(x + w, y + h), true);
        for (auto &ln : links)
            draw_link(dl, origin, ln);
        for (auto &n : nodes) {
            if (n.compute)
                draw_compute(dl, origin, n);
            else
                draw_edge(dl, origin, n, cards);
        }
        dl->PopClipRect();
        draw_legend(dl, x, y, w, h);
    }

    void GraphView::draw_edge(
        SDL *dl, const double *origin, const Node &n, const CardPreviews &cards) {
        const double z = zoom;
        auto p0        = to_screen(origin, n.x, n.y);
        double x0 = p0[0], y0 = p0[1], x1 = x0 + n.w * z, y1 = y0 + n.h() * z;
        dl->AddRectFilled(V(x0, y0), V(x1, y1), C::CARD, float(10 * z));
        dl->AddRect(
            V(x0, y0),
            V(x1, y1),
            selected == n.id ? C::ACCENT : C::NODE_BORDER,
            float(10 * z),
            0,
            1.0f);
        dl->AddLine(V(x0 + 1, y0 + 30 * z), V(x1 - 1, y0 + 30 * z), rgba("#26282d"));
        double hc = y0 + 15 * z;
        dl->AddRectFilled(
            V(x0 + 12 * z, hc - 4 * z), V(x0 + 20 * z, hc + 4 * z), n.color, float(2 * z));
        draw_text_vc(dl, g_fonts.medium, 13 * z, x0 + 27 * z, hc, C::TEXT, n.title);
        double mw = text_w(g_fonts.mono, 11 * z, n.meta);
        draw_text_vc(dl, g_fonts.mono, 11 * z, x1 - 10 * z - mw, hc, C::MUTED, n.meta);
        port(dl, x0, y0 + 16 * z, n.color);
        port(dl, x1, y0 + 16 * z, n.color);

        double bx0 = x0 + 7 * z, by0 = y0 + 37 * z;
        if (n.preview == Preview::None) {
            // start allow utf-8
            draw_text(
                dl, g_fonts.mono, 11 * z, x0 + 12 * z, y0 + 38 * z, C::TEXT_3, "256³ · 8 patches");
            // end allow utf-8
            draw_text(dl, g_fonts.mono, 11 * z, x0 + 12 * z, y0 + 54 * z, C::DIM, "no preview");
            return;
        }
        double img_h = n.preview == Preview::Series ? 56 : 88;
        double bx1 = bx0 + 136 * z, by1 = by0 + img_h * z;
        if (!previews_on) {
            dl->AddRectFilled(V(bx0, by0), V(bx1, by1), C::DARK, float(4 * z));
            draw_text_vc(
                dl, g_fonts.mono, 11 * z, bx0 + 8 * z, (by0 + by1) / 2, C::DIM, "preview paused");
        } else if (n.preview == Preview::Slice) {
            dl->AddImageRounded(
                cards.state,
                V(bx0, by0),
                V(bx1, by1),
                ImVec2(0, 0),
                ImVec2(1, 1),
                rgba("#ffffff"),
                float(4 * z));
        } else if (n.preview == Preview::Tracers) {
            dl->AddImageRounded(
                cards.tracers,
                V(bx0, by0),
                V(bx1, by1),
                ImVec2(0, 0),
                ImVec2(1, 1),
                rgba("#ffffff"),
                float(4 * z));
        } else {
            dl->AddRectFilled(V(bx0, by0), V(bx1, by1), C::DARK, float(4 * z));
            sparkline(dl, bx0, by0 + 3 * z, bx1, by1 - 3 * z, cards.dt, C::ACCENT, 1.5);
            sparkline(dl, bx0, by0 + 3 * z, bx1, by1 - 3 * z, cards.e_tot, C::TEAL, 1.2, 0.35);
        }
        double fy = by1 + 11 * z;
        draw_text_vc(dl, g_fonts.mono, 11 * z, bx0, fy, C::TEXT_3, n.footer);
        if (previews_on)
            draw_live_dot(dl, bx1 - 3 * z, fy - 1 * z, 3 * z);
    }

    GraphPane::GraphPane() : view(build_demo_graph().first, build_demo_graph().second) {}

    void GraphPane::create_textures() {
        tex_card_state.create(136, 88);
        tex_card_tracers.create(136, 88);
    }

    double GraphPane::refresh(const DemoSimulation &sim, double t, bool force) {
        if (!force && t < next_cards) // edge cards: 4 Hz
            return 0;
        if (view.previews_on) {
            tex_card_state.upload(sim.slice(0, 136, 88).px);
            tex_card_tracers.upload(sim.tracers_image(136, 88).px);
        }
        next_cards = t + 0.25;
        return 2 * 136 * 88 * 4;
    }

    void GraphPane::draw(
        SDL *dl, double x, double y, double w, double h, const DemoSimulation &sim) {
        double cy = pane_header(dl, x, y, w);
        int n_c   = 0;
        for (auto &n : view.nodes)
            n_c += n.compute;
        int n_e = int(view.nodes.size()) - n_c;
        // start allow utf-8
        std::string counts = std::to_string(n_c) + " nodes · " + std::to_string(n_e) + " edges · "
                             + std::to_string(view.links.size()) + " links";
        double counts_x    = x + 12;
        std::string prev_label = view.previews_on ? "Previews · 4 Hz" : "Previews off";
        // end allow utf-8
        double widths[3]
            = {20 + 12 + text_w(g_fonts.sans, 12, prev_label),
               20 + text_w(g_fonts.sans, 12, "Auto-layout"),
               20 + text_w(g_fonts.sans, 12, "Fit")};
        double bx        = x + w - 12 - (widths[0] + widths[1] + widths[2]) - 6 * 2;
        std::string zoom = std::to_string(int(std::nearbyint(view.zoom * 100))) + "%";
        double zx        = bx - 8 - text_w(g_fonts.mono, 11, zoom);
        draw_text_vc(dl, g_fonts.mono, 11, zx, cy, C::MUTED, zoom);
        if (counts_x + text_w(g_fonts.mono, 12, counts) + 12 <= zx)
            draw_text_vc(dl, g_fonts.mono, 12, counts_x, cy, C::MUTED, counts);
        Btn b = text_button(dl, "##fit", bx, cy, "Fit");
        if (b.clicked) {
            view.user_view = false;
            view.fit(w, h - PANE_HDR);
        }
        bx += b.w + 6;
        b = text_button(dl, "##autolayout", bx, cy, "Auto-layout");
        if (b.clicked) {
            auto fresh = build_demo_graph().first;
            for (auto &n : view.nodes)
                for (auto &f : fresh)
                    if (f.id == n.id) {
                        n.x = f.x;
                        n.y = f.y;
                    }
            view.user_view = false;
            view.fit(w, h - PANE_HDR);
        }
        bx += b.w + 6;
        if (text_button(dl, "##previews", bx, cy, prev_label, view.previews_on, view.previews_on)
                .clicked)
            view.previews_on = !view.previews_on;
        const CardPreviews cards{
            tex_card_state.ref,
            tex_card_tracers.ref,
            sim.history.at("dt"),
            sim.history.at("E_tot")};
        view.draw(x, y + PANE_HDR, w, h - PANE_HDR, cards);
    }

} // namespace sham::gui
