// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ScriptPane.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Script pane: demo run script, tabs, apply bar and editor.
 *
 */

#include "sham/gui/ScriptPane.hpp"
#include "sham/gui/style.hpp"
#include <cstring>
#include <string>

namespace sham::gui {

    namespace {
        constexpr const char *RUN_SCRIPT = R"(from hydro import Graph, nodes as n, data as d

g      = Graph("sedov_tracers")
mesh   = g.edge(d.Mesh(cells=(256, 256, 256)))
state  = g.edge(d.FluidState(fields=("rho", "v", "p")))
trc    = g.edge(d.Particles())
diag   = g.edge(d.Scalars(("E_tot", "dt", "cells_per_s")))

g.node(n.SedovInit(E0=1.0, n_tracers=1_200_000), out=[mesh, state, trc])
g.node(n.HydroStep(riemann="hllc", cfl=0.40), inp=[mesh], rw=[state], out=[diag])
g.node(n.AdvectTracers(scheme="rk2"), inp=[state], rw=[trc])

g.run(until=0.05)
)";
    } // namespace

    void ScriptPane::init() {
        editor.SetLanguage(TextEditor::Language::Python());
        editor.SetText(RUN_SCRIPT);
        editor.SetTabSize(4);
        editor.SetShowLineNumbersEnabled(true);
        editor.SetLineSpacing(1.15f);
        editor.SetShowWhitespacesEnabled(false);
        // Same built-in dark palette as the Python app; in C++ you can customise it with
        // editor.SetPalette(p) where p is a copy of TextEditor::GetDarkPalette().
    }

    void ScriptPane::draw(SDL *dl, double x, double y, double w, double h) {
        dl->AddRectFilled(V(x, y), V(x + w, y + 36), theme::PANEL);
        dl->AddLine(V(x, y + 35.5), V(x + w, y + 35.5), theme::DIVIDER);
        double tx = x;
        struct Tab {
            const char *label;
            bool active;
        };
        const Tab tabs[3] = {{"run_sedov.py", true}, {"Log", false}, {"Problems", false}};
        for (const Tab &t : tabs) {
            ImFont *font  = t.active ? g_fonts.mono : g_fonts.sans;
            bool problems = !std::strcmp(t.label, "Problems");
            double tw = 28 + text_w(font, 12, t.label) + (t.active ? 14 : 0) + (problems ? 22 : 0);
            if (t.active)
                dl->AddRectFilled(V(tx, y), V(tx + tw, y + 36), theme::CANVAS);
            std::string id = std::string("##tab_") + t.label;
            hit(id.c_str(), tx, y, tw, 36);
            draw_text_vc(
                dl, font, 12, tx + 14, y + 18, t.active ? theme::TEXT : theme::MUTED, t.label);
            double lw = text_w(font, 12, t.label);
            if (t.active)
                draw_live_dot(dl, tx + 14 + lw + 10, y + 18);
            if (problems) {
                double bx = tx + 14 + lw + 6;
                dl->AddRectFilled(V(bx, y + 10), V(bx + 16, y + 26), theme::ROW_HL, 8);
                draw_text_vc(dl, g_fonts.sans, 11, bx + 5, y + 18, theme::TEXT_2, "0");
            }
            dl->AddLine(V(tx + tw - 0.5, y), V(tx + tw - 0.5, y + 36), theme::DIVIDER);
            tx += tw;
        }
        double ay = y + 36;
        dl->AddRectFilled(V(x, ay), V(x + w, ay + 38), theme::CANVAS);
        dl->AddLine(V(x, ay + 37.5), V(x + w, ay + 37.5), rgba("#25272b"));
        double acy     = ay + 19;
        double apply_w = 20 + text_w(g_fonts.sans, 12, "Apply at next step"),
               dry_w   = 20 + text_w(g_fonts.sans, 12, "Dry run");
        double bx      = x + w - 12 - apply_w - 6 - dry_w;
        dl->AddCircleFilled(V(x + 15, acy), 3, theme::TEAL);
        std::string sync;
        for (const char *s : {"in sync with graph", "in sync", ""}) {
            sync = s;
            if (x + 24 + text_w(g_fonts.sans, 12, sync) + 12 <= bx)
                break;
        }
        draw_text_vc(dl, g_fonts.sans, 12, x + 24, acy, theme::TEAL_TEXT, sync);
        text_button(dl, "##dry", bx, acy, "Dry run");
        text_button(dl, "##apply", bx + dry_w + 6, acy, "Apply at next step", true);
        double ey = ay + 38;
        set_cursor(V(x, ey + 6));
        ImGui::PushFont(g_fonts.mono, S(13));
        editor.Render("##run_script", V(w * UI::scale, (h - 38 - 36 - 6) * UI::scale));
        ImGui::PopFont();
    }

} // namespace sham::gui
