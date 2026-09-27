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
 * @file GraphPane.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Graph pane: the simulation graph drawn as a node editor (compute nodes, data edges with
 * live previews, links).
 *
 */

#include "imgui.h"
#include "sham/gui/DemoSimulation.hpp"
#include "sham/gui/GLTexture.hpp"
#include "sham/gui/style.hpp"
#include "sham/gui/ui.hpp"
#include <array>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace sham::gui {

    // ============================================================================
    //  Graph model + in-house canvas (compute nodes, data edges, links)
    // ============================================================================
    enum class RowKind { Param, In, RW, Out };
    struct Row {
        RowKind kind;
        std::string label, value;
        ImU32 color    = 0;
        bool highlight = false;
    };
    enum class Preview { None, Slice, Tracers, Series };

    struct Node {
        std::string id;
        bool compute;
        std::string title;
        double x, y, w;
        C::HeaderStyle style = C::SOLVER;
        bool gpu             = false;
        std::vector<Row> rows;
        ImU32 color = 0;
        std::string meta;
        Preview preview = Preview::None;
        std::string footer;
        double h() const {
            if (compute)
                return 30 + 8 + 26.0 * rows.size();
            switch (preview) {
            case Preview::Slice  :
            case Preview::Tracers: return 153;
            case Preview::Series : return 121;
            default              : return 72;
            }
        }
    };
    struct Link {
        std::string src, src_port, dst, dst_port;
        ImU32 color;
    };

    // Node positions and links of the demo run script.
    std::pair<std::vector<Node>, std::vector<Link>> build_demo_graph();

    // What the data-edge cards show: the two preview textures and the diagnostics history.
    struct CardPreviews {
        ImTextureRef state, tracers;
        const std::deque<double> &dt, &e_tot;
    };

    // In-house canvas: pan, zoom around the cursor, selection, node dragging and fit-to-view.
    class GraphView {
        public:
        std::vector<Node> nodes;
        std::vector<Link> links;
        double zoom = 1.0, pan[2] = {0, 0}, grid_offset[2] = {0, 0};
        bool user_view       = false;
        double last_size[2]  = {0, 0};
        std::string selected = "state", dragging;
        bool previews_on     = true;

        GraphView(std::vector<Node> n, std::vector<Link> l)
            : nodes(std::move(n)), links(std::move(l)) {}

        Node *by_id(const std::string &id);
        std::array<double, 2> to_screen(const double *origin, double wx, double wy) const;
        std::array<double, 2> port_world(const Node &n, const std::string &port) const;
        void fit(double w, double h);
        void draw(double x, double y, double w, double h, const CardPreviews &cards);

        private:
        void draw_grid(SDL *dl, double x, double y, double w, double h) const;
        void interact(double x, double y, double w, double h, const double *origin);
        void draw_link(SDL *dl, const double *origin, const Link &ln);
        void port(SDL *dl, double cx, double cy, ImU32 color, bool diamond = false) const;
        void draw_compute(SDL *dl, const double *origin, const Node &n);
        void draw_edge(SDL *dl, const double *origin, const Node &n, const CardPreviews &cards);
        void draw_legend(SDL *dl, double x, double y, double w, double h) const;
    };

    // Header bar (counts, zoom, Fit / Auto-layout / Previews) above the canvas; owns the edge-card
    // preview textures, refreshed at 4 Hz.
    struct GraphPane {
        GraphView view;
        GLTexture tex_card_state, tex_card_tracers;
        double next_cards = 0;

        GraphPane();
        void create_textures();
        // Uploads the card previews when due; returns the bytes a remote link would have sent.
        double refresh(const DemoSimulation &sim, double t, bool force);
        void draw(SDL *dl, double x, double y, double w, double h, const DemoSimulation &sim);
    };

} // namespace sham::gui
