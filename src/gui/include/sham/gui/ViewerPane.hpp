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
 * @file ViewerPane.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Viewer pane: live view of the selected field, colorbar, diagnostics plots and preview
 * subscriptions.
 *
 */

#include "sham/gui/DemoSimulation.hpp"
#include "sham/gui/GLTexture.hpp"
#include "sham/gui/ui.hpp"
#include <array>
#include <functional>
#include <vector>

namespace sham::gui {

    // What the "Preview subscriptions" table reports about the other panes.
    struct SubscriptionSummary {
        bool cards_on;      // graph pane: edge-card previews enabled
        bool profile_on;    // profile pane shown and live
        double bytes_per_s; // estimated preview traffic
    };

    // Main view (3D half-cut cube or 2D slice, tracers, probe) refreshed at 10 Hz, with the field
    // chips, 3D / Slice switch, colorbar, diagnostics sparklines and subscription list.
    struct ViewerPane {
        int field         = 0;
        bool show_tracers = true, view3d = true;
        double yaw = -38 * PI / 180, pitch = 24 * PI / 180;
        GLTexture tex_main;
        double main_lo = 0.02, main_hi = 4.0;
        std::vector<std::array<double, 2>> tracer_pts;
        double next_refresh = 0;
        // Called right after the user picks another field (the app then refreshes every preview).
        std::function<void()> on_field_change;

        void create_textures();
        // Uploads the main slice when due; returns the bytes a remote link would have sent.
        double refresh(const DemoSimulation &sim, double t, bool force);
        void draw(
            SDL *dl,
            double x,
            double y,
            double w,
            double h,
            const DemoSimulation &sim,
            const SubscriptionSummary &subs);

        private:
        void view_image(SDL *dl, double x, double y, double s, const DemoSimulation &sim);
        void draw_cube(SDL *dl, double x, double y, double s, const DemoSimulation &sim) const;
        void colorbar(SDL *dl, double x, double y, double w) const;
        double plots(SDL *dl, double x, double y, double w, const DemoSimulation &sim) const;
        void subscriptions(
            SDL *dl, double x, double y, double w, const SubscriptionSummary &subs) const;
    };

} // namespace sham::gui
