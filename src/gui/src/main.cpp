// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file main.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Shamrock control GUI: for now a Dear ImGui (docking branch) frame loop showing an empty
 * dock area that fills the window, with a headless test mode (deterministic 60 fps clock for
 * --screenshot / --bench).
 *
 * Interactive runs remember the dock arrangement in shamrock_gui_layout.ini.
 *
 * Usage:
 *
 *     ./shamrock_gui                        interactive
 *     ./shamrock_gui --screenshot shot.png  render 45 frames (or --frames N), save PNG, exit
 *     ./shamrock_gui --bench 300            print per-frame CPU timings as JSON
 *
 */

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#if defined(__APPLE__)
    #include <OpenGL/gl3.h>
#else
    #include <GL/gl.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace sham::gui {

    struct App {
        bool deterministic;
        long long frame = 0;
        int frames_left, bench_frames;
        bool screenshot, want_exit = false;
        std::map<std::string, std::vector<double>> timings{
            {"update", {}}, {"ui", {}}, {"frame", {}}};
        double prev_frame_start = -1;

        App(bool screenshot_, int frames, int bench)
            : deterministic(screenshot_ || bench > 0), frames_left(frames), bench_frames(bench),
              screenshot(screenshot_) {}
        // single instance owned by main, the panes will keep callbacks into it
        App(const App &)            = delete;
        App &operator=(const App &) = delete;

        static double wall() {
            using namespace std::chrono;
            return duration<double>(steady_clock::now().time_since_epoch()).count();
        }
        double now() const { return deterministic ? double(frame) / 60.0 : wall(); }

        /// Build one frame: a full-screen host window holding the dock area.
        void gui() {
            double t0 = wall();
            // update phase (empty for now)
            double t1 = wall();

            const ImGuiViewport *vp = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(vp->Pos);
            ImGui::SetNextWindowSize(vp->Size);
            ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove
                                     | ImGuiWindowFlags_NoSavedSettings
                                     | ImGuiWindowFlags_NoBringToFrontOnFocus
                                     | ImGuiWindowFlags_NoScrollWithMouse;
            // no padding or border, so the dock area covers the whole window
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::Begin("##shamrock_main", nullptr, flags);
            ImGui::PopStyleVar(2);
            ImGui::DockSpace(ImGui::GetID("##body_dockspace"), ImVec2(0, 0));
            ImGui::End();

            double t2 = wall();
            if (bench_frames) {
                timings["update"].push_back(t1 - t0);
                timings["ui"].push_back(t2 - t1);
                if (prev_frame_start >= 0)
                    timings["frame"].push_back(t0 - prev_frame_start);
                prev_frame_start = t0;
            }
            frame += 1;
            if (screenshot || bench_frames)
                if (--frames_left <= 0)
                    want_exit = true;
        }
    };

    static void print_bench(const App &app, int warmup) {
        std::printf("BENCH {\"impl\": \"cpp\"");
        for (const char *k : {"update", "ui", "frame"}) {
            std::vector<double> a(
                app.timings.at(k).begin() + std::min<size_t>(warmup, app.timings.at(k).size()),
                app.timings.at(k).end());
            for (double &v : a)
                v *= 1e3;
            std::sort(a.begin(), a.end());
            double mean = a.empty() ? 0 : std::accumulate(a.begin(), a.end(), 0.0) / a.size();
            auto pct    = [&](double p) { // numpy's default (linear) percentile
                if (a.empty())
                    return 0.0;
                double idx = p / 100.0 * (a.size() - 1);
                size_t lo = size_t(idx), hi = std::min(lo + 1, a.size() - 1);
                return a[lo] + (a[hi] - a[lo]) * (idx - lo);
            };
            std::printf(
                ", \"%s\": {\"mean_ms\": %.3f, \"median_ms\": %.3f, \"p95_ms\": %.3f}",
                k,
                mean,
                pct(50),
                pct(95));
        }
        std::printf("}\n");
    }

    /// Command-line options of shamrock_gui.
    struct CliArgs {
        std::string screenshot; ///< --screenshot: PNG to save, empty for an interactive run
        int frames = 45;        ///< frames rendered before exiting (--frames, or --bench + 30)
        int bench  = 0;         ///< --bench: timed frames, 0 when not benchmarking
        std::optional<int> exit_code; ///< set when main must return right away (usage printed)
    };

    /// Parse argv; prints the usage and sets exit_code on -h / --help or an unknown option.
    static CliArgs parse_cli(int argc, char **argv) {
        CliArgs cli;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next     = [&]() {
                return i + 1 < argc ? std::string(argv[++i]) : std::string();
            };
            if (a == "--screenshot")
                cli.screenshot = next();
            else if (a == "--frames")
                cli.frames = std::stoi(next());
            else if (a == "--bench")
                cli.bench = std::stoi(next());
            else {
                std::printf("usage: %s [--screenshot out.png] [--frames N] [--bench N]\n", argv[0]);
                cli.exit_code = a == "-h" || a == "--help" ? 0 : 1;
                return cli;
            }
        }
        if (cli.bench)
            cli.frames = cli.bench + 30;
        return cli;
    }

} // namespace sham::gui

int main(int argc, char **argv) {
    using namespace sham::gui;
    const CliArgs cli = parse_cli(argc, argv);
    if (cli.exit_code)
        return *cli.exit_code;

    if (!glfwInit()) {
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    GLFWwindow *window = glfwCreateWindow(1440, 960, "Shamrock", nullptr, nullptr);
    if (window == nullptr) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(cli.bench || !cli.screenshot.empty() ? 0 : 1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // interactive runs remember the arrangement; screenshots and benchmarks always start from
    // scratch
    io.IniFilename = (cli.bench || !cli.screenshot.empty()) ? nullptr : "shamrock_gui_layout.ini";
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    App app(!cli.screenshot.empty(), cli.frames, cli.bench);

    int fbw = 0, fbh = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        app.gui();
        // temporary: something moving to check --screenshot / --bench, removed with the real panes
        {
            const double t = app.now();
            const ImVec2 c(360 + 200 * float(std::cos(t)), 240 + 120 * float(std::sin(2 * t)));
            ImGui::GetForegroundDrawList()->AddRectFilled(
                ImVec2(c.x - 20, c.y - 20),
                ImVec2(c.x + 20, c.y + 20),
                IM_COL32(232, 163, 61, 255));
        }
        ImGui::Render();
        glfwGetFramebufferSize(window, &fbw, &fbh);
        glViewport(0, 0, fbw, fbh);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        if (app.want_exit && !cli.screenshot.empty()) {
            std::vector<uint8_t> px(size_t(fbw) * fbh * 4), flipped(px.size());
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, fbw, fbh, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            for (int j = 0; j < fbh; ++j)
                std::memcpy(
                    &flipped[size_t(j) * fbw * 4],
                    &px[size_t(fbh - 1 - j) * fbw * 4],
                    size_t(fbw) * 4);
            for (size_t k = 3; k < flipped.size(); k += 4)
                flipped[k] = 255;
            stbi_write_png(cli.screenshot.c_str(), fbw, fbh, 4, flipped.data(), fbw * 4);
            std::printf("saved %s\n", cli.screenshot.c_str());
        }
        glfwSwapBuffers(window);
        if (app.want_exit)
            break;
    }
    if (cli.bench)
        print_bench(app, 30);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
