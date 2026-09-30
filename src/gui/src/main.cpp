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
 * --screenshot).
 *
 * Interactive runs remember the dock arrangement in shamrock_gui_layout.ini.
 *
 * Usage:
 *
 *     ./shamrock_gui                        interactive
 *     ./shamrock_gui --screenshot shot.png  render 45 frames (or --frames N), save PNG, exit
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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace sham::gui {

    /// Build one frame: a full-screen host window holding the dock area.
    void gui() {
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
    }

    /// Command-line options of shamrock_gui.
    struct CliArgs {
        /// --screenshot: save PNG of window before exit
        std::string screenshot = {};

        /// frames rendered before exiting: --frames N (default 45) with --screenshot, empty for an
        /// interactive run
        std::optional<int> frames_before_exit = std::nullopt;

        /// set when main must return right away (usage printed)
        std::optional<int> exit_code = std::nullopt;
    };

    /// Parse argv; prints the usage and sets exit_code on -h / --help or an unknown option.
    static CliArgs parse_cli(int argc, char **argv) {
        CliArgs cli;
        std::optional<int> frames;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next     = [&]() {
                return i + 1 < argc ? std::string(argv[++i]) : std::string();
            };
            if (a == "--screenshot")
                cli.screenshot = next();
            else if (a == "--frames")
                frames = std::stoi(next());
            else {
                std::printf("usage: %s [--screenshot out.png] [--frames N]\n", argv[0]);
                cli.exit_code = a == "-h" || a == "--help" ? 0 : 1;
                return cli;
            }
        }
        if (!cli.screenshot.empty())
            cli.frames_before_exit = frames.value_or(45);
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
    glfwSwapInterval(cli.screenshot.empty() ? 1 : 0);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // interactive runs remember the arrangement; screenshots always start from scratch
    io.IniFilename = cli.screenshot.empty() ? "shamrock_gui_layout.ini" : nullptr;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    // screenshots run on a fixed 60 fps virtual clock so they are reproducible
    const bool deterministic = !cli.screenshot.empty();
    long long frame          = 0;
    auto now                 = [&]() {
        using namespace std::chrono;
        return deterministic ? double(frame) / 60.0
                             : duration<double>(steady_clock::now().time_since_epoch()).count();
    };

    int fbw = 0, fbh = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        gui();
        frame += 1;
        const bool want_exit = cli.frames_before_exit && frame >= *cli.frames_before_exit;
        // temporary: something moving to check --screenshot, removed with the real panes
        {
            const double t = now();
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
        if (want_exit && !cli.screenshot.empty()) {
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
        if (want_exit)
            break;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
