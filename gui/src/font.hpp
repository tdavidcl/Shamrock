// IBM Plex fonts used by the GUI (Sans for UI text, Mono for values and code), loaded from
// <assets>/fonts.

#pragma once

#include "imgui.h"
#include <filesystem>
#include <string>

struct Fonts {
    static inline ImFont *sans, *medium, *semibold, *mono;
};

inline void load_fonts(const std::filesystem::path &assets) {
    ImGuiIO &io = ImGui::GetIO();
    auto load   = [&](const char *name, bool merge = false) {
        ImFontConfig cfg;
        cfg.MergeMode    = merge;
        std::string path = (assets / "fonts" / (std::string(name) + ".ttf")).string();
        ImFont *f        = io.Fonts->AddFontFromFileTTF(path.c_str(), 13.0f, &cfg);
        IM_ASSERT(f && "font not found: run from the project folder or pass --assets");
        return f;
    };
    Fonts::sans     = load("IBMPlexSans-Regular");
    Fonts::medium   = load("IBMPlexSans-Medium");
    Fonts::semibold = load("IBMPlexSans-SemiBold");
    Fonts::mono     = load("IBMPlexMono-Regular");
    load("IBMPlexSans-Regular", true); // Plex Mono has no Greek (rho): borrow it from Sans
}
