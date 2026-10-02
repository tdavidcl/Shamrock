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
 * @file GLTexture.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief RGBA8 OpenGL texture updated in place and shown through ImGui.
 *
 */

#include "imgui.h"
#include <GLFW/glfw3.h>
#if defined(__APPLE__)
    #include <OpenGL/gl3.h>
#else
    #include <GL/gl.h>
#endif
#ifndef GL_CLAMP_TO_EDGE
    #define GL_CLAMP_TO_EDGE 0x812F
#endif
#include <cstdint>
#include <vector>

namespace sham::gui {

    // ============================================================================
    //  GPU textures and colormaps
    // ============================================================================
    struct GLTexture {
        int w = 0, h = 0;
        GLuint id = 0;
        ImTextureRef ref;
        void create(int w_, int h_, const unsigned char *data = nullptr) {
            w = w_;
            h = h_;
            glGenTextures(1, &id);
            glBindTexture(GL_TEXTURE_2D, id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            ref = ImTextureRef(ImTextureID(intptr_t(id)));
        }
        void upload(const std::vector<uint8_t> &rgba_img) const {
            glBindTexture(GL_TEXTURE_2D, id);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(
                GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba_img.data());
        }
    };

} // namespace sham::gui
