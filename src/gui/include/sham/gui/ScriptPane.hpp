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
 * @file ScriptPane.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Script pane: the run script in a Python editor, with its file tabs and apply bar.
 *
 */

#include "TextEditor.h"
#include "sham/gui/ui.hpp"

namespace sham::gui {

    struct ScriptPane {
        TextEditor editor;

        // Loads the demo run script; needs the ImGui context.
        void init();
        void draw(SDL *dl, double x, double y, double w, double h);
    };

} // namespace sham::gui
