// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file version.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief typedefs and macros
 * @date 2021-09-17
 * @copyright Copyright Timothée David--Cléris (c) 2021
 *
 */

#include "shambase/aliases_int.hpp"
#include "shamcmdopt/term_colors.hpp"
#include "shamcomm/logs.hpp"
#include <string>

extern const std::string git_info_str;
extern const std::string git_commit_hash;
extern const std::string compile_arg;
extern const std::string version_string;
extern const bool is_git;

const u32 term_width = 64;

inline std::string shamrock_title_bar_big = "\n\
  █████████  █████   █████   █████████   ██████   ██████ ███████████      ███████      █████████  █████   ████\n\
 ███░░░░░███░░███   ░░███   ███░░░░░███ ░░██████ ██████ ░░███░░░░░███   ███░░░░░███   ███░░░░░███░░███   ███░ \n\
░███    ░░░  ░███    ░███  ░███    ░███  ░███░█████░███  ░███    ░███  ███     ░░███ ███     ░░░  ░███  ███   \n\
░░█████████  ░███████████  ░███████████  ░███░░███ ░███  ░██████████  ░███      ░███░███          ░███████    \n\
 ░░░░░░░░███ ░███░░░░░███  ░███░░░░░███  ░███ ░░░  ░███  ░███░░░░░███ ░███      ░███░███          ░███░░███   \n\
 ███    ░███ ░███    ░███  ░███    ░███  ░███      ░███  ░███    ░███ ░░███     ███ ░░███     ███ ░███ ░░███  \n\
░░█████████  █████   █████ █████   █████ █████     █████ █████   █████ ░░░███████░   ░░█████████  █████ ░░████\n\
 ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░   ░░░░░ ░░░░░     ░░░░░ ░░░░░   ░░░░░    ░░░░░░░      ░░░░░░░░░  ░░░░░   ░░░░ \n\
";

inline void print_title_bar() {
    logger::raw_ln(shamrock_title_bar_big);
    logger::raw_ln(
        shambase::term_colors::col8b_cyan()
        + "Copyright (c) 2021-2024 Timothée David--Cléris (tim.shamrock@proton.me)"
        + shambase::term_colors::reset());
    logger::raw_ln(
        shambase::term_colors::col8b_cyan() + "SPDX-License-Identifier"
        + shambase::term_colors::reset() + " : CeCILL Free Software License Agreement v2.1");
    logger::print_faint_row();

    logger::raw_ln(
        "\n" + shambase::term_colors::col8b_cyan() + "Shamrock version "
        + shambase::term_colors::reset() + ": " + version_string + "\n");

    if (is_git) {
        logger::raw_ln(
            shambase::term_colors::col8b_cyan() + "Git infos " + shambase::term_colors::reset()
            + ":\n" + shambase::trunc_str(git_info_str, 512));
    }
}
