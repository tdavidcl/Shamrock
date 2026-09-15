// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file color.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Terminal color escape sequence constants and enable/disable control implementation
 *
 */

#include <sham/term/color.hpp>
#include <sham/term/env.hpp>
#include <string_view>
#include <cstdlib>
#include <optional>
#include <string>

#define TERM_ESCAPTE_CHAR "\x1b["
namespace {
    /// Detected/forced terminal color support level, as set by sham::term::set_color_level.
    /// Single source of truth for both "are colors enabled" (level != NoColor) and "which tiered
    /// palette can be rendered".
    sham::term::ColorLevel color_level_value = sham::term::ColorLevel::NoColor;

    /// Read a raw process environment variable as an optional string_view, for the detection
    /// performed by detected_color_level() below.
    std::optional<std::string_view> getenv_view(const char *name) {
        const char *value = std::getenv(name);
        if (value == nullptr) {
            return std::nullopt;
        }
        return std::string_view(value);
    }

    /// Last non-NoColor level passed to set_color_level(), i.e. the last detected/forced
    /// terminal capability. Lazily detected on first use straight from the real TERM/COLORTERM
    /// process environment variables (see sham::term::detect_color_level), so a program that
    /// never calls parse_terminal_support()/set_color_level() still gets a sensible level for
    /// enable_colors() to restore instead of unconditionally falling back to Basic.
    sham::term::ColorLevel &detected_color_level() {
        static sham::term::ColorLevel level = sham::term::detect_color_level(
            {.TERM = getenv_view("TERM"), .COLORTERM = getenv_view("COLORTERM")});
        return level;
    }

    const char *_empty_str     = "";
    const char *_esc_char      = TERM_ESCAPTE_CHAR;
    const char *_reset         = TERM_ESCAPTE_CHAR "0m";
    const char *_bold          = TERM_ESCAPTE_CHAR "1m";
    const char *_faint         = TERM_ESCAPTE_CHAR "2m";
    const char *_underline     = TERM_ESCAPTE_CHAR "4m";
    const char *_blink         = TERM_ESCAPTE_CHAR "5m";
    const char *_col8b_black   = TERM_ESCAPTE_CHAR "30m";
    const char *_col8b_red     = TERM_ESCAPTE_CHAR "31m";
    const char *_col8b_green   = TERM_ESCAPTE_CHAR "32m";
    const char *_col8b_yellow  = TERM_ESCAPTE_CHAR "33m";
    const char *_col8b_blue    = TERM_ESCAPTE_CHAR "34m";
    const char *_col8b_magenta = TERM_ESCAPTE_CHAR "35m";
    const char *_col8b_cyan    = TERM_ESCAPTE_CHAR "36m";
    const char *_col8b_white   = TERM_ESCAPTE_CHAR "37m";
} // namespace

namespace sham::term {

    ColorLevel color_level() { return color_level_value; }
    void set_color_level(ColorLevel level) {
        color_level_value = level;
        if (level != ColorLevel::NoColor) {
            detected_color_level() = level;
        }
    }

    namespace style {
        const char *reset() {
            return (color_level_value != ColorLevel::NoColor) ? _reset : _empty_str;
        }
        const char *bold() {
            return (color_level_value != ColorLevel::NoColor) ? _bold : _empty_str;
        }
        const char *faint() {
            return (color_level_value != ColorLevel::NoColor) ? _faint : _empty_str;
        }
        const char *underline() {
            return (color_level_value != ColorLevel::NoColor) ? _underline : _empty_str;
        }
        const char *blink() {
            return (color_level_value != ColorLevel::NoColor) ? _blink : _empty_str;
        }
    } // namespace style

    namespace colors_8b {
        const char *black() {
            return (color_level_value != ColorLevel::NoColor) ? _col8b_black : _empty_str;
        }
        const char *red() {
            return (color_level_value != ColorLevel::NoColor) ? _col8b_red : _empty_str;
        }
        const char *green() {
            return (color_level_value != ColorLevel::NoColor) ? _col8b_green : _empty_str;
        }
        const char *yellow() {
            return (color_level_value != ColorLevel::NoColor) ? _col8b_yellow : _empty_str;
        }
        const char *blue() {
            return (color_level_value != ColorLevel::NoColor) ? _col8b_blue : _empty_str;
        }
        const char *magenta() {
            return (color_level_value != ColorLevel::NoColor) ? _col8b_magenta : _empty_str;
        }
        const char *cyan() {
            return (color_level_value != ColorLevel::NoColor) ? _col8b_cyan : _empty_str;
        }
        const char *white() {
            return (color_level_value != ColorLevel::NoColor) ? _col8b_white : _empty_str;
        }
    } // namespace colors_8b

    namespace colors_256 {
        namespace {
            /// Build a \x1b[<mode>;5;Nm 256-color escape sequence, or an empty string if the
            /// terminal was not detected (or forced) to support the 256-color palette.
            std::string build(int mode, std::uint8_t index) {
                if (color_level_value < ColorLevel::ANSI256) {
                    return "";
                }
                return std::string(TERM_ESCAPTE_CHAR) + std::to_string(mode) + ";5;"
                       + std::to_string(index) + "m";
            }
        } // namespace

        std::string foreground(std::uint8_t index) { return build(38, index); }
        std::string background(std::uint8_t index) { return build(48, index); }
    } // namespace colors_256

    namespace colors_24b {
        namespace {
            /// Build a \x1b[<mode>;2;r;g;bm truecolor escape sequence, or an empty string if the
            /// terminal was not detected (or forced) to support truecolor.
            std::string build(int mode, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
                if (color_level_value < ColorLevel::TrueColor) {
                    return "";
                }
                return std::string(TERM_ESCAPTE_CHAR) + std::to_string(mode) + ";2;"
                       + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b)
                       + "m";
            }
        } // namespace

        std::string foreground(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
            return build(38, r, g, b);
        }
        std::string background(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
            return build(48, r, g, b);
        }
    } // namespace colors_24b

    /// Enable colors: restore the last detected/forced non-NoColor level (or ColorLevel::Basic
    /// if none was ever detected), so style/colors_8b/colors_256/colors_24b escapes are emitted
    /// up to whatever tier the terminal actually supports, instead of unconditionally dropping
    /// back to basic colors.
    void enable_colors() {
        if (color_level_value == ColorLevel::NoColor) {
            color_level_value = detected_color_level();
        }
    }

    /// Disable all colors: no tier is emitted anymore, regardless of the previously
    /// detected/forced level.
    void disable_colors() { color_level_value = ColorLevel::NoColor; }

    /// Are colors enabled
    bool are_colors_enabled() { return color_level_value != ColorLevel::NoColor; }

} // namespace sham::term
