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
 * @file colormap.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief 256-entry RGBA colormap lookup tables (viridis and the tracer map), shared by the demo
 * data source and the UI.
 *
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

using Lut = std::array<std::array<uint8_t, 4>, 256>;

constexpr int hexv(char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; }

inline Lut make_lut(const std::vector<const char *> &stops) {
    // same as numpy: np.interp on a 256-sample linspace, then truncation to uint8
    const size_t n = stops.size();
    std::vector<double> xs(n);
    for (size_t i = 0; i < n; ++i)
        xs[i] = double(i) / double(n - 1);
    xs[n - 1] = 1.0;
    Lut lut{};
    for (int i = 0; i < 256; ++i) {
        double t = i == 255 ? 1.0 : i * (1.0 / 255.0);
        size_t j = std::min<size_t>(
            size_t(std::upper_bound(xs.begin(), xs.end(), t) - xs.begin()), n - 1);
        j = j == 0 ? 0 : j - 1;
        for (int k = 0; k < 3; ++k) {
            const char *s  = stops[j];
            const char *s2 = stops[std::min(j + 1, n - 1)];
            double f0      = hexv(s[1 + 2 * k]) * 16 + hexv(s[2 + 2 * k]);
            double f1      = hexv(s2[1 + 2 * k]) * 16 + hexv(s2[2 + 2 * k]);
            double v  = (t >= xs[n - 1]) ? f1 : (f1 - f0) / (xs[j + 1] - xs[j]) * (t - xs[j]) + f0;
            lut[i][k] = uint8_t(v);
        }
        lut[i][3] = 255;
    }
    return lut;
}
inline const Lut VIRIDIS = make_lut(
    {"#440154",
     "#482878",
     "#3e4989",
     "#31688e",
     "#26828e",
     "#1f9e89",
     "#35b779",
     "#6ece58",
     "#fde725"});
inline const Lut TRACER_LUT = make_lut({"#0b0c10", "#1c2a4a", "#3d5f9e", "#9ec0ff", "#e4eeff"});

inline void colormap_px(float v01, const Lut &lut, uint8_t *out) {
    int idx = std::clamp(int(v01 * 255.0f), 0, 255);
    std::memcpy(out, lut[idx].data(), 4);
}
