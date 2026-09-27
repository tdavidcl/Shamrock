// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DemoSimulation.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Demo data source (stand-in for the remote solver): Sedov blast slices, tracer particles,
 * diagnostics history and a profiling-scope tree, all deterministic (fixed SplitMix64 seeds).
 *
 */

#include "DemoSimulation.hpp"
#include "colormap.hpp"
#include <cstring>
#include <numbers>

// start allow utf-8
const FieldInfo FIELDS[3]
    = {{"rho", "ρ", "ρ log", true}, {"p", "p", "p log", true}, {"v", "|v|", "|v| lin", false}};
// end allow utf-8

double Rng::normal(double mu, double sigma) {
    double u1 = 1.0 - uniform(), u2 = uniform();
    return mu + sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
}

namespace {
    using Cat   = DemoSimulation::Cat;
    using Scope = DemoSimulation::Scope;

    std::vector<Scope> default_profile() {
        return {
            {"step", -1, Cat::Root, 0.03},
            {"hydro step", 0, Cat::Host, 0.05},
            {"reconstruct (PPM)", 1, Cat::Gpu, 0.62},
            {"riemann (HLLC)", 1, Cat::Gpu, 0.95},
            {"flux update", 1, Cat::Gpu, 0.48},
            {"halo exchange", 1, Cat::Mpi, 0.02},
            {"pack", 5, Cat::Gpu, 0.12},
            {"MPI_Isend / Irecv", 5, Cat::Mpi, 0.31},
            {"unpack", 5, Cat::Gpu, 0.10},
            {"boundaries", 1, Cat::Gpu, 0.12},
            {"CFL reduce", 1, Cat::Mpi, 0.18},
            {"advect tracers", 0, Cat::Host, 0.02},
            {"velocity interp", 11, Cat::Gpu, 0.30},
            {"RK2 push", 11, Cat::Gpu, 0.22},
            {"migrate", 11, Cat::Mpi, 0.10},
            {"diagnostics", 0, Cat::Host, 0.01},
            {"E_tot allreduce", 15, Cat::Mpi, 0.12},
            {"speed counter", 15, Cat::Host, 0.04},
            {"preview extract", 0, Cat::Host, 0.01},
            {"slice", 18, Cat::Gpu, 0.12},
            {"bin tracers", 18, Cat::Gpu, 0.08},
            {"D2H copy", 18, Cat::Gpu, 0.05},
            {"compress", 18, Cat::Host, 0.03},
            {"python driver", 0, Cat::Host, 0.10},
            {"checkpoint write", 0, Cat::Io, 0.0},
        };
    }
} // namespace

DemoSimulation::DemoSimulation() : prof(default_profile()), rng_(3) {
    for (auto &s : prof)
        if (s.parent >= 0)
            prof[s.parent].leaf = false;
    for (int k = 0; k < 20; ++k)
        sample_profile(); // start from a settled mean
    Rng rng(7);
    const int n = 40000;
    ang_.resize(n);
    frac_.resize(n);
    for (int i = 0; i < n; ++i) {
        ang_[i]    = rng.uniform() * 2.0 * std::numbers::pi;
        bool shell = rng.uniform() < 0.86;
        double u   = rng.uniform();
        frac_[i]   = shell ? 1.0 - 0.16 * std::pow(u, 0.6) : std::sqrt(u) * 0.8;
    }
    for (const char *k : {"E_tot", "dt", "speed"})
        history[k];
    for (int i = 0; i < 160; ++i)
        push_history(T_REF * (0.25 + 0.75 * i / 159), i < 3);
}

void DemoSimulation::advance(double wall_dt) {
    if (!running)
        return;
    int steps = int(std::max(1.0, wall_dt * 240));
    for (int i = 0; i < steps; ++i) {
        t += dt() * 1.1;
        step += 1;
    }
    if (t >= UNTIL)
        t = T_REF * 0.6; // loop the demo
    step_ms = 4.1 + rng_.normal(0, 0.08);
    push_history(t);
    sample_profile();
}

void DemoSimulation::sample_profile() {
    std::vector<double> v(prof.size(), 0.0);
    for (int i = int(prof.size()) - 1; i >= 0; --i) { // children come after their parent
        const Scope &s = prof[i];
        double self    = s.base_ms * std::max(0.0, 1 + prof_rng_.normal(0, 0.06));
        if (s.cat == Cat::Io)
            self = checkpoint_now_ ? 6.0 * (1 + prof_rng_.normal(0, 0.1)) : 0.0;
        v[i] += self;
        if (s.parent >= 0)
            v[s.parent] += v[i];
    }
    checkpoint_now_ = false;
    for (size_t i = 0; i < prof.size(); ++i)
        prof[i].ema += 0.08 * (v[i] - prof[i].ema);
}

std::pair<double, double> DemoSimulation::value_range(int field) const {
    double R = shock_radius();
    if (field == 0)
        return {0.02, 4.0};
    if (field == 1)
        return {1e-5, 2.4 * std::pow(R_REF / R, 3.0)};
    return {0.0, 1.9 * std::pow(R_REF / R, 1.5)};
}

Image DemoSimulation::slice(int field, int w, int h, double *lo_out, double *hi_out) const {
    const double aspect = double(w) / h;
    auto [lo, hi]       = value_range(field);
    const double R      = shock_radius();
    Image img{w, h, std::vector<uint8_t>(size_t(w) * h * 4)};
    std::vector<float> xs(w), ys(h);
    const double x0 = -0.5 * aspect, x1 = 0.5 * aspect, sx = (x1 - x0) / (w - 1),
                 sy = -1.0 / (h - 1);
    for (int i = 0; i < w; ++i)
        xs[i] = float(i == w - 1 ? x1 : x0 + i * sx);
    for (int j = 0; j < h; ++j)
        ys[j] = float(j == h - 1 ? -0.5 : 0.5 + j * sy);
    // outside the shock the field is constant: colour it once (identical result, far fewer libm
    // calls)
    uint8_t outside[4];
    colormap_px(
        normalize<float>(field, profile<float>(field, 2.0f * float(R) + 1.0f, 0.0f), lo, hi),
        VIRIDIS,
        outside);
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i) {
            float X = xs[i], Y = ys[j];
            float r      = std::hypot(X, Y);
            uint8_t *out = &img.px[(size_t(j) * w + i) * 4];
            if (r / float(R) < 1.0f)
                colormap_px(
                    normalize<float>(field, profile<float>(field, r, std::atan2(Y, X)), lo, hi),
                    VIRIDIS,
                    out);
            else
                std::memcpy(out, outside, 4);
        }
    if (lo_out)
        *lo_out = lo;
    if (hi_out)
        *hi_out = hi;
    return img;
}

std::vector<std::array<double, 2>> DemoSimulation::tracers_xy(int count) const {
    int n    = count < 0 ? int(ang_.size()) : count;
    double R = shock_radius();
    std::vector<std::array<double, 2>> out(n);
    for (int i = 0; i < n; ++i) {
        double rad = frac_[i] * R;
        out[i]     = {std::cos(ang_[i]) * rad, std::sin(ang_[i]) * rad};
    }
    return out;
}

Image DemoSimulation::tracers_image(int w, int h) const {
    auto xy             = tracers_xy();
    const double aspect = double(w) / h;
    std::vector<float> bins(size_t(w) * h, 0.0f);
    for (auto &p : xy) {
        int ix = std::clamp(int((p[0] / aspect + 0.5) * w), 0, w - 1);
        int iy = std::clamp(int((0.5 - p[1]) * h), 0, h - 1);
        bins[size_t(iy) * w + ix] += 1.0f;
    }
    float mx = 0;
    for (float &b : bins) {
        b  = std::log1p(b);
        mx = std::max(mx, b);
    }
    mx = std::max(mx, 1e-6f);
    Image img{w, h, std::vector<uint8_t>(size_t(w) * h * 4)};
    for (size_t k = 0; k < bins.size(); ++k)
        colormap_px(bins[k] / mx, TRACER_LUT, &img.px[k * 4]);
    return img;
}

void DemoSimulation::push_history(double tt, bool warmup) {
    auto push = [&](const char *k, double v) {
        auto &d = history[k];
        d.push_back(v);
        if (d.size() > 160)
            d.pop_front();
    };
    push("E_tot", 1.0 + 2e-7 * (tt / T_REF) + rng_.normal(0, 1.2e-8));
    push("dt", dt_at(tt) * (1 + rng_.normal(0, 0.01)));
    double speed = 4.1e9 * (1 + rng_.normal(0, 0.03));
    if (warmup)
        speed *= 0.45;
    if (rng_.uniform() < 0.015) { // an occasional checkpoint write
        speed *= 0.35;
        checkpoint_now_ = true;
    }
    push("speed", speed);
}
