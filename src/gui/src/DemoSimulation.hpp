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
 * @file DemoSimulation.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Demo data source (stand-in for the remote solver).
 *
 * The UI depends only on the public members below; the real SSH client will replace this class
 * behind the same interface.
 *
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct FieldInfo {
    const char *id;
    const char *label;
    const char *cb_label;
    bool log;
};
extern const FieldInfo FIELDS[3];

struct Rng { // SplitMix64, bit-identical to the Python Rng
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed) {}
    uint64_t next_u64() {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return double(next_u64() >> 11) * (1.0 / 9007199254740992.0); }
    double normal(double mu, double sigma);
};

struct Image {
    int w, h;
    std::vector<uint8_t> px;
};

class DemoSimulation {
    public:
    static constexpr double T_REF = 0.03214, R_REF = 0.37, UNTIL = 0.05;
    double t       = T_REF;
    long long step = 18420;
    bool running   = true;
    double step_ms = 4.1;
    std::map<std::string, std::deque<double>> history;

    // Per-step profiling scopes as the solver would report them (rank 0). Scopes are listed parents
    // first; `ema` is the running mean the flamegraph shows. In the real app this arrives as one
    // more subscription on the SSH channel.
    enum class Cat { Root, Gpu, Mpi, Host, Io };
    struct Scope {
        const char *name;
        int parent;
        Cat cat;
        double base_ms;
        double ema = 0;
        bool leaf  = true;
    };
    std::vector<Scope> prof;

    DemoSimulation();

    double dt() const { return dt_at(t); }
    static double dt_at(double tt) { return 1.81e-6 * std::pow(tt / T_REF, 0.6); }

    void advance(double wall_dt);
    void sample_profile();

    double shock_radius() const { return R_REF * std::pow(t / T_REF, 0.4); }

    // float instantiation mirrors the numpy float32 slice maths, double mirrors `sample`
    template<class T>
    T profile(int field, T r, T theta) const {
        const double R = shock_radius();
        const T q      = r / T(R);
        if (!(q < T(1))) // outside the shock the value is constant: skip the trig (same result)
            return field == 0 ? T(1) : field == 1 ? T(1e-5) : T(0);
        const T ripple = T(1)
                         + T(0.04) * std::sin(T(7) * theta + T(3) * q)
                               * std::pow(std::clamp(q, T(0), T(1)), T(4));
        if (field == 0)
            return (T(0.02) + T(3.98) * std::pow(q, T(9))) * ripple;
        if (field == 1) {
            T ps = T(2.4 * std::pow(R_REF / R, 3.0));
            return ps * (T(0.36) + T(0.64) * std::pow(q, T(6))) * ripple;
        }
        T vs = T(1.9 * std::pow(R_REF / R, 1.5));
        return vs * q * ripple;
    }

    double sample(int field, double x, double y) const {
        return profile<double>(field, std::hypot(x, y), std::atan2(y, x));
    }

    std::pair<double, double> value_range(int field) const;

    template<class T>
    static T normalize(int field, T f, double lo, double hi) {
        if (FIELDS[field].log)
            return (std::log10(std::max(f, T(lo))) - T(std::log10(lo)))
                   / T(std::log10(hi) - std::log10(lo));
        return (f - T(lo)) / T(std::max(hi - lo, 1e-30));
    }

    Image slice(int field, int w, int h, double *lo_out = nullptr, double *hi_out = nullptr) const;
    std::vector<std::array<double, 2>> tracers_xy(int count = -1) const;
    Image tracers_image(int w, int h) const;

    private:
    std::vector<double> ang_, frac_;
    Rng rng_;
    Rng prof_rng_{11};
    bool checkpoint_now_ = false;

    void push_history(double tt, bool warmup = false);
};
