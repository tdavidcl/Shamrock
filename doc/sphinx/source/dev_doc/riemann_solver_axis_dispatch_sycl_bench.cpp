// SYCL microbenchmark: axis-permutation dispatch (`_mz`) vs direct
// projection (`_n`) for a generic Riemann solver flux, run on real
// sycl::vec<double, 3> data over 10^7 randomly generated face states.
//
// Companion to riemann_solver_axis_dispatch_godbolt.cpp (same comparison,
// CPU-only, single-call assembly diff): this file measures the actual
// wall-clock cost of the pattern across a full kernel launch on whatever
// SYCL device you point it at. See dev_doc/riemann_solver.md, section
// "Axis permutation vs projection".
//
// Build (pick whichever SYCL implementation you have):
//   acpp -O3 riemann_solver_axis_dispatch_sycl_bench.cpp -o bench
//   icpx -fsycl -O3 riemann_solver_axis_dispatch_sycl_bench.cpp -o bench
//   clang++ -fsycl -O3 riemann_solver_axis_dispatch_sycl_bench.cpp -o bench
//
// Run:
//   ./bench

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using Tscal = double;
using Tvec  = sycl::vec<double, 3>;

constexpr std::size_t N       = 10'000'000; // 10^7 elements
constexpr int repeats         = 20;         // timed repeats per case (min is reported)
constexpr Tscal validate_tol  = 1e-12;

struct PrimState {
    Tscal rho;
    Tvec vel;
};

struct ConsState {
    Tscal rho;
    Tvec rhovel;

    ConsState &operator*=(Tscal f) {
        rho *= f;
        rhovel *= f;
        return *this;
    }
};

inline ConsState operator+(const ConsState &a, const ConsState &b) {
    return ConsState{a.rho + b.rho, a.rhovel + b.rhovel};
}

// --- riemann_common.hpp helpers used on the mz path ---
inline ConsState hydro_flux_n(PrimState prim, Tvec n, Tscal vn) {
    return ConsState{prim.rho * vn, prim.vel * (prim.rho * vn)};
}
inline ConsState hydro_flux_n(PrimState prim, Tvec n) {
    Tscal vn = sycl::dot(n, prim.vel);
    return hydro_flux_n(prim, n, vn);
}

inline ConsState x_to_z(ConsState c) {
    return ConsState{c.rho, Tvec{-c.rhovel[2], c.rhovel[1], c.rhovel[0]}};
}
inline ConsState invert_axis(ConsState c) {
    return ConsState{c.rho, -c.rhovel};
}
inline PrimState prim_z_to_x(PrimState p) {
    return PrimState{p.rho, Tvec{p.vel[2], p.vel[1], -p.vel[0]}};
}
inline PrimState prim_invert_axis(PrimState p) {
    return PrimState{p.rho, -p.vel};
}

// --- a generic Riemann solver, standing in for any of Rusanov/HLL/HLLC/
//     dust-HLL/Huang-Bai: only the "_n" variant does real physics, every
//     per-axis wrapper below is pure plumbing around it. ---
inline ConsState riemann_solver_flux_n(PrimState primL, PrimState primR, Tvec n) {
    Tscal vnL = sycl::dot(n, primL.vel);
    Tscal vnR = sycl::dot(n, primR.vel);

    ConsState fL = hydro_flux_n(primL, n, vnL);
    ConsState fR = hydro_flux_n(primR, n, vnR);

    ConsState flux{0, Tvec{0, 0, 0}};

    if (vnL > 0 && vnR > 0)
        flux = fL;
    else if (vnL < 0 && vnR < 0)
        flux = fR;
    else if (vnL < 0 && vnR > 0)
        flux *= 0;
    else if (vnL > 0 && vnR < 0)
        flux = fL + fR;

    return flux;
}

inline ConsState riemann_solver_flux_x(PrimState primL, PrimState primR) {
    return riemann_solver_flux_n(primL, primR, Tvec{1, 0, 0});
}
inline ConsState riemann_solver_flux_z(PrimState pL, PrimState pR) {
    return x_to_z(riemann_solver_flux_x(prim_z_to_x(pL), prim_z_to_x(pR)));
}
// axis permutation: rotate the inputs to +x, solve there, rotate the result back
inline ConsState riemann_solver_flux_mz(PrimState pL, PrimState pR) {
    return invert_axis(riemann_solver_flux_z(prim_invert_axis(pL), prim_invert_axis(pR)));
}

int main() {
    sycl::queue q;
    std::printf(
        "Device: %s\n",
        q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("N = %zu elements, %d repeats per case (best of N reported)\n\n", N, repeats);

    std::vector<PrimState> hL(N), hR(N);
    {
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<Tscal> rho_dist(0.1, 5.0);
        std::uniform_real_distribution<Tscal> vel_dist(-2.0, 2.0);
        for (std::size_t i = 0; i < N; ++i) {
            hL[i] = PrimState{rho_dist(rng), Tvec{vel_dist(rng), vel_dist(rng), vel_dist(rng)}};
            hR[i] = PrimState{rho_dist(rng), Tvec{vel_dist(rng), vel_dist(rng), vel_dist(rng)}};
        }
    }

    sycl::buffer<PrimState, 1> bL(hL.data(), sycl::range<1>(N));
    sycl::buffer<PrimState, 1> bR(hR.data(), sycl::range<1>(N));

    auto submit_mz = [&](sycl::buffer<ConsState, 1> &bOut) {
        return q.submit([&](sycl::handler &cgh) {
            sycl::accessor accL{bL, cgh, sycl::read_only};
            sycl::accessor accR{bR, cgh, sycl::read_only};
            sycl::accessor accOut{bOut, cgh, sycl::write_only, sycl::no_init};
            cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                accOut[i] = riemann_solver_flux_mz(accL[i], accR[i]);
            });
        });
    };
    auto submit_n = [&](sycl::buffer<ConsState, 1> &bOut) {
        return q.submit([&](sycl::handler &cgh) {
            sycl::accessor accL{bL, cgh, sycl::read_only};
            sycl::accessor accR{bR, cgh, sycl::read_only};
            sycl::accessor accOut{bOut, cgh, sycl::write_only, sycl::no_init};
            cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                accOut[i] = riemann_solver_flux_n(accL[i], accR[i], Tvec{0, 0, -1});
            });
        });
    };

    // --- correctness check: both variants must agree everywhere ---
    {
        sycl::buffer<ConsState, 1> bOutMz{sycl::range<1>(N)};
        sycl::buffer<ConsState, 1> bOutN{sycl::range<1>(N)};
        submit_mz(bOutMz).wait();
        submit_n(bOutN).wait();

        sycl::host_accessor hOutMz{bOutMz, sycl::read_only};
        sycl::host_accessor hOutN{bOutN, sycl::read_only};

        Tscal max_diff = 0;
        for (std::size_t i = 0; i < N; ++i) {
            max_diff = std::max(max_diff, std::abs(hOutMz[i].rho - hOutN[i].rho));
            for (int c = 0; c < 3; ++c)
                max_diff
                    = std::max(max_diff, std::abs(hOutMz[i].rhovel[c] - hOutN[i].rhovel[c]));
        }
        std::printf(
            "correctness: max |via_mz_dispatch - via_flux_n| = %.3e  (%s)\n\n",
            max_diff,
            max_diff < validate_tol ? "PASS" : "FAIL");
    }

    // --- timing ---
    auto time_case = [&](const char *label, auto &&submit) {
        sycl::buffer<ConsState, 1> bOut{sycl::range<1>(N)};

        submit(bOut).wait(); // warm-up (first-touch / JIT / allocator effects)

        double best_ms = 1e300;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            submit(bOut).wait();
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            best_ms = std::min(best_ms, ms);
        }
        std::printf(
            "%-16s : best of %2d runs = %9.3f ms  (%.3f ns/elem)\n",
            label,
            repeats,
            best_ms,
            best_ms * 1e6 / static_cast<double>(N));
    };

    time_case("via_mz_dispatch", submit_mz);
    time_case("via_flux_n", submit_n);

    return 0;
}
