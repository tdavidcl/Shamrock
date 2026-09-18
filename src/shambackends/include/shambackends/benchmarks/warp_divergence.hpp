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
 * @file warp_divergence.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Branch divergence penalty benchmark
 *
 * The same kernel is run twice with the same amount of useful work per work item,
 * once with neighbouring work items taking the same branch and once with each of
 * them taking a different one. The ratio of the two is the divergence penalty.
 *
 * Sweeping n_paths also exposes the SIMD width of the device: the penalty stops
 * growing once n_paths exceeds the number of lanes that execute in lockstep, so a
 * 32 wide warp saturates at 32 while a 64 wide wavefront keeps growing to 64.
 */

#include "shambase/assert.hpp"
#include "shambase/time.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"

namespace sham::benchmarks {

    /// Shift applied to the work item id to build a divergence free path assignment.
    /// 1024 consecutive work items share a path, which is above any SIMD width.
    inline constexpr u32 divergence_uniform_shift = 10;

    /**
     * @brief Kernel for the warp_divergence benchmark.
     *
     * Every work item runs exactly one branch body of `nrotation` iterations, so the
     * useful work does not depend on the path assignment. Only the number of bodies
     * a SIMD group has to walk through does.
     *
     * The body is seeded from `k` so that the compiler cannot hoist it out of the
     * path loop and collapse the control flow.
     *
     * @tparam T floating point type used for the chain
     * @param i index of the element to process
     * @param n_paths number of branches, must be a power of two
     * @param path_shift shift applied to `i` before selecting the path
     * @param nrotation number of chain rotations inside the branch body
     * @param y0 initial value of the chain
     * @param in input vector
     * @param out output vector
     */
    template<class T>
    inline void divergence_chains(
        u32 i,
        u32 n_paths,
        u32 path_shift,
        int nrotation,
        T y0,
        T *__restrict in,
        T *__restrict out) {

        T x = in[i];
        T y = y0;

        u32 path = (i >> path_shift) & (n_paths - 1);

        for (u32 k = 0; k < n_paths; k++) {
            if (path == k) {
                T a = y0 + T(k);
                for (int j = 0; j < nrotation; j++) {
                    x = a * x + y;
                    y = x * y + a;
                }
            }
        }

        out[i] = y;
    }

    /// Structure containing the results of a warp_divergence benchmark
    struct divergence_result {
        std::string func_name; ///< Name of the function
        f64 seconds_uniform;   ///< Time of the divergence free run in seconds
        f64 seconds_divergent; ///< Time of the divergent run in seconds
        f64 penalty;           ///< Slowdown caused by divergence alone
        u32 n_paths;           ///< Number of branches
        u32 nrotations;        ///< Number of rotations inside a branch body
    };

    /**
     * @brief Run the warp_divergence benchmark for one branch count.
     *
     * @tparam T floating point type used for the chain
     * @param sched scheduler for the target device
     * @param N number of elements to process
     * @param n_paths number of branches, must be a power of two
     * @param time_threshold minimum wall-clock time to run the benchmark in seconds
     * @return benchmark results as a divergence_result
     */
    template<class T>
    inline divergence_result divergence_bench(
        DeviceScheduler_ptr sched, int N, u32 n_paths, f64 time_threshold) {

        SHAM_ASSERT(n_paths > 0 && (n_paths & (n_paths - 1)) == 0);

        sham::DeviceQueue &q = sched->get_queue();

        sham::DeviceBuffer<T> x = {size_t(N), sched};
        sham::DeviceBuffer<T> y = {size_t(N), sched};

        const T x0 = T{1.1};
        const T y0 = -x0;

        x.fill(x0);
        y.fill(y0);

        sham::EventList depends_list;

        auto x_ptr = x.get_write_access(depends_list);
        auto y_ptr = y.get_write_access(depends_list);

        depends_list.wait();

        auto run_bench
            = [&q, &N, &x_ptr, &y_ptr, y0, n_paths](u32 nrotation, u32 path_shift) -> f64 {
            sham::EventList empty_list{};

            shambase::Timer t;
            t.start();
            auto e = q.submit(empty_list, [=](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>{size_t(N)}, [=](sycl::item<1> item) {
                    divergence_chains<T>(
                        item.get_linear_id(), n_paths, path_shift, nrotation, y0, x_ptr, y_ptr);
                });
            });
            e.wait();
            t.stop();

            return t.elapsed_sec();
        };

        // warmup kernel
        run_bench(4, divergence_uniform_shift);

        u32 nrotation = 8;
        f64 sec_uni   = 0;

        for (;;) {

            sec_uni = run_bench(nrotation, divergence_uniform_shift);

            if (sec_uni >= time_threshold || nrotation >= 256 * 256) {
                break;
            }

            nrotation *= 2;
        }

        // launch overhead and empty path loop, measured for both assignments since
        // walking the path loop can itself diverge
        f64 ref_uni = run_bench(0, divergence_uniform_shift);
        f64 ref_div = run_bench(0, 0);

        sec_uni     = run_bench(nrotation, divergence_uniform_shift) - ref_uni;
        f64 sec_div = run_bench(nrotation, 0) - ref_div;

        x.complete_event_state(sycl::event{});
        y.complete_event_state(sycl::event{});

        f64 penalty = (sec_uni > 0) ? (sec_div / sec_uni) : f64(0);

        return {
            .func_name         = SourceLocation{}.loc.function_name(),
            .seconds_uniform   = sec_uni,
            .seconds_divergent = sec_div,
            .penalty           = penalty,
            .n_paths           = n_paths,
            .nrotations        = nrotation};
    }

} // namespace sham::benchmarks
