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
 * @file cache_chase.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Cache hierarchy benchmark based on parallel random pointer chasing
 *
 * Each work item follows its own dependent chain of random indices inside a
 * working set of a given size. Sweeping the working set size makes the cache
 * tiers visible: as long as it fits in a cache level the per-hop cost stays flat,
 * and it jumps when the level overflows.
 *
 * This is the access pattern of the tree traversal in the neighbour search
 * (dependent, irregular, index driven loads), which the streaming saxpy and the
 * ALU bound fma_chains benchmarks do not exercise.
 */

#include "shambase/assert.hpp"
#include "shambase/time.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include <vector>

namespace sham::benchmarks {

    /**
     * @brief Build a next-index table forming a single random cycle over [0, n_elem[.
     *
     * A single cycle (rather than independently drawn random indices) is required:
     * a random mapping collapses onto a cycle of length O(sqrt(n_elem)), which would
     * fit in cache and hide the very effect being measured.
     *
     * @param n_elem number of elements in the cycle, must be >= 2
     * @param seed seed of the shuffle
     * @return the next-index table
     */
    inline std::vector<u32> make_random_cycle(u32 n_elem, u64 seed) {

        SHAM_ASSERT(n_elem >= 2);

        std::vector<u32> perm(n_elem);
        for (u32 i = 0; i < n_elem; i++) {
            perm[i] = i;
        }

        u64 state     = (seed == 0) ? 0x9E3779B97F4A7C15ULL : seed;
        auto rand_u64 = [&state]() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            return state;
        };

        for (u32 i = n_elem - 1; i > 0; i--) {
            u32 j = u32(rand_u64() % (u64(i) + 1));
            std::swap(perm[i], perm[j]);
        }

        std::vector<u32> next(n_elem);
        for (u32 k = 0; k < n_elem; k++) {
            next[perm[k]] = perm[(k + 1) % n_elem];
        }

        return next;
    }

    /**
     * @brief Kernel for the cache_chase benchmark.
     *
     * Pure dependent loads: no arithmetic is mixed in, so the measurement is not
     * polluted by the integer throughput of the device.
     *
     * @param w index of the chain to follow
     * @param nsteps number of hops to perform
     * @param next next-index table (the working set)
     * @param starts starting offset of each chain
     * @param out output vector, keeps the chain alive
     */
    inline void cache_chase(
        u32 w,
        u32 nsteps,
        const u32 *__restrict next,
        const u32 *__restrict starts,
        u32 *__restrict out) {

        u32 idx = starts[w];
        for (u32 s = 0; s < nsteps; s++) {
            idx = next[idx];
        }
        out[w] = idx;
    }

    /// Structure containing the results of a cache_chase benchmark
    struct cache_chase_result {
        std::string func_name; ///< Name of the function
        f64 seconds;           ///< Computation time in seconds
        f64 latency;     ///< Wall-clock time per hop with every chain in flight, in seconds. Not an
                         ///< unloaded latency, use it to compare working set sizes on one device
        f64 hop_rate;    ///< Hops per second, summed over all chains
        u64 working_set; ///< Size of the working set in bytes
        u32 nsteps;      ///< Number of hops per chain
        u32 n_chains;    ///< Number of chains followed in parallel
    };

    /**
     * @brief Run the cache_chase benchmark for one working set size.
     *
     * Note that a hop reads 4 bytes but pulls a full cache line on a miss, so
     * hop_rate is a rate of dependent accesses, not a bandwidth.
     *
     * @param sched scheduler for the target device
     * @param n_elem number of u32 entries in the working set
     * @param n_chains number of chains followed in parallel
     * @param time_threshold minimum wall-clock time to run the benchmark in seconds
     * @return benchmark results as a cache_chase_result
     */
    inline cache_chase_result cache_chase_bench(
        DeviceScheduler_ptr sched, u32 n_elem, u32 n_chains, f64 time_threshold) {

        sham::DeviceQueue &q = sched->get_queue();

        std::vector<u32> next_host = make_random_cycle(n_elem, 0x1234567ULL);

        // spread the chain starts over the array, any position is a random point
        // of the cycle since the cycle itself is randomly ordered
        std::vector<u32> starts_host(n_chains);
        for (u32 w = 0; w < n_chains; w++) {
            starts_host[w] = u32((u64(w) * u64(n_elem)) / u64(n_chains));
        }

        sham::DeviceBuffer<u32> next{size_t(n_elem), sched};
        sham::DeviceBuffer<u32> starts{size_t(n_chains), sched};
        sham::DeviceBuffer<u32> out{size_t(n_chains), sched};

        next.copy_from_stdvec(next_host);
        starts.copy_from_stdvec(starts_host);

        sham::EventList depends_list;

        auto next_ptr   = next.get_read_access(depends_list);
        auto starts_ptr = starts.get_read_access(depends_list);
        auto out_ptr    = out.get_write_access(depends_list);

        depends_list.wait();

        u32 nsteps = 64;
        double sec = 0;

        auto run_bench = [&q, n_chains, next_ptr, starts_ptr, out_ptr](u32 nsteps) -> f64 {
            sham::EventList empty_list{};

            shambase::Timer t;
            t.start();
            auto e = q.submit(empty_list, [=](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>{size_t(n_chains)}, [=](sycl::item<1> item) {
                    cache_chase(item.get_linear_id(), nsteps, next_ptr, starts_ptr, out_ptr);
                });
            });
            e.wait();
            t.stop();

            return t.elapsed_sec();
        };

        // warmup kernel, also brings the working set into cache if it fits
        run_bench(64);

        double ref = run_bench(0);

        for (;;) {

            sec = run_bench(nsteps);

            if (sec >= time_threshold || nsteps >= (1u << 22)) {
                break;
            }

            nsteps *= 2;
        }

        next.complete_event_state(sycl::event{});
        starts.complete_event_state(sycl::event{});
        out.complete_event_state(sycl::event{});

        sec -= ref;

        return {
            .func_name   = SourceLocation{}.loc.function_name(),
            .seconds     = sec,
            .latency     = sec / double(nsteps),
            .hop_rate    = (double(n_chains) * double(nsteps)) / sec,
            .working_set = u64(n_elem) * sizeof(u32),
            .nsteps      = nsteps,
            .n_chains    = n_chains};
    }

} // namespace sham::benchmarks
