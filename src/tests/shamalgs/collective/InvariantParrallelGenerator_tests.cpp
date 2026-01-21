// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shamalgs/collective/InvariantParrallelGenerator.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <array>

using Gen = shamalgs::collective::InvariantParrallelGenerator<std::mt19937_64>;

void test_next_case(Gen &generator, Gen &generator_ref, u64 count_test_per_rank) {
    std::vector<u64> ref_datased
        = generator_ref.next_n(count_test_per_rank * u64(shamcomm::world_size()), true);
    std::vector<u64> test_data = generator.next_n(count_test_per_rank, false);

    std::vector<u64> collected_data{};
    shamalgs::collective::vector_allgatherv(test_data, collected_data, MPI_COMM_WORLD);

    REQUIRE_EQUAL(ref_datased, collected_data);
    REQUIRE_EQUAL(generator.is_done(), generator_ref.is_done());

    REQUIRE(generator.all_ranks_are_in_sync());
    REQUIRE(generator_ref.all_ranks_are_in_sync());
}

std::vector<u64> benchmark(u64 nval_max, u64 step_size) {
    Gen generator(42, nval_max);

    std::vector<u64> data;

    while (!generator.is_done()) {
        auto tmp = generator.next_n(step_size);
        data.insert(data.end(), tmp.begin(), tmp.end());
    }
    return data;
}

TestStart(
    Unittest,
    "shamalgs/collective/InvariantParrallelGenerator",
    test_invariant_parrallel_generator,
    -1) {

    u64 count_test_per_rank_all = 100_u64;
    u64 count_test              = u64(shamcomm::world_size()) * count_test_per_rank_all;

    u64 seed = 42;

    shamalgs::collective::InvariantParrallelGenerator generator_ref(seed, count_test);
    shamalgs::collective::InvariantParrallelGenerator generator(seed, count_test);

    u64 count_test_per_rank = 10_u64; // 10 steps
    for (u64 i = 0; i < count_test_per_rank_all; i += count_test_per_rank) {
        test_next_case(generator, generator_ref, count_test_per_rank);
    }

    REQUIRE(generator.is_done());
    REQUIRE(generator_ref.is_done());
}

TestStart(
    Benchmark,
    "shamalgs/collective/InvariantParrallelGenerator_benchmark",
    benchmark_invariant_parrallel_generator,
    -1) {

    std::vector<u64> data;
    f64 time = shambase::timeitfor([&]() {
        data = benchmark(10000000 * shamcomm::world_size(), 100000);
    });

    logger::info_ln(
        "InvariantParrallelGenerator_benchmark",
        "time",
        time,
        "rate",
        10000000. * shamcomm::world_size() / time);
}

// core ultra 9 285K
// 1rank -> Info: time 0.1386704975 rate 72113392.39624493
// 2ranks -> time 0.1672155935 rate 119606070.11211546
// 4ranks -> Info: time 0.18985734683333336 rate 210684499.00500336
// 8ranks -> time 0.2366780588 rate 338011898.5495076
