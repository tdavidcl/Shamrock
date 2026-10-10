// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/Timer.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shamrock/scheduler/loadbalance/LoadBalanceStrategy.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <random>
#include <utility>

using namespace shamrock::scheduler;
using Tweight = u64;
using Torder  = u64;
using LBTile  = TileWithLoad<Torder, Tweight>;

void add_strategy_plot(
    std::string strat_name,
    std::string filename,
    const std::vector<LBTile> &vec_test,
    const std::vector<i32> &result,
    i32 wsize) {

    std::vector<f64> load_values;
    std::vector<f64> order_values;
    std::vector<i32> node_owner;

    for (u32 i = 0; i < vec_test.size(); i++) {
        load_values.push_back(f64(vec_test[i].load_value));
        order_values.push_back(f64(vec_test[i].ordering_val));
        node_owner.push_back(result[i]);
    }

    PyScriptHandle hndl{};

    hndl.data()["wsize"]      = wsize;
    hndl.data()["loads"]      = load_values;
    hndl.data()["order"]      = order_values;
    hndl.data()["node_owner"] = node_owner;
    hndl.data()["filename"]   = std::string("tests/figures/load_balance_strat") + filename + ".pdf";
    hndl.data()["strat_name"] = strat_name;

    hndl.exec(R"py(

        import matplotlib.pyplot as plt
        import numpy as np

        plt.close('all')

        range_wsize = range(wsize)

        ptch_lst_node = [[] for i in range_wsize]

        for load,ord,own in zip(loads, order, node_owner):

            ptch_lst_node[own].append(load)

        lens = [len(i) for i in ptch_lst_node]
        mx = max(lens)

        for i in ptch_lst_node:
            while len(i) < mx:
                i.append(0)


        bar_lst = np.transpose(ptch_lst_node)

        cummul = np.array([0. for j in range_wsize])

        for bindex in range(len(bar_lst)):
            i = bar_lst[bindex]
            plt.bar(range(len(i)), i, bottom=cummul)

            cummul += np.array(i)

        plt.xlabel("nodes id")
        plt.ylabel("load")
        plt.title(strat_name)

        plt.savefig(filename)

        plt.close('all')

    )py");

    TEX_REPORT(
        R"tex(

        \begin{figure}[ht!]
        \center
        \includegraphics[width=0.95\linewidth]{figures/load_balance_strat)tex"
        + filename + R"tex(.pdf}
        \caption{Load balancing strategy}
        \end{figure}

    )tex")
}

NEW_TEST(TestType::ValidationTest, "shamrock/scheduler/loadbalance", 1) {

    i32 fake_world_size = 64;

    auto make_tile_list = [](u32 count, u64 min_load, u64 max_load) -> std::vector<LBTile> {
        std::vector<LBTile> res;
        std::mt19937 eng{0x111};

        for (u32 i = 0; i < count; i++) {
            res.push_back(
                LBTile{
                    .ordering_val = shamalgs::primitives::mock_value(eng, 0_u64, u64_max),
                    .load_value   = shamalgs::primitives::mock_value(eng, min_load, max_load),
                });
        }

        return res;
    };

    std::vector<LBTile> vec_test = make_tile_list(64 * 4, 1000000, 1200000);

    std::vector<i32> result1     = details::lb_startegy_parallel_sweep(vec_test, fake_world_size);
    std::vector<i32> result2     = details::lb_startegy_roundrobin(vec_test, fake_world_size);
    std::vector<i32> result_best = load_balance(std::vector(vec_test), fake_world_size);

    add_strategy_plot("parallel sweep", "psweep", vec_test, result1, fake_world_size);
    add_strategy_plot("round robin", "rrobin", vec_test, result2, fake_world_size);
    add_strategy_plot("best", "rrobin", vec_test, result_best, fake_world_size);
}

namespace {

    /**
     * @brief Copy of the load balancing implementation before the tiles were sorted only once
     * in load_balance (debug logs removed), kept as a reference for the current one.
     *
     * With stable_ordering = true, ties on the ordering value keep the original index order,
     * which is how the current implementation breaks them. Otherwise the plain std::sort of the
     * original code is used, so the result is only defined for distinct ordering values.
     */
    namespace reference_lb {

        template<class Torder, class Tweight>
        struct LoadBalancedTile {
            Torder ordering_val;
            Tweight load_value;
            Tweight accumulated_load_value;
            u64 index;
            i32 new_owner;

            LoadBalancedTile() = default;

            LoadBalancedTile(TileWithLoad<Torder, Tweight> in, u64 inindex)
                : ordering_val(in.ordering_val), load_value(in.load_value), index(inindex) {}
        };

        template<bool stable_ordering, class Torder, class Tweight>
        inline void apply_ordering(std::vector<LoadBalancedTile<Torder, Tweight>> &lb_vec) {
            using LBTileResult = LoadBalancedTile<Torder, Tweight>;
            auto comp          = [](const LBTileResult &left, const LBTileResult &right) {
                return left.ordering_val < right.ordering_val;
            };
            if constexpr (stable_ordering) {
                std::stable_sort(lb_vec.begin(), lb_vec.end(), comp);
            } else {
                std::sort(lb_vec.begin(), lb_vec.end(), comp);
            }
        }

        template<bool stable_ordering, class Torder, class Tweight>
        inline std::vector<i32> lb_startegy_parallel_sweep(
            const std::vector<TileWithLoad<Torder, Tweight>> &lb_vector, i32 wsize) {

            using LBTileResult = LoadBalancedTile<Torder, Tweight>;

            std::vector<LBTileResult> res(lb_vector.size());
#pragma omp parallel for
            for (u64 i = 0; i < lb_vector.size(); i++) {
                res[i] = LBTileResult{lb_vector[i], i};
            }

            // apply the ordering
            apply_ordering<stable_ordering>(res);

            // compute increments for load
            u64 accum = 0;
            for (LBTileResult &tile : res) {
                u64 cur_val                 = tile.load_value;
                tile.accumulated_load_value = accum;
                accum += cur_val;
            }

            double target_datacnt = double(res[res.size() - 1].accumulated_load_value) / wsize;

#pragma omp parallel for
            for (u64 i = 0; i < res.size(); i++) {
                LBTileResult &tile = res[i];
                tile.new_owner
                    = (target_datacnt == 0)
                          ? 0
                          : sycl::clamp(
                                i32(tile.accumulated_load_value / target_datacnt), 0, wsize - 1);
            }

            std::vector<i32> new_owners(res.size());
            for (LBTileResult &tile : res) {
                new_owners[tile.index] = tile.new_owner;
            }

            return new_owners;
        }

        template<bool stable_ordering, class Torder, class Tweight>
        inline std::vector<i32> lb_startegy_roundrobin(
            const std::vector<TileWithLoad<Torder, Tweight>> &lb_vector, i32 wsize) {

            using LBTileResult = LoadBalancedTile<Torder, Tweight>;

            std::vector<LBTileResult> res(lb_vector.size());
#pragma omp parallel for
            for (u64 i = 0; i < lb_vector.size(); i++) {
                res[i] = LBTileResult{lb_vector[i], i};
            }

            // apply the ordering
            apply_ordering<stable_ordering>(res);

            // compute increments for load
            u64 accum = 0;
            for (LBTileResult &tile : res) {
                tile.accumulated_load_value = accum;
                accum += 1;
            }

            double target_datacnt = double(res[res.size() - 1].accumulated_load_value) / wsize;

#pragma omp parallel for
            for (u64 i = 0; i < res.size(); i++) {
                LBTileResult &tile = res[i];
                tile.new_owner
                    = (target_datacnt == 0)
                          ? 0
                          : sycl::clamp(
                                i32(tile.accumulated_load_value / target_datacnt), 0, wsize - 1);
            }

            std::vector<i32> new_owners(res.size());
            for (LBTileResult &tile : res) {
                new_owners[tile.index] = tile.new_owner;
            }

            return new_owners;
        }

        template<bool stable_ordering, class Torder, class Tweight>
        inline std::vector<i32> load_balance(
            std::vector<TileWithLoad<Torder, Tweight>> &&lb_vector, i32 world_size) {

            using namespace details;

            f64 factor_boost_psweep = 1;
            auto tmpres = lb_startegy_parallel_sweep<stable_ordering>(lb_vector, world_size);
            auto metric_psweep
                = compute_LB_metric(lb_vector, tmpres, world_size, factor_boost_psweep);

            f64 factor_boost_rrobin = 0.95;
            auto tmpres_2 = lb_startegy_roundrobin<stable_ordering>(lb_vector, world_size);
            auto metric_rrobin
                = compute_LB_metric(lb_vector, tmpres_2, world_size, factor_boost_rrobin);

            if (metric_rrobin.max < metric_psweep.max) {
                tmpres = tmpres_2;
            }

            return tmpres;
        }

    } // namespace reference_lb

    /// bijective u64 mixer (splitmix64 finalizer), gives distinct pseudo random keys
    inline u64 mix_u64(u64 x) {
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9_u64;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb_u64;
        return x ^ (x >> 31);
    }

    enum class KeyKind { Distinct, ManyTies };
    enum class LoadKind { Narrow, SmallWithZeros, AllZeros, HeavyTailed };

    template<class Torder>
    Torder make_key(std::mt19937_64 &eng, u64 i, u64 seed, KeyKind kind) {
        u64 key = (kind == KeyKind::Distinct) ? mix_u64(i ^ seed) : (eng() % 16);
        if constexpr (std::is_same_v<Torder, u64>) {
            return key;
        } else {
            // quad hilbert like key, the first component carries ties on purpose
            return Torder{eng() % 4, key};
        }
    }

    u64 make_load(std::mt19937_64 &eng, LoadKind kind) {
        switch (kind) {
        case LoadKind::Narrow        : return 1000000 + eng() % 200001;
        case LoadKind::SmallWithZeros: return eng() % 11;
        case LoadKind::AllZeros      : return 0;
        case LoadKind::HeavyTailed: return (eng() % 50 == 0) ? 1 + eng() % 100000000 : eng() % 1000;
        }
        return 0;
    }

    template<class Torder>
    std::vector<TileWithLoad<Torder, u64>> make_random_tiles(
        u64 count, u64 seed, KeyKind key_kind, LoadKind load_kind) {
        std::mt19937_64 eng{seed};
        std::vector<TileWithLoad<Torder, u64>> res(count);
        for (u64 i = 0; i < count; i++) {
            res[i] = {make_key<Torder>(eng, i, seed, key_kind), make_load(eng, load_kind)};
        }
        return res;
    }

    template<class Torder>
    void check_lb_against_reference(KeyKind key_kind) {
        using Tile = TileWithLoad<Torder, u64>;

        std::vector<u64> counts  = {1, 2, 7, 256, 1000, 10007, 100000};
        std::vector<i32> wsizes  = {1, 3, 64, 1000};
        std::vector<LoadKind> lk = {
            LoadKind::Narrow, LoadKind::SmallWithZeros, LoadKind::AllZeros, LoadKind::HeavyTailed};

        u64 seed = 0x111;
        for (u64 count : counts) {
            for (i32 wsize : wsizes) {
                for (LoadKind load_kind : lk) {
                    seed++;
                    std::vector<Tile> tiles
                        = make_random_tiles<Torder>(count, seed, key_kind, load_kind);

                    auto check = [&](auto stable_ordering) {
                        constexpr bool stable = decltype(stable_ordering)::value;

                        std::string case_name = sham::format(
                            "count={} wsize={} load_kind={} stable_ref={}",
                            count,
                            wsize,
                            i32(load_kind),
                            stable);

                        std::vector<i32> psweep = details::lb_startegy_parallel_sweep(tiles, wsize);
                        std::vector<i32> psweep_ref
                            = reference_lb::lb_startegy_parallel_sweep<stable>(tiles, wsize);
                        REQUIRE_NAMED("psweep " + case_name, psweep == psweep_ref);

                        std::vector<i32> rrobin = details::lb_startegy_roundrobin(tiles, wsize);
                        std::vector<i32> rrobin_ref
                            = reference_lb::lb_startegy_roundrobin<stable>(tiles, wsize);
                        REQUIRE_NAMED("rrobin " + case_name, rrobin == rrobin_ref);

                        std::vector<i32> best = load_balance(std::vector(tiles), wsize);
                        std::vector<i32> best_ref
                            = reference_lb::load_balance<stable>(std::vector(tiles), wsize);
                        REQUIRE_NAMED("load_balance " + case_name, best == best_ref);
                    };

                    if (key_kind == KeyKind::Distinct) {
                        // distinct keys: the original std::sort based code is well defined
                        check(std::false_type{});
                    }
                    // ties are broken by the original index
                    check(std::true_type{});
                }
            }
        }
    }

} // namespace

NEW_TEST(TestType::Unittest, "shamrock/scheduler/loadbalance:compare_reference", 1) {
    check_lb_against_reference<u64>(KeyKind::Distinct);
    check_lb_against_reference<u64>(KeyKind::ManyTies);
    check_lb_against_reference<std::pair<u64, u64>>(KeyKind::Distinct);
    check_lb_against_reference<std::pair<u64, u64>>(KeyKind::ManyTies);
}

NEW_TEST(TestType::Benchmark, "shamrock/scheduler/loadbalance:benchmark", 1) {

    // ~800k patches over ~100k ranks, with shuffled hilbert keys
    u64 count = 800000;
    i32 wsize = 98304;
    u32 nrep  = 7;

    std::vector<TileWithLoad<u64, u64>> tiles
        = make_random_tiles<u64>(count, 0x5eed, KeyKind::Distinct, LoadKind::Narrow);

    auto bench = [&](auto &&func) -> std::pair<f64, std::vector<i32>> {
        std::vector<f64> times;
        std::vector<i32> res;
        for (u32 i = 0; i < nrep; i++) {
            std::vector<TileWithLoad<u64, u64>> in = tiles;
            shambase::Timer timer;
            timer.start();
            res = func(std::move(in));
            timer.stop();
            times.push_back(timer.elapsed_sec());
        }
        std::sort(times.begin(), times.end());
        return {times[times.size() / 2], res};
    };

    auto [t_ref, res_ref] = bench([&](std::vector<TileWithLoad<u64, u64>> &&in) {
        return reference_lb::load_balance<false>(std::move(in), wsize);
    });
    auto [t_new, res_new] = bench([&](std::vector<TileWithLoad<u64, u64>> &&in) {
        return load_balance(std::move(in), wsize);
    });

    logger::raw_ln(
        sham::format(
            "load_balance on {} tiles, {} ranks (median of {}) : old = {:.2f} ms, new = {:.2f} "
            "ms, speedup = {:.2f}",
            count,
            wsize,
            nrep,
            t_ref * 1e3,
            t_new * 1e3,
            t_ref / t_new));

    REQUIRE(res_ref == res_new);
}
