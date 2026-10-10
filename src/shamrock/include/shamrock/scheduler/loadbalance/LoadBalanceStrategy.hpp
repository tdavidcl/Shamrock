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
 * @file LoadBalanceStrategy.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief implementation of the hilbert curve load balancing
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include <algorithm>
#include <vector>

// libstdc++ parallel mode sort (OpenMP multiway mergesort), used for the tiles sort when OpenMP
// is enabled
#if defined(_OPENMP) && defined(__GLIBCXX__)
    #define SHAMROCK_LB_USE_GNU_PARALLEL_SORT
    #if defined(__clang__)
        #pragma clang diagnostic push
        // libstdc++ parallel mode still uses std::binary_function
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    #endif
    #include <parallel/algorithm>
    #if defined(__clang__)
        #pragma clang diagnostic pop
    #endif
#endif

namespace shamrock::scheduler {
    template<class Torder, class Tweight>
    struct TileWithLoad {
        Torder ordering_val;
        Tweight load_value;
    };
} // namespace shamrock::scheduler

namespace shamrock::scheduler::details {

    template<class Torder, class Tweight>
    struct LoadBalancedTile {
        Torder ordering_val;
        Tweight load_value;
        u64 index;

        LoadBalancedTile() = default;

        LoadBalancedTile(TileWithLoad<Torder, Tweight> in, u64 inindex)
            : ordering_val(in.ordering_val), load_value(in.load_value), index(inindex) {}
    };

    /**
     * @brief Sort tiles by their ordering value
     *
     * Ties on the ordering value are broken by the original tile index, so the comparison is a
     * strict total order and the sorted order is unique. Every rank computes the load balancing
     * redundantly, so this keeps the result independent of the sort algorithm and of the
     * number of threads used to sort.
     *
     * @tparam Torder Ordering value type
     * @tparam Tweight Load weight type
     * @param lb_vec Vector of load-balanced tiles to sort
     */
    template<class Torder, class Tweight>
    inline void apply_ordering(std::vector<LoadBalancedTile<Torder, Tweight>> &lb_vec) {
        using LBTileResult = LoadBalancedTile<Torder, Tweight>;
        auto comp          = [](const LBTileResult &left, const LBTileResult &right) {
            if (left.ordering_val < right.ordering_val) {
                return true;
            }
            if (right.ordering_val < left.ordering_val) {
                return false;
            }
            return left.index < right.index;
        };
#ifdef SHAMROCK_LB_USE_GNU_PARALLEL_SORT
        __gnu_parallel::sort(lb_vec.begin(), lb_vec.end(), comp);
#else
        std::sort(lb_vec.begin(), lb_vec.end(), comp);
#endif
    }

    /**
     * @brief Build the tiles list sorted by ordering value
     *
     * The result can be shared by all strategies, which all walk the tiles in this order.
     *
     * @tparam Torder Ordering value type
     * @tparam Tweight Load weight type
     * @param lb_vector Tiles with load information
     * @return std::vector<LoadBalancedTile<Torder, Tweight>> Tiles sorted by ordering value
     */
    template<class Torder, class Tweight>
    inline std::vector<LoadBalancedTile<Torder, Tweight>> make_sorted_tiles(
        const std::vector<TileWithLoad<Torder, Tweight>> &lb_vector) {

        using LBTileResult = LoadBalancedTile<Torder, Tweight>;

        std::vector<LBTileResult> res(lb_vector.size());
#pragma omp parallel for
        for (u64 i = 0; i < lb_vector.size(); i++) {
            res[i] = LBTileResult{lb_vector[i], i};
        }

        apply_ordering(res);

        return res;
    }

    /**
     * @brief Assign owners by cutting the sorted tiles into world size equal accumulated loads
     *
     * @param sorted_tiles Tiles sorted by ordering value
     * @param wsize Number of workers
     * @param get_accumulated Returns the accumulated load of the i-th sorted tile (exclusive
     * prefix sum of the load used by the strategy)
     * @return std::vector<i32> New owner assignments for each tile (in the original order)
     */
    template<class Torder, class Tweight, class Fget>
    inline std::vector<i32> assign_owners_sorted(
        const std::vector<LoadBalancedTile<Torder, Tweight>> &sorted_tiles,
        i32 wsize,
        Fget &&get_accumulated) {

        std::vector<i32> new_owners(sorted_tiles.size());

        if (sorted_tiles.empty()) {
            return new_owners;
        }

        double target_datacnt = double(get_accumulated(sorted_tiles.size() - 1)) / wsize;

#pragma omp parallel for
        for (u64 i = 0; i < sorted_tiles.size(); i++) {
            Tweight accumulated_load_value = get_accumulated(i);
            new_owners[sorted_tiles[i].index]
                = (target_datacnt == 0)
                      ? 0
                      : sycl::clamp(i32(accumulated_load_value / target_datacnt), 0, wsize - 1);
        }

        if (shamcomm::world_rank() == 0
            && shamcomm::logs::get_loglevel() >= shamcomm::logs::log_debug) {
            for (u64 i = 0; i < sorted_tiles.size(); i++) {
                const auto &t                  = sorted_tiles[i];
                Tweight accumulated_load_value = get_accumulated(i);
                shamlog_debug_ln(
                    "HilbertLoadBalance",
                    t.ordering_val,
                    accumulated_load_value,
                    t.index,
                    (target_datacnt == 0)
                        ? 0
                        : sycl::clamp(
                              i32(accumulated_load_value / target_datacnt), 0, i32(wsize) - 1),
                    (target_datacnt == 0) ? 0 : (accumulated_load_value / target_datacnt));
            }
        }

        return new_owners;
    }

    /**
     * @brief Parallel sweep strategy on already sorted tiles
     *
     * @tparam Torder Ordering value type
     * @tparam Tweight Load weight type
     * @param sorted_tiles Tiles sorted by ordering value (see make_sorted_tiles)
     * @param wsize Number of workers
     * @return std::vector<i32> New owner assignments for each tile
     */
    template<class Torder, class Tweight>
    inline std::vector<i32> lb_startegy_parallel_sweep_sorted(
        const std::vector<LoadBalancedTile<Torder, Tweight>> &sorted_tiles, i32 wsize) {

        // compute increments for load
        std::vector<Tweight> accumulated_load(sorted_tiles.size());
        u64 accum = 0;
        for (u64 i = 0; i < sorted_tiles.size(); i++) {
            u64 cur_val         = sorted_tiles[i].load_value;
            accumulated_load[i] = accum;
            accum += cur_val;
        }

        return assign_owners_sorted(sorted_tiles, wsize, [&](u64 i) -> Tweight {
            return accumulated_load[i];
        });
    }

    /**
     * @brief Round-robin strategy on already sorted tiles
     *
     * @tparam Torder Ordering value type
     * @tparam Tweight Load weight type
     * @param sorted_tiles Tiles sorted by ordering value (see make_sorted_tiles)
     * @param wsize Number of workers
     * @return std::vector<i32> New owner assignments for each tile
     */
    template<class Torder, class Tweight>
    inline std::vector<i32> lb_startegy_roundrobin_sorted(
        const std::vector<LoadBalancedTile<Torder, Tweight>> &sorted_tiles, i32 wsize) {

        // assume that each patch has the same load, which effectivelly does a round robin
        // balancing, the accumulated load of the i-th tile is then i
        return assign_owners_sorted(sorted_tiles, wsize, [](u64 i) -> Tweight {
            return i;
        });
    }

    /**
     * @brief Load balance using parallel sweep strategy based on accumulated load
     *
     * @tparam Torder Ordering value type
     * @tparam Tweight Load weight type
     * @param lb_vector Tiles with load information
     * @param wsize Number of workers
     * @return std::vector<i32> New owner assignments for each tile
     */
    template<class Torder, class Tweight>
    inline std::vector<i32> lb_startegy_parallel_sweep(
        const std::vector<TileWithLoad<Torder, Tweight>> &lb_vector, i32 wsize) {
        return lb_startegy_parallel_sweep_sorted(make_sorted_tiles(lb_vector), wsize);
    }

    /**
     * @brief Load balance using round-robin strategy ignoring actual load values
     *
     * @tparam Torder Ordering value type
     * @tparam Tweight Load weight type
     * @param lb_vector Tiles with load information
     * @param wsize Number of workers
     * @return std::vector<i32> New owner assignments for each tile
     */
    template<class Torder, class Tweight>
    inline std::vector<i32> lb_startegy_roundrobin(
        const std::vector<TileWithLoad<Torder, Tweight>> &lb_vector, i32 wsize) {
        return lb_startegy_roundrobin_sorted(make_sorted_tiles(lb_vector), wsize);
    }

    struct LBMetric {
        f64 min;
        f64 max;
        f64 mean;
        f64 stddev;
    };

    /**
     * @brief Compute load balance quality metrics
     *
     * @tparam Torder Ordering value type
     * @tparam Tweight Load weight type
     * @param lb_vector Tiles with load information
     * @param new_owners Owner assignments to evaluate
     * @param world_size Number of workers
     * @return LBMetric Statistics about load distribution (min, max, mean, stddev)
     */
    template<class Torder, class Tweight>
    inline LBMetric compute_LB_metric(
        const std::vector<TileWithLoad<Torder, Tweight>> &lb_vector,
        const std::vector<i32> &new_owners,
        i32 world_size,
        f64 strategy_weight) {

        std::vector<u64> load_per_node(world_size, 0);

        for (u64 i = 0; i < lb_vector.size(); i++) {
            load_per_node[new_owners[i]] += lb_vector[i].load_value;
        }

        f64 min = shambase::VectorProperties<f64>::get_inf();
        f64 max = -shambase::VectorProperties<f64>::get_inf();
        f64 avg = 0;
        f64 var = 0;

        for (i32 nid = 0; nid < world_size; nid++) {
            f64 val = load_per_node[nid];
            min     = sycl::fmin(min, val);
            max     = sycl::fmax(max, val);
            avg += val;

            // shamlog_debug_ln("HilbertLoadBalance", "node :",nid, "load :",load_per_node[nid]);
        }
        avg /= world_size;
        for (i32 nid = 0; nid < world_size; nid++) {
            f64 val = load_per_node[nid];
            var += (val - avg) * (val - avg);
        }
        var /= world_size;

        return {
            .min    = min * strategy_weight,
            .max    = max * strategy_weight,
            .mean   = avg * strategy_weight,
            .stddev = sycl::sqrt(var) * strategy_weight};
    }

} // namespace shamrock::scheduler::details

namespace shamrock::scheduler {

    /**
     * @brief load balance the input vector
     *
     * @tparam Torder ordering value (hilbert, morton, ...)
     * @tparam Tweight weight type
     * @param lb_vector
     * @return std::vector<i32> The new owner list
     */
    template<class Torder, class Tweight>
    inline std::vector<i32> load_balance(
        std::vector<TileWithLoad<Torder, Tweight>> &&lb_vector,
        i32 world_size = shamcomm::world_size()) {

        using namespace details;

        // both strategies walk the tiles in the same order, sort them only once
        auto sorted_tiles = make_sorted_tiles(lb_vector);

        f64 factor_boost_psweep = 1;
        auto tmpres             = lb_startegy_parallel_sweep_sorted(sorted_tiles, world_size);
        auto metric_psweep = compute_LB_metric(lb_vector, tmpres, world_size, factor_boost_psweep);

        // We boost the round robin strategy to favor it if the difference is around 5% since the
        // increased uniformity will probably offset the cost anyway
        f64 factor_boost_rrobin = 0.95;
        auto tmpres_2           = lb_startegy_roundrobin_sorted(sorted_tiles, world_size);
        auto metric_rrobin
            = compute_LB_metric(lb_vector, tmpres_2, world_size, factor_boost_rrobin);

        std::string strategy_name = "parallel sweep";
        if (metric_rrobin.max < metric_psweep.max) {
            tmpres        = std::move(tmpres_2);
            strategy_name = "round robin";
        }

        if (shamcomm::world_rank() == 0) {
            logger::info_ln(
                "LoadBalance",
                sham::format(
                    R"=(Summary (strategy = {0:}):
 - strategy "psweep"      : max = {1:.1f} min = {2:.1f} factor = {3:}
 - strategy "round robin" : max = {4:.1f} min = {5:.1f} factor = {6:})=",
                    strategy_name,
                    metric_psweep.max,
                    metric_psweep.min,
                    factor_boost_psweep,
                    metric_rrobin.max,
                    metric_rrobin.min,
                    factor_boost_rrobin));
        }
        return tmpres;
    }

} // namespace shamrock::scheduler
