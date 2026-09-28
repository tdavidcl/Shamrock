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
 * @file sort_by_keys_onesweep_radix_sort.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Onesweep LSD radix sort by keys (single pass digit binning with decoupled look-back).
 *
 * Implementation of the Onesweep radix sort (A. Adinets & D. Merrill, "Onesweep: A Faster Least
 * Significant Digit Radix Sort for GPUs", 2022, arXiv:2206.01784), following the structure of
 * https://github.com/Vincenwwx/hpcGPU :
 *
 *  1. an upfront kernel (shamalgs::primitives::digit_histogram) computes the digit histograms of
 *     every digit place in a single read of the keys,
 *  2. for every digit place, a single "chained scan digit binning" kernel is launched. The input
 *     is split in tiles of `tile_size` keys, each work-group :
 *      - takes a tile using a dynamic tile id (so that tile `t` can only wait on tiles that are
 *        already running),
 *      - computes the digit histogram of its tile and publishes it right away as the tile
 *        aggregate (state A) of the look-back lanes, one lane per digit,
 *      - stably sorts the tile in local memory on the current digit (1-bit splits),
 *      - resolves, in parallel over the digits, the exclusive prefix of every digit over the
 *        previous tiles with a decoupled look-back (Merrill & Garland 2016, same tile states as
 *        `ScanTile30bitint` used by the decoupled look-back scan of shamalgs) and publishes its
 *        inclusive prefix (state P),
 *      - scatters the keys to `global digit offset + digit prefix over previous tiles + rank in
 *        the tile`, then the values using the permutation of the local sort.
 *
 * The sort is stable, reads and writes each key and value only once per digit place (the
 * per-pass histogram / scan / scatter of the basic LSD radix sort are fused in a single kernel).
 *
 * The tile states pack the look-back values on 30 bits, the length is therefore limited to
 * `2^30 - 1` elements (callers are expected to fall back to another implementation above).
 */

#include "shambase/aliases_int.hpp"
#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shambase/string.hpp"
#include "shamalgs/details/numeric/scanDecoupledLookback.hpp"
#include "shamalgs/primitives/digit_histogram.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include <type_traits>
#include <algorithm>
#include <limits>
#include <utility>

namespace shamalgs::primitives::device::details {

    /// Maximal length accepted by sort_by_keys_onesweep_radix_sort (look-back values on 30 bits)
    inline constexpr u32 onesweep_radix_sort_max_len = (1U << 30U) - 1U;

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION

    /**
     * @brief Stable Onesweep LSD radix sort of (keys, values) on the first `len` elements
     * (unsigned keys only)
     *
     * @tparam Tkey unsigned integer key type
     * @tparam Tval value type
     * @tparam items_per_thread number of keys handled by each work-item of a tile (the tile size
     * is `256 * items_per_thread`)
     */
    template<class Tkey, class Tval, u32 items_per_thread = 8>
    inline void sort_by_keys_onesweep_radix_sort(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len) {

        static_assert(
            std::is_unsigned_v<Tkey>, "the radix sort is only implemented for unsigned keys");

        constexpr u32 radix_bits = 8;
        constexpr u32 nbuckets   = 1u << radix_bits;
        constexpr u32 digit_mask = nbuckets - 1;
        constexpr u32 npasses    = (sizeof(Tkey) * 8) / radix_bits;
        static_assert(npasses % 2 == 0, "the result is expected in the input buffers");

        // one look-back lane (digit) per work-item
        constexpr u32 group_size = nbuckets;
        constexpr u32 tile_size  = group_size * items_per_thread;

        // padding key of incomplete tiles : all digits are maximal, so padding elements always
        // end up after the valid ones in the (stable) local sort
        constexpr Tkey key_padding = std::numeric_limits<Tkey>::max();

        using Tile = shamalgs::numeric::details::ScanTile30bitint;
        static_assert(Tile::STATE_X == 0, "the tile states are reset with a zero fill");

        using atomic_ref_local = sycl::atomic_ref<
            u32,
            sycl::memory_order_relaxed,
            sycl::memory_scope_work_group,
            sycl::access::address_space::local_space>;

        using atomic_ref_global = sycl::atomic_ref<
            u32,
            sycl::memory_order_relaxed,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>;

        if (len <= 1) {
            return;
        }

        if (len > onesweep_radix_sort_max_len) {
            shambase::throw_with_loc<std::invalid_argument>(sham::format(
                "the onesweep radix sort is limited to {} elements, len = {}",
                onesweep_radix_sort_max_len,
                len));
        }

        if (len > buf_key.get_size() || len > buf_values.get_size()) {
            shambase::throw_with_loc<std::invalid_argument>(sham::format(
                "the buffers are smaller than the length of the sort\n"
                "len = {}, buf_key.get_size() = {}, buf_values.get_size() = {}",
                len,
                buf_key.get_size(),
                buf_values.get_size()));
        }

        auto &dev_prop = sched->ctx->device->prop;
        if (dev_prop.max_work_group_size < group_size) {
            shambase::throw_with_loc<std::runtime_error>(sham::format(
                "the onesweep radix sort requires work-groups of {} work-items, the device "
                "supports at most {}",
                group_size,
                dev_prop.max_work_group_size));
        }

        constexpr usize local_mem_needed
            = tile_size * (sizeof(Tkey) + sizeof(u32)) + (2 * nbuckets + 1) * sizeof(u32);
        if (dev_prop.local_mem_size < local_mem_needed) {
            shambase::throw_with_loc<std::runtime_error>(sham::format(
                "the onesweep radix sort requires {} bytes of local memory, the device has {}",
                local_mem_needed,
                dev_prop.local_mem_size));
        }

        u32 ntiles = shambase::group_count(len, tile_size);

        sham::DeviceQueue &q = sched->get_queue();

        sham::DeviceBuffer<Tkey> key_tmp(len, sched);
        sham::DeviceBuffer<Tval> val_tmp(len, sched);

        // digit histograms of every digit place (pass major, see digit_histogram)
        sham::DeviceBuffer<u32> digit_hist(0, sched);
        // look-back tile states, index `tile * nbuckets + digit`
        sham::DeviceBuffer<u32> tile_states(ntiles * nbuckets, sched);
        // dynamic tile id counter
        sham::DeviceBuffer<u32> tile_counter(1, sched);

        ////////////////////////////////////////////////////////////////////////////////////////
        // 1. upfront histograms of all the digit places
        ////////////////////////////////////////////////////////////////////////////////////////

        shamalgs::primitives::digit_histogram<Tkey, radix_bits>(sched, buf_key, digit_hist, len);

        ////////////////////////////////////////////////////////////////////////////////////////
        // 2. one chained scan digit binning kernel per digit place
        ////////////////////////////////////////////////////////////////////////////////////////

        sham::DeviceBuffer<Tkey> *key_in  = &buf_key;
        sham::DeviceBuffer<Tkey> *key_out = &key_tmp;
        sham::DeviceBuffer<Tval> *val_in  = &buf_values;
        sham::DeviceBuffer<Tval> *val_out = &val_tmp;

        for (u32 pass = 0; pass < npasses; pass++) {
            u32 shift = pass * radix_bits;

            tile_states.fill(0); // Tile::STATE_X
            tile_counter.fill(0);

            sham::kernel_call_hndl(
                q,
                sham::MultiRef{*key_in, *val_in, digit_hist},
                sham::MultiRef{*key_out, *val_out, tile_states, tile_counter},
                ntiles * group_size,
                [=](u32 nthreads,
                    const Tkey *__restrict keys,
                    const Tval *__restrict vals,
                    const u32 *__restrict hist,
                    Tkey *__restrict keys_out,
                    Tval *__restrict vals_out,
                    u32 *__restrict states,
                    u32 *__restrict counter) {
                    return [=](sycl::handler &cgh) {
                        sycl::local_accessor<Tkey, 1> l_keys{tile_size, cgh};
                        sycl::local_accessor<u32, 1> l_idx{tile_size, cgh};
                        sycl::local_accessor<u32, 1> l_count{nbuckets, cgh};
                        sycl::local_accessor<u32, 1> l_offset{nbuckets, cgh};
                        sycl::local_accessor<u32, 1> l_tile_id{1, cgh};

                        cgh.parallel_for(
                            sycl::nd_range<1>{nthreads, group_size}, [=](sycl::nd_item<1> item) {
                                u32 lid = item.get_local_id(0);
                                auto g  = item.get_group();

                                auto get_digit = [shift](Tkey k) -> u32 {
                                    return u32(k >> shift) & digit_mask;
                                };

                                // dynamic tile id : tiles are taken in increasing order, so that
                                // every tile we look back on is already owned by a running group
                                if (lid == 0) {
                                    l_tile_id[0] = atomic_ref_global(counter[0]).fetch_add(1U);
                                }
                                l_count[lid] = 0;
                                item.barrier(sycl::access::fence_space::local_space);

                                u32 tile_id    = l_tile_id[0];
                                u32 tile_begin = tile_id * tile_size;
                                u32 tile_len   = sham::min(tile_size, len - tile_begin);

                                // coalesced load of the tile & digit histogram of the tile
                                for (u32 j = 0; j < items_per_thread; j++) {
                                    u32 li     = j * group_size + lid;
                                    bool valid = li < tile_len;
                                    Tkey k     = valid ? keys[tile_begin + li] : key_padding;
                                    l_keys[li] = k;
                                    l_idx[li]  = li;
                                    if (valid) {
                                        atomic_ref_local(l_count[get_digit(k)]).fetch_add(1U);
                                    }
                                }
                                item.barrier(sycl::access::fence_space::local_space);

                                // publish the tile aggregate of the look-back lane `lid` (digit)
                                // as soon as possible so that the next tiles can progress
                                u32 count = l_count[lid];
                                atomic_ref_global tile_state(states[tile_id * nbuckets + lid]);
                                if (tile_id == 0) {
                                    tile_state.store(Tile::pack(Tile::STATE_P, count));
                                } else {
                                    tile_state.store(Tile::pack(Tile::STATE_A, count));
                                }

                                // stable local sort of the tile on the current digit, one
                                // 1-bit split per bit (blocked arrangement : work-item `lid` owns
                                // the local positions [lid * items_per_thread, ...[)
                                Tkey rk[items_per_thread];
                                u32 ri[items_per_thread];
                                for (u32 b = 0; b < radix_bits; b++) {
                                    u32 bit = shift + b;

                                    u32 zeros = 0;
                                    for (u32 j = 0; j < items_per_thread; j++) {
                                        rk[j] = l_keys[lid * items_per_thread + j];
                                        ri[j] = l_idx[lid * items_per_thread + j];
                                        zeros += ((rk[j] >> bit) & 1U) ? 0U : 1U;
                                    }

                                    u32 zeros_before = sycl::exclusive_scan_over_group(
                                        g, zeros, sycl::plus<u32>{});
                                    u32 total_zeros
                                        = sycl::reduce_over_group(g, zeros, sycl::plus<u32>{});

                                    // every work-item must be done reading the tile
                                    item.barrier(sycl::access::fence_space::local_space);

                                    for (u32 j = 0; j < items_per_thread; j++) {
                                        u32 pos  = lid * items_per_thread + j;
                                        bool one = (rk[j] >> bit) & 1U;
                                        // zeros keep their order at the front, ones at the back
                                        u32 dst = one ? total_zeros + (pos - zeros_before)
                                                      : zeros_before;
                                        zeros_before += one ? 0U : 1U;
                                        l_keys[dst] = rk[j];
                                        l_idx[dst]  = ri[j];
                                    }
                                    item.barrier(sycl::access::fence_space::local_space);
                                }

                                // decoupled look-back : exclusive prefix of the digit `lid` over
                                // the previous tiles
                                u32 exclusive = 0;
                                if (tile_id > 0) {
                                    u32 tile_ptr = tile_id - 1;
                                    while (true) {
                                        atomic_ref_global prev_state(
                                            states[tile_ptr * nbuckets + lid]);

                                        Tile s = Tile::invalid();
                                        do {
                                            s = Tile::unpack(prev_state.load());
                                        } while (s.is_invalid());

                                        exclusive += s.get_prefix();

                                        if (s.has_prefix_available()) {
                                            break;
                                        }
                                        // tile 0 always publishes a prefix, so no underflow here
                                        tile_ptr--;
                                    }
                                    tile_state.store(Tile::pack(Tile::STATE_P, exclusive + count));
                                }

                                // output position of the element at local sorted position `li` of
                                // digit `d` : global digit offset + look-back prefix + rank of the
                                // element among the digit `d` of the tile (li - local digit start)
                                //
                                // global start - local start is computed with a single scan of
                                // (global count - tile count) : the partial sums wrap around, but
                                // the final sum is correct in modular arithmetic. Two consecutive
                                // group scans must be avoided : AdaptiveCpp's work-group scan on
                                // CUDA has no trailing barrier, so a fast warp entering the second
                                // scan can overwrite the shared scratch of the first one before a
                                // slower warp has read its prefix.
                                u32 start_diff = sycl::exclusive_scan_over_group(
                                    g, hist[pass * nbuckets + lid] - count, sycl::plus<u32>{});
                                l_offset[lid] = start_diff + exclusive;
                                item.barrier(sycl::access::fence_space::local_space);

                                // scatter (padding elements are at the end of the sorted tile)
                                for (u32 j = 0; j < items_per_thread; j++) {
                                    u32 li = j * group_size + lid;
                                    if (li < tile_len) {
                                        Tkey k        = l_keys[li];
                                        u32 dst       = l_offset[get_digit(k)] + li;
                                        keys_out[dst] = k;
                                        vals_out[dst] = vals[tile_begin + l_idx[li]];
                                    }
                                }
                            });
                    };
                });

            std::swap(key_in, key_out);
            std::swap(val_in, val_out);
        }
    }

#endif

} // namespace shamalgs::primitives::device::details
