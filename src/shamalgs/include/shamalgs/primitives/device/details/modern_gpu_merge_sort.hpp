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
 * @file modern_gpu_merge_sort.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/integer.hpp"
#include "shamalgs/primitives/workitem/odd_even_transpose_sort.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace shamalgs::primitives::device::details {

    namespace debug {

        inline constexpr const char *ansi_green = "\033[32m";
        inline constexpr const char *ansi_red   = "\033[31m";
        inline constexpr const char *ansi_reset = "\033[0m";

        /// Flags indices that continue an ascending run from the previous element (mirrors
        /// `_sorted_run_mask` in examples/benchmarks/run_sort_by_keys_tmp_dev.py).
        template<class T>
        inline std::vector<bool> sorted_run_mask(const std::vector<T> &values) {
            std::vector<bool> mask(values.size(), false);
            for (size_t i = 0; i < values.size(); ++i) {
                mask[i] = (i == 0) || (values[i - 1] < values[i]);
            }
            return mask;
        }

        template<class T>
        inline std::string to_str(const T &v) {
            std::ostringstream oss;
            oss << v;
            return oss.str();
        }

    } // namespace debug

    /**
     * @brief Debug pretty-printer for key/value arrays, mirroring `print_key_val_table` in
     * examples/benchmarks/run_sort_by_keys_tmp_dev.py.
     *
     * Prints the indices, the keys (highlighted green if ascending from the previous key, red
     * otherwise), and the values as an aligned table.
     *
     * @tparam Tkey Type of the keys.
     * @tparam Tval Type of the values.
     * @param keys Host-side keys to print.
     * @param vals Host-side values to print.
     * @param idx_label Label of the index row.
     * @param key_label Label of the key row.
     * @param val_label Label of the value row.
     */
    template<class Tkey, class Tval>
    inline void print_key_val_table(
        const std::vector<Tkey> &keys,
        const std::vector<Tval> &vals,
        const std::string &idx_label = "i",
        const std::string &key_label = "key",
        const std::string &val_label = "val") {

        using namespace debug;

        size_t n = keys.size();

        std::vector<std::string> idx_str(n), key_str(n), val_str(n);
        size_t col_width = 0;
        for (size_t i = 0; i < n; ++i) {
            idx_str[i] = to_str(i);
            key_str[i] = to_str(keys[i]);
            val_str[i] = to_str(vals[i]);
            col_width
                = std::max({col_width, idx_str[i].size(), key_str[i].size(), val_str[i].size()});
        }

        size_t label_width = std::max({idx_label.size(), key_label.size(), val_label.size()});

        auto pad_label = [&](const std::string &label) {
            std::string s = label;
            s.resize(label_width, ' ');
            return s;
        };

        auto pad_cell = [&](const std::string &s) {
            return std::string(col_width - s.size(), ' ') + s;
        };

        auto fmt_row = [&](const std::string &label,
                           const std::vector<std::string> &values,
                           const std::vector<bool> *mask) {
            std::string line = pad_label(label) + " | ";
            for (size_t i = 0; i < values.size(); ++i) {
                if (i > 0) {
                    line += "  ";
                }
                std::string cell = pad_cell(values[i]);
                if (mask != nullptr) {
                    cell = std::string((*mask)[i] ? ansi_green : ansi_red) + cell + ansi_reset;
                }
                line += cell;
            }
            return line;
        };

        std::vector<bool> mask = sorted_run_mask(keys);

        std::string idx_line = fmt_row(idx_label, idx_str, nullptr);
        std::string key_line = fmt_row(key_label, key_str, &mask);
        std::string val_line = fmt_row(val_label, val_str, nullptr);

        std::cout << std::string(idx_line.size(), '-') << "\n";
        std::cout << idx_line << "\n";
        std::cout << std::string(idx_line.size(), '-') << "\n";
        std::cout << key_line << "\n";
        std::cout << val_line << "\n";
    }

    enum MgpuBounds { MgpuBoundsLower, MgpuBoundsUpper };

    template<MgpuBounds Bounds, typename It1, typename It2, typename Comp>
    inline int MergePath(It1 a, int aCount, It2 b, int bCount, int diag, Comp comp) {

        typedef typename std::iterator_traits<It1>::value_type T;
        int begin = std::max(0, diag - bCount);
        int end   = std::min(diag, aCount);

        while (begin < end) {
            int mid   = (begin + end) >> 1;
            T aKey    = a[mid];
            T bKey    = b[diag - 1 - mid];
            bool pred = (MgpuBoundsUpper == Bounds) ? comp(aKey, bKey) : !comp(bKey, aKey);
            if (pred)
                begin = mid + 1;
            else
                end = mid;
        }
        return begin;
    }

    template<int VT, bool RangeCheck, typename T, typename Comp>
    inline void SerialMerge(
        sycl::nd_item<1> &item,
        const T *keys_shared,
        int aBegin,
        int aEnd,
        int bBegin,
        int bEnd,
        T *results,
        int *indices,
        Comp comp) {

        T aKey = keys_shared[aBegin];
        T bKey = keys_shared[bBegin];

#pragma unroll
        for (int i = 0; i < VT; ++i) {
            bool p;
            if (RangeCheck)
                p = (bBegin >= bEnd) || ((aBegin < aEnd) && !comp(bKey, aKey));
            else
                p = !comp(bKey, aKey);

            results[i] = p ? aKey : bKey;
            indices[i] = p ? aBegin : bBegin - !RangeCheck;

            if (p)
                aKey = keys_shared[++aBegin];
            else
                bKey = keys_shared[++bBegin];
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    template<int NT, int VT, typename T, typename Comp>
    inline void CTABlocksortPass(
        sycl::nd_item<1> &item,
        T *keys_shared,
        int tid,
        int count,
        int coop,
        T *keys,
        int *indices,
        Comp comp) {

        int list  = ~(coop - 1) & tid;
        int diag  = std::min(count, VT * ((coop - 1) & tid));
        int start = VT * list;
        int a0    = std::min(count, start);
        int b0    = std::min(count, start + VT * (coop / 2));
        int b1    = std::min(count, start + VT * coop);

        int p = MergePath<MgpuBoundsLower>(
            keys_shared + a0, b0 - a0, keys_shared + b0, b1 - b0, diag, comp);

        SerialMerge<VT, true>(
            item, keys_shared, a0 + p, b0, b0 + diag - p, b1, keys, indices, comp);
    }

    template<int VT, typename T>
    inline void DeviceThreadToShared(
        sycl::nd_item<1> &item, const T *threadReg, int tid, T *shared, bool sync = true) {

// Odd grain size. Store as type T.
#pragma unroll
        for (int i = 0; i < VT; ++i)
            shared[VT * tid + i] = threadReg[i];

        // In modern GPU there is 8-byte branch to exploit some instruction to store element 2 by 2

        if (sync)
            item.barrier(sycl::access::fence_space::local_space);
    }

    template<int NT, int VT, typename OutputIt, typename T>
    inline void DeviceRegToShared(
        sycl::nd_item<1> &item, const T *reg, int tid, OutputIt dest, bool sync) {

        typedef typename std::iterator_traits<OutputIt>::value_type T2;
#pragma unroll
        for (int i = 0; i < VT; ++i)
            dest[NT * i + tid] = (T2) reg[i];

        if (sync)
            item.barrier(sycl::access::fence_space::local_space);
    }

    template<int NT, int VT, typename OutputIt, typename T>
    inline void DeviceRegToGlobal(
        sycl::nd_item<1> &item, int count, const T *reg, int tid, OutputIt dest, bool sync) {

#pragma unroll
        for (int i = 0; i < VT; ++i) {
            int index = NT * i + tid;
            if (index < count)
                dest[index] = reg[i];
        }
        if (sync)
            item.barrier(sycl::access::fence_space::local_space);
    }

    template<int VT, typename T>
    inline void DeviceSharedToThread(
        sycl::nd_item<1> &item, const T *shared, int tid, T *threadReg, bool sync = true) {

#pragma unroll
        for (int i = 0; i < VT; ++i)
            threadReg[i] = shared[VT * tid + i];

        // In modern GPU there is 8-byte branch to exploit some instruction to store element 2 by 2

        if (sync)
            item.barrier(sycl::access::fence_space::local_space);
    }

    template<int NT, int VT, typename InputIt, typename T>
    inline void DeviceGlobalToRegPred(
        sycl::nd_item<1> &item, int count, InputIt data, int tid, T *reg, bool sync) {

// TODO: Attempt to issue 4 loads at a time.
#pragma unroll
        for (int i = 0; i < VT; ++i) {
            int index = NT * i + tid;
            if (index < count)
                reg[i] = data[index];
        }
        if (sync)
            item.barrier(sycl::access::fence_space::local_space);
    }

    template<int NT, int VT, typename InputIt, typename T>
    inline void DeviceGlobalToReg(
        sycl::nd_item<1> &item, int count, InputIt data, int tid, T *reg, bool sync) {

        if (count >= NT * VT) {
#pragma unroll
            for (int i = 0; i < VT; ++i)
                reg[i] = data[NT * i + tid];
        } else
            DeviceGlobalToRegPred<NT, VT>(item, count, data, tid, reg, false);

        if (sync)
            item.barrier(sycl::access::fence_space::local_space);
    }

    template<int NT, int VT, typename T, typename OutputIt>
    inline void DeviceSharedToGlobal(
        sycl::nd_item<1> &item,
        int count,
        const T *source,
        int tid,
        OutputIt dest,
        bool sync = true) {

        typedef typename std::iterator_traits<OutputIt>::value_type T2;
#pragma unroll
        for (int i = 0; i < VT; ++i) {
            int index = NT * i + tid;
            if (index < count)
                dest[index] = (T2) source[index];
        }
        if (sync)
            item.barrier(sycl::access::fence_space::local_space);
    }

    template<int NT, int VT, typename InputIt, typename T>
    inline void DeviceGlobalToShared(
        sycl::nd_item<1> &item, int count, InputIt source, int tid, T *dest, bool sync = true) {

        T reg[VT];
        DeviceGlobalToReg<NT, VT>(item, count, source, tid, reg, false);
        DeviceRegToShared<NT, VT>(item, reg, tid, dest, sync);
    }

    template<int NT, int VT, typename InputIt, typename T>
    inline void DeviceGather(
        sycl::nd_item<1> &item,
        int count,
        InputIt data,
        int indices[VT],
        int tid,
        T *reg,
        bool sync = true) {

        if (count >= NT * VT) {
#pragma unroll
            for (int i = 0; i < VT; ++i)
                reg[i] = data[indices[i]];
        } else {
#pragma unroll
            for (int i = 0; i < VT; ++i) {
                int index = NT * i + tid;
                if (index < count)
                    reg[i] = data[indices[i]];
            }
        }

        if (sync)
            item.barrier(sycl::access::fence_space::local_space);
    }

    template<int NT, int VT, bool HasValues, typename KeyType, typename ValType, typename Comp>
    inline void CTABlocksortLoop(
        sycl::nd_item<1> &item,
        ValType threadValues[VT],
        KeyType *keys_shared,
        ValType *values_shared,
        int tid,
        int count,
        Comp comp) {

#pragma unroll
        for (int coop = 2; coop <= NT; coop *= 2) {
            int indices[VT];
            KeyType keys[VT];
            CTABlocksortPass<NT, VT>(item, keys_shared, tid, count, coop, keys, indices, comp);

            if (HasValues) {
                // Exchange the values through shared memory.
                DeviceThreadToShared<VT>(item, threadValues, tid, values_shared);
                DeviceGather<NT, VT>(item, NT * VT, values_shared, indices, tid, threadValues);
            }

            // Store results in shared memory in sorted order.
            DeviceThreadToShared<VT>(item, keys, tid, keys_shared);
        }
    }

    template<int NT, int VT, bool HasValues, typename KeyType, typename ValType, typename Comp>
    inline void CTAMergesort(
        sycl::nd_item<1> &item,
        KeyType threadKeys[VT],
        ValType threadValues[VT],
        KeyType *keys_shared,
        ValType *values_shared,
        int count,
        int tid,
        Comp comp) {

        // Stable sort the keys in the thread.
        if (VT * tid < count) {
            workitem::odd_even_transpose_sort<VT>(threadKeys, threadValues, comp);
        }

        // Store the locally sorted keys into shared memory.
        DeviceThreadToShared<VT>(item, threadKeys, tid, keys_shared);

        // Recursively merge lists until the entire CTA is sorted.
        CTABlocksortLoop<NT, VT, HasValues>(
            item, threadValues, keys_shared, values_shared, tid, count, comp);
    }

    template<class Tkey, class Tval, int NT, int VT>
    union Shared {
        static constexpr int NV = NT * VT;

        Tkey keys[NT * (VT + 1)];
        Tval values[NV];
    };

    template<int NT, int VT, bool HasValues, typename Tkey, typename Tval, typename Comp>
    inline void KernelBlocksort(
        sycl::nd_item<1> &item,
        int tid,
        int block,
        Shared<Tkey, Tval, NT, VT> &shared,
        Tkey *keysSource_global,
        Tval *valsSource_global,
        int count,
        Tkey *keysDest_global,
        Tval *valsDest_global,
        Comp comp) {

        static constexpr int NV = NT * VT;

        int gid    = NV * block;
        int count2 = std::min(NV, count - gid);

        // Load the values into thread order.
        Tval threadValues[VT];
        if (HasValues) {
            DeviceGlobalToShared<NT, VT>(item, count2, valsSource_global + gid, tid, shared.values);
            DeviceSharedToThread<VT>(item, shared.values, tid, threadValues);
        }

        // Load keys into shared memory and transpose into register in thread order.
        Tkey threadKeys[VT];
        DeviceGlobalToShared<NT, VT>(item, count2, keysSource_global + gid, tid, shared.keys);
        DeviceSharedToThread<VT>(item, shared.keys, tid, threadKeys);

        // If we're in the last tile, set the uninitialized keys for the thread with
        // a partial number of keys.
        int first = VT * tid;
        if (first + VT > count2 && first < count2) {
            Tkey maxKey = threadKeys[0];
#pragma unroll
            for (int i = 1; i < VT; ++i)
                if (first + i < count2)
                    maxKey = comp(maxKey, threadKeys[i]) ? threadKeys[i] : maxKey;

// Fill in the uninitialized elements with max key.
#pragma unroll
            for (int i = 0; i < VT; ++i)
                if (first + i >= count2)
                    threadKeys[i] = maxKey;
        }

        CTAMergesort<NT, VT, HasValues>(
            item, threadKeys, threadValues, shared.keys, shared.values, count2, tid, comp);

        // Store the sorted keys to global.
        DeviceSharedToGlobal<NT, VT>(item, count2, shared.keys, tid, keysDest_global + gid);

        if (HasValues) {
            DeviceThreadToShared<VT>(item, threadValues, tid, shared.values);
            DeviceSharedToGlobal<NT, VT>(item, count2, shared.values, tid, valsDest_global + gid);
        }
    }

    ///// Fine //////

    /// Builds an `nd_range` with work-group size `wg_size`, rounding `nthread` up to the next
    /// multiple of `wg_size` so every work-group launched is full.
    inline sycl::nd_range<1> ndrange(u32 wg_size, u32 nthread) {
        u32 corrected_len = shambase::group_count(nthread, wg_size) * wg_size;
        return sycl::nd_range<1>{corrected_len, wg_size};
    }

    template<class Tkey, class Tval>
    inline void sort_by_keys_modern_gpu_mergesort(
        sham::DeviceBuffer<Tkey> &buf_key, sham::DeviceBuffer<Tval> &buf_values, u32 len) {

        auto dev_sched = buf_key.get_dev_scheduler_ptr();

        bool do_print = false;

        if (do_print) {
            std::cout << "-------------------------------------------------" << std::endl;
            std::cout << "------- sort_by_keys_modern_gpu_mergesort -------" << std::endl;
            std::cout << "-------------------------------------------------" << std::endl;

            std::cout << "init state:" << std::endl;
            print_key_val_table(buf_key.copy_to_stdvec(), buf_values.copy_to_stdvec());
        }

        static constexpr int VT = 7;
        static constexpr int NT = 4;
        static constexpr int NV = NT * VT;

        u32 nthreads = len / VT + 1;

        sham::kernel_call_hndl(
            dev_sched->get_queue(),
            sham::MultiRef{},
            sham::MultiRef{buf_key, buf_values},
            [nthreads, len](Tkey *__restrict keys, Tval *__restrict vals) {
                return [=](sycl::handler &cgh) {
                    using Shared = Shared<Tkey, Tval, NT, VT>;

                    sycl::local_accessor<Shared> shared_mem(1, cgh);

                    cgh.parallel_for(ndrange(NT, nthreads), [=](sycl::nd_item<1> item) {
                        u32 gid   = item.get_global_linear_id();
                        u32 lid   = item.get_local_linear_id();
                        u32 block = item.get_group_linear_id();

                        Tkey loc_key[VT];
                        Tval loc_val[VT];

                        for (int i = 0; i < VT; i++) {
                            u32 idx    = gid * VT + i;
                            loc_key[i] = (idx < len) ? keys[idx] : shambase::get_max<Tkey>();
                            loc_val[i] = (idx < len) ? vals[idx] : Tval{};
                        }

                        workitem::odd_even_transpose_sort<VT>(loc_key, loc_val, [](Tkey a, Tkey b) {
                            return a < b;
                        });

                        for (int i = 0; i < VT; i++) {
                            u32 idx = gid * VT + i;
                            if (idx < len) {
                                keys[idx] = loc_key[i];
                                vals[idx] = loc_val[i];
                            }
                        }

                        KernelBlocksort<NT, VT, true, Tkey, Tval>(
                            item,
                            lid,
                            block,
                            shared_mem[0],
                            keys,
                            vals,
                            len,
                            keys,
                            vals,
                            [](Tkey a, Tkey b) {
                                return a < b;
                            });
                    });
                };
            });

        if (do_print) {
            std::cout << "after local sort state:" << std::endl;
            print_key_val_table(buf_key.copy_to_stdvec(), buf_values.copy_to_stdvec());

            std::cout << "-------------------------------------------------" << std::endl;
            std::cout << "------- sort_by_keys_modern_gpu_mergesort end -------" << std::endl;
            std::cout << "-------------------------------------------------" << std::endl;
        }
    }

} // namespace shamalgs::primitives::device::details
