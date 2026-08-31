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
#include "shambackends/make_ndrange.hpp"
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
    struct KernelBlocksort {

        static constexpr int NV = NT * VT;

        union Shared {

            Tkey keys[NT * (VT + 1)];
            Tval values[NV];
        };

        template<bool HasValues, typename Comp>
        inline void Kernel(
            sycl::nd_item<1> &item,
            int tid,
            int block,
            Shared &shared,
            Tkey *keysSource_global,
            Tval *valsSource_global,
            int count,
            Tkey *keysDest_global,
            Tval *valsDest_global,
            Comp comp) {

            int gid    = NV * block;
            int count2 = std::min(NV, count - gid);

            // Load the values into thread order.
            Tval threadValues[VT];
            if (HasValues) {
                DeviceGlobalToShared<NT, VT>(
                    item, count2, valsSource_global + gid, tid, shared.values);
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
                DeviceSharedToGlobal<NT, VT>(
                    item, count2, shared.values, tid, valsDest_global + gid);
            }
        }
    };

    ///// Fine //////

    template<int NT, int VT, class Tkey, class Tval>
    inline void kernel_blocksort(
        sham::DeviceQueue &q,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len) {

        static constexpr int NV = NT * VT;

        u32 nthreads = len / VT + 1;

        using Kernel = KernelBlocksort<Tkey, Tval, NT, VT>;

        sham::kernel_call_hndl(
            q,
            sham::MultiRef{},
            sham::MultiRef{buf_key, buf_values},
            [nthreads, len](Tkey *__restrict keys, Tval *__restrict vals) {
                return [=](sycl::handler &cgh) {
                    using Shared = Kernel::Shared;

                    sycl::local_accessor<Shared> shared_mem(1, cgh);

                    cgh.parallel_for(sham::make_ndrange(NT, nthreads), [=](sycl::nd_item<1> item) {
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

                        Kernel::Kernel<true>(
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
    }

#define MGPU_DIV_UP(x, y) (((x) + (y) - 1) / (y))
#define MGPU_IS_POW_2(x) (0 == ((x) & ((x) - 1)))

    // Find log2(x) and optionally round up to the next integer logarithm.
    inline int FindLog2(int x, bool roundUp = false) {
        int a = 31 - sycl::clz(x);
        if (roundUp)
            a += !MGPU_IS_POW_2(x);
        return a;
    }

    inline sycl::vec<int, 3> FindMergesortFrame(int coop, int block, int nv) {
        // coop is the number of CTAs or threads cooperating to merge two lists into
        // one. We round block down to the first CTA's ID that is working on this
        // merge.
        int start = ~(coop - 1) & block;
        int size  = nv * (coop >> 1);
        return {nv * start, nv * start + size, size};
    }

    template<int NT, MgpuBounds Bounds, typename It1, typename It2, typename Comp>
    inline void KernelMergePartition(
        int block,
        int tid,
        It1 a_global,
        int aCount,
        It2 b_global,
        int bCount,
        int nv,
        int coop,
        int *mp_global,
        int numSearches,
        Comp comp) {

        int partition = NT * block + tid;
        if (partition < numSearches) {
            int a0 = 0, b0 = 0;
            int gid = nv * partition;
            if (coop) {
                auto frame = FindMergesortFrame(coop, partition, nv);
                a0         = frame.x();
                b0         = std::min(aCount, frame.y());
                bCount     = std::min(aCount, frame.y() + frame.z()) - b0;
                aCount     = std::min(aCount, frame.x() + frame.z()) - a0;

                // Put the cross-diagonal into the coordinate system of the input
                // lists.
                gid -= a0;
            }
            int mp = MergePath<Bounds>(
                a_global + a0, aCount, b_global + b0, bCount, std::min(gid, aCount + bCount), comp);
            mp_global[partition] = mp;
        }
    }

    template<MgpuBounds Bounds, typename T1, typename T2, typename Comp>
    inline void MergePathPartitions(
        sham::DeviceQueue &q,
        sham::DeviceBuffer<T1> &a_global,
        int aCount,
        sham::DeviceBuffer<T2> &b_global,
        int bCount,
        int nv,
        int coop,
        Comp comp,
        sham::DeviceBuffer<int> &partitionsDevice) {

        const int NT           = 64;
        int numPartitions      = MGPU_DIV_UP(aCount + bCount, nv);
        int numSearches        = numPartitions + 1;
        int numPartitionBlocks = MGPU_DIV_UP(numSearches, NT);
        partitionsDevice.resize(numSearches);

        sham::kernel_call_hndl(
            q,
            sham::MultiRef{},
            sham::MultiRef{a_global, b_global, partitionsDevice},
            [=](auto a, auto b, int *__restrict part_dev) {
                return [=](sycl::handler &cgh) {
                    cgh.parallel_for(
                        sham::make_ndrange(NT, numPartitionBlocks * NT),
                        [=](sycl::nd_item<1> item) {
                            int block = static_cast<int>(item.get_group_linear_id());
                            int tid   = static_cast<int>(item.get_local_linear_id());

                            KernelMergePartition<NT, Bounds>(
                                block,
                                tid,
                                a,
                                aCount,
                                b,
                                bCount,
                                nv,
                                coop,
                                part_dev,
                                numSearches,
                                comp);
                        });
                };
            });
    }

    // Returns (a0, a1, b0, b1) into mergesort input lists between mp0 and mp1.
    inline sycl::vec<int, 4> FindMergesortInterval(
        sycl::vec<int, 3> frame, int coop, int block, int nv, int count, int mp0, int mp1) {

        // Locate diag from the start of the A sublist.
        int diag = nv * block - frame.x();
        int a0   = frame.x() + mp0;
        int a1   = std::min(count, frame.x() + mp1);
        int b0   = std::min(count, frame.y() + diag - mp0);
        int b1   = std::min(count, frame.y() + diag + nv - mp1);

        // The end partition of the last block for each merge operation is computed
        // and stored as the begin partition for the subsequent merge. i.e. it is
        // the same partition but in the wrong coordinate system, so its 0 when it
        // should be listSize. Correct that by checking if this is the last block
        // in this merge operation.
        if (coop - 1 == ((coop - 1) & block)) {
            a1 = std::min(count, frame.x() + frame.z());
            b1 = std::min(count, frame.y() + frame.z());
        }
        return {a0, a1, b0, b1};
    }

    inline sycl::vec<int, 4> ComputeMergeRange(
        int aCount, int bCount, int block, int coop, int NV, const int *mp_global) {

        // Load the merge paths computed by the partitioning kernel.
        int mp0 = mp_global[block];
        int mp1 = mp_global[block + 1];
        int gid = NV * block;

        // Compute the ranges of the sources in global memory.
        sycl::vec<int, 4> range;
        if (coop) {
            sycl::vec<int, 3> frame = FindMergesortFrame(coop, block, NV);
            range = FindMergesortInterval(frame, coop, block, NV, aCount, mp0, mp1);
        } else {
            range.x() = mp0;                                             // a0
            range.y() = mp1;                                             // a1
            range.z() = gid - range.x();                                 // b0
            range.w() = std::min(aCount + bCount, gid + NV) - range.y(); // b1
        }
        return range;
    }

    template<
        int NT,
        int VT,
        bool LoadExtended,
        typename It1,
        typename It2,
        typename T,
        typename Comp>
    MGPU_DEVICE void DeviceMergeKeysIndices(
        It1 a_global,
        int aCount,
        It2 b_global,
        int bCount,
        int4 range,
        int tid,
        T *keys_shared,
        T *results,
        int *indices,
        Comp comp) {

        int a0 = range.x;
        int a1 = range.y;
        int b0 = range.z;
        int b1 = range.w;

        if (LoadExtended) {
            bool extended = (a1 < aCount) && (b1 < bCount);
            aCount        = a1 - a0;
            bCount        = b1 - b0;
            int aCount2   = aCount + (int) extended;
            int bCount2   = bCount + (int) extended;

            // Load one element past the end of each input to avoid having to use
            // range checking in the merge loop.
            DeviceLoad2ToShared<NT, VT, VT + 1>(
                a_global + a0, aCount2, b_global + b0, bCount2, tid, keys_shared);

            // Run a Merge Path search for each thread's starting point.
            int diag = VT * tid;
            int mp   = MergePath<MgpuBoundsLower>(
                keys_shared, aCount, keys_shared + aCount2, bCount, diag, comp);

            // Compute the ranges of the sources in shared memory.
            int a0tid = mp;
            int b0tid = aCount2 + diag - mp;
            if (extended) {
                SerialMerge<VT, false>(keys_shared, a0tid, 0, b0tid, 0, results, indices, comp);
            } else {
                int a1tid = aCount;
                int b1tid = aCount2 + bCount;
                SerialMerge<VT, true>(
                    keys_shared, a0tid, a1tid, b0tid, b1tid, results, indices, comp);
            }
        } else {
            // Use the input intervals from the ranges between the merge path
            // intersections.
            aCount = a1 - a0;
            bCount = b1 - b0;

            // Load the data into shared memory.
            DeviceLoad2ToShared<NT, VT, VT>(
                a_global + a0, aCount, b_global + b0, bCount, tid, keys_shared);

            // Run a merge path to find the start of the serial merge for each
            // thread.
            int diag = VT * tid;
            int mp   = MergePath<MgpuBoundsLower>(
                keys_shared, aCount, keys_shared + aCount, bCount, diag, comp);

            // Compute the ranges of the sources in shared memory.
            int a0tid = mp;
            int a1tid = aCount;
            int b0tid = aCount + diag - mp;
            int b1tid = aCount + bCount;

            // Serial merge into register.
            SerialMerge<VT, true>(keys_shared, a0tid, a1tid, b0tid, b1tid, results, indices, comp);
        }
    }

    template<
        int NT,
        int VT,
        bool HasValues,
        bool LoadExtended,
        typename KeysIt1,
        typename KeysIt2,
        typename KeysIt3,
        typename ValsIt1,
        typename ValsIt2,
        typename KeyType,
        typename ValsIt3,
        typename Comp>
    inline void DeviceMerge(
        sycl::nd_item<1> &item,
        KeysIt1 aKeys_global,
        ValsIt1 aVals_global,
        int aCount,
        KeysIt2 bKeys_global,
        ValsIt2 bVals_global,
        int bCount,
        int tid,
        int block,
        sycl::vec<int, 4> range,
        KeyType *keys_shared,
        int *indices_shared,
        KeysIt3 keys_global,
        ValsIt3 vals_global,
        Comp comp) {

        KeyType results[VT];
        int indices[VT];
        DeviceMergeKeysIndices<NT, VT, LoadExtended>(
            aKeys_global,
            aCount,
            bKeys_global,
            bCount,
            range,
            tid,
            keys_shared,
            results,
            indices,
            comp);

        // Store merge results back to shared memory.
        DeviceThreadToShared<VT>(results, tid, keys_shared);

        // Store merged keys to global memory.
        aCount = range.y() - range.x();
        bCount = range.w() - range.z();
        DeviceSharedToGlobal<NT, VT>(
            aCount + bCount, keys_shared, tid, keys_global + NT * VT * block);

        // Copy the values.
        if (HasValues) {
            DeviceThreadToShared<VT>(item, indices, tid, indices_shared);

            DeviceTransferMergeValuesShared<NT, VT>(
                aCount + bCount,
                aVals_global + range.x(),
                bVals_global + range.z(),
                aCount,
                indices_shared,
                tid,
                vals_global + NT * VT * block);
        }
    }

    template<
        int NT,
        int VT,
        bool HasValues,
        bool LoadExtended,
        typename Tkey,
        typename Tval,
        typename Comp>
    inline void KernelMerge(
        sycl::nd_item<1> &item,
        int tid,
        int block,
        Shared<Tkey, int, NT, VT> &shared,
        Tkey *aKeys_global,
        Tval *aVals_global,
        int aCount,
        Tkey *bKeys_global,
        Tval *bVals_global,
        int bCount,
        const int *mp_global,
        int coop,
        Tkey *keys_global,
        Tval *vals_global,
        Comp comp) {

        static constexpr int NV = NT * VT;

        sycl::vec<int, 4> range
            = ComputeMergeRange(aCount, bCount, block, coop, NT * VT, mp_global);

        DeviceMerge<NT, VT, HasValues, LoadExtended>(
            item,
            aKeys_global,
            aVals_global,
            aCount,
            bKeys_global,
            bVals_global,
            bCount,
            tid,
            block,
            range,
            shared.keys,
            shared.values,
            keys_global,
            vals_global,
            comp);
    }

    template<class Tkey, class Tval>
    inline void sort_by_keys_modern_gpu_mergesort(
        sham::DeviceBuffer<Tkey> &buf_key, sham::DeviceBuffer<Tval> &buf_values, u32 len) {

        auto dev_sched = buf_key.get_dev_scheduler_ptr();

        bool do_print = false;

        static constexpr int VT = 7;
        static constexpr int NT = 4; // 256
        static constexpr int NV = NT * VT;

        int numBlocks = MGPU_DIV_UP(len, NV);
        int numPasses = FindLog2(numBlocks, true);

        if (do_print) {
            std::cout << "-------------------------------------------------" << std::endl;
            std::cout << "------- sort_by_keys_modern_gpu_mergesort -------" << std::endl;
            std::cout << "-------------------------------------------------" << std::endl;

            std::cout << "init state:" << std::endl;
            print_key_val_table(buf_key.copy_to_stdvec(), buf_values.copy_to_stdvec());
        }

        kernel_blocksort<NT, VT>(dev_sched->get_queue(), buf_key, buf_values, len);

        if (do_print) {
            std::cout << "after local sort state:" << std::endl;
            print_key_val_table(buf_key.copy_to_stdvec(), buf_values.copy_to_stdvec());

            std::cout << "-------------------------------------------------" << std::endl;
            std::cout << "------- sort_by_keys_modern_gpu_mergesort end -------" << std::endl;
            std::cout << "-------------------------------------------------" << std::endl;
        }

        auto buf_key2    = sham::DeviceBuffer<Tkey>(len, dev_sched);
        auto buf_values2 = sham::DeviceBuffer<Tval>(len, dev_sched);

        if (1 & numPasses) {
            std::swap(buf_key, buf_key2);
            std::swap(buf_values, buf_values2);
        }

        auto partitionsDevice = sham::DeviceBuffer<int>(0, dev_sched);

        for (int pass = 0; pass < numPasses; ++pass) {
            int coop = 2 << pass;

            MergePathPartitions<MgpuBoundsLower>(
                dev_sched->get_queue(),
                buf_key,
                len,
                buf_key,
                0,
                NV,
                coop,
                [](Tkey a, Tkey b) {
                    return a < b;
                },
                partitionsDevice);

            sham::kernel_call_hndl(
                dev_sched->get_queue(),
                sham::MultiRef{},
                sham::MultiRef{buf_key, buf_values, partitionsDevice, buf_key2, buf_values2},
                [=](auto key, auto values, auto part_dev, auto key2, auto values2) {
                    return [=](sycl::handler &cgh) {
                        using Shared = Shared<Tkey, int, NT, VT>;

                        sycl::local_accessor<Shared> shared_mem(1, cgh);

                        cgh.parallel_for(ndrange(NT, numBlocks * NT), [=](sycl::nd_item<1> item) {
                            KernelMerge<NT, VT, true, false, Tkey, Tval>(
                                item,
                                item.get_local_linear_id(),
                                item.get_group_linear_id(),
                                shared_mem[0],
                                key,
                                values,
                                len,
                                key,
                                values,
                                0,
                                part_dev,
                                coop,
                                key2,
                                values2,
                                [](Tkey a, Tkey b) {
                                    return a < b;
                                });
                        });
                    };
                });

            std::swap(buf_key, buf_key2);
            std::swap(buf_values, buf_values2);
        }
    }

} // namespace shamalgs::primitives::device::details
