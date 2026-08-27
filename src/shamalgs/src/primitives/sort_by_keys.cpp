// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sort_by_keys.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Sort by keys algorithms
 *
 */

#include "shambase/exception.hpp"
#include "shambase/overloaded.hpp"
#include "shamalgs/ImplVariant.hpp"
#include "shamalgs/details/algorithm/bitonicSort.hpp"
#include "shamalgs/details/algorithm/bitonicSort_updated_usm.hpp"
#include "shamalgs/primitives/sort_by_keys.hpp"
#include "shamcomm/logs.hpp"
#include <algorithm>
#include <utility>
#include <vector>

namespace shamalgs::primitives::details {

    /// Copy both buffers to host, std::sort the zipped key/value pairs, and copy back
    template<class Tkey, class Tval>
    inline void sort_by_keys_std_sort(
        sham::DeviceBuffer<Tkey> &buf_key, sham::DeviceBuffer<Tval> &buf_values, u32 len) {

        std::vector<Tkey> key_stdvec = buf_key.copy_to_stdvec();
        std::vector<Tval> val_stdvec = buf_values.copy_to_stdvec();

        std::vector<std::pair<Tkey, Tval>> zipped(len);
        for (u32 i = 0; i < len; ++i) {
            zipped[i] = std::make_pair(key_stdvec[i], val_stdvec[i]);
        }

        std::sort(zipped.begin(), zipped.end(), [](const auto &a, const auto &b) {
            return a.first < b.first;
        });

        for (u32 i = 0; i < len; ++i) {
            key_stdvec[i] = zipped[i].first;
            val_stdvec[i] = zipped[i].second;
        }

        buf_key.copy_from_stdvec(key_stdvec);
        buf_values.copy_from_stdvec(val_stdvec);
    }

} // namespace shamalgs::primitives::details

namespace shamalgs::primitives {

    /// namespace to control implementation behavior
    namespace impl {

        /// Copy the buffers to host, std::sort the zipped key/value pairs, and copy back
        struct StdSort {
            static constexpr std::string_view variant_type_name = "std_sort";
        };

        shamalgs::ImplVariantGlobal<StdSort> sort_by_keys_impl;

        /// Get list of available sort by keys implementations
        std::vector<std::string> get_default_impl_list_sort_by_keys() {
            return decltype(sort_by_keys_impl)::get_default_config_list();
        }

        /// Get the current implementation for sort by keys
        std::string get_current_impl_sort_by_keys() {
            return sort_by_keys_impl.get_current_config();
        }

        /// Check if an implementation has been selected for sort by keys
        bool is_impl_set_sort_by_keys() { return sort_by_keys_impl.is_set(); }

        /// Set the implementation for sort by keys
        void set_impl_sort_by_keys(const std::string &impl) {
            shamlog_info_ln("algs", "setting sort by keys implementation to impl :", impl);
            sort_by_keys_impl.set(impl);
        }

        /// Select the default implementation for sort by keys
        void autoselect_impl_sort_by_keys() {
            sort_by_keys_impl.set(StdSort{});
            shamlog_info_ln(
                "algs",
                "defaulting sort by keys implementation to impl :",
                get_current_impl_sort_by_keys());
        }

    } // namespace impl

    template<class Tkey, class Tval>
    void sort_by_keys(
        sham::DeviceBuffer<Tkey> &buf_key, sham::DeviceBuffer<Tval> &buf_values, u32 len) {

        if (!impl::sort_by_keys_impl.is_set()) {
            impl::autoselect_impl_sort_by_keys();
        }

        std::visit(
            shambase::overloaded{
                [&](impl::StdSort) {
                    details::sort_by_keys_std_sort(buf_key, buf_values, len);
                },
            },
            impl::sort_by_keys_impl.get());
    }

    template void sort_by_keys(
        sham::DeviceBuffer<u32> &buf_key, sham::DeviceBuffer<u32> &buf_values, u32 len);

    template void sort_by_keys(
        sham::DeviceBuffer<u64> &buf_key, sham::DeviceBuffer<u32> &buf_values, u32 len);

    template void sort_by_keys(
        sham::DeviceBuffer<f64> &buf_key, sham::DeviceBuffer<f64> &buf_values, u32 len);

    template void sort_by_keys(
        sham::DeviceBuffer<f32> &buf_key, sham::DeviceBuffer<f32> &buf_values, u32 len);

    template<class Tkey, class Tval>
    void sort_by_key_pow2_len(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len) {

        if (!shambase::is_pow_of_two(len)) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "Length must be a power of 2");
        }

        if (len < 5e3) {
            shamalgs::algorithm::details::sort_by_key_bitonic_fallback(q, buf_key, buf_values, len);
        } else {
            shamalgs::algorithm::details::sort_by_key_bitonic_updated<Tkey, Tval, 16>(
                q, buf_key, buf_values, len);
        }
    }

    template<class Tkey, class Tval>
    void sort_by_key_pow2_len(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len) {

        if (!shambase::is_pow_of_two(len)) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "Length must be a power of 2");
        }

        shamalgs::algorithm::details::sort_by_key_bitonic_updated_usm<Tkey, Tval, 16>(
            sched, buf_key, buf_values, len);
    }

    template void sort_by_key_pow2_len(
        sycl::queue &q, sycl::buffer<u32> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

    template void sort_by_key_pow2_len(
        sycl::queue &q, sycl::buffer<u64> &buf_key, sycl::buffer<u32> &buf_values, u32 len);

    template void sort_by_key_pow2_len(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<u32> &buf_key,
        sham::DeviceBuffer<u32> &buf_values,
        u32 len);

    template void sort_by_key_pow2_len(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<u64> &buf_key,
        sham::DeviceBuffer<u32> &buf_values,
        u32 len);

    template void sort_by_key_pow2_len(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<f64> &buf_key,
        sham::DeviceBuffer<f64> &buf_values,
        u32 len);

    template void sort_by_key_pow2_len(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<f32> &buf_key,
        sham::DeviceBuffer<f32> &buf_values,
        u32 len);

} // namespace shamalgs::primitives
