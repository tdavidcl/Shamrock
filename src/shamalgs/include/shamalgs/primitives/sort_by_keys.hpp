// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sort_by_keys.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Sort by keys algorithms
 *
 */

#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/sycl.hpp"

namespace shamalgs::primitives {

    /**
     * @brief Sort the buffer according to the key order
     *
     * @tparam Tkey Key type
     * @tparam Tval Value type
     * @param q Queue
     * @param buf_key Key buffer
     * @param buf_values Value buffer
     * @param len Length of the buffers (must be a power of 2)
     */
    template<class Tkey, class Tval>
    void sort_by_key_pow2_len(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len);

    /**
     * @brief Sort the buffer according to the key order
     *
     * @tparam Tkey Key type
     * @tparam Tval Value type
     * @param sched Device scheduler
     * @param buf_key Key buffer
     * @param buf_values Value buffer
     * @param len Length of the buffers (must be a power of 2)
     */
    template<class Tkey, class Tval>
    void sort_by_key_pow2_len(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len);

    /**
     * @brief Sort the buffer according to the key order
     *
     * @tparam Tkey Key type
     * @tparam Tval Value type
     * @param q Queue
     * @param buf_key Key buffer
     * @param buf_values Value buffer
     * @param len Length of the buffers
     */
    template<class Tkey, class Tval>
    void sort_by_key(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len) {
        if (!shambase::is_pow_of_two(len))
            shambase::throw_with_loc<std::invalid_argument>("Length must be a power of 2");
        sort_by_key_pow2_len(q, buf_key, buf_values, len);
    }

    /**
     * @brief Sort the buffer according to the key order
     *
     * @tparam Tkey Key type
     * @tparam Tval Value type
     * @param sched Device scheduler
     * @param buf_key Key buffer
     * @param buf_values Value buffer
     * @param len Length of the buffers
     */
    template<class Tkey, class Tval>
    void sort_by_key(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len) {
        if (!shambase::is_pow_of_two(len))
            shambase::throw_with_loc<std::invalid_argument>("Length must be a power of 2");
        sort_by_key_pow2_len(sched, buf_key, buf_values, len);
    }

} // namespace shamalgs::primitives
