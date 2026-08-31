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
 * @file make_ndrange.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Helper to build a `sycl::nd_range<1>` from a work-group size and a thread count
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/assert.hpp"
#include "shambase/exception.hpp"
#include "sham/format/format.hpp"
#include "shambackends/Device.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/sycl.hpp"

namespace sham {

    /**
     * @brief Builds an `nd_range` with work-group size `wg_size`, rounding `nthread` up to the
     * next multiple of `wg_size` so every work-group launched is full
     *
     * @param wg_size the work-group size, must be > 0
     * @param nthread the number of threads to cover, must be > 0
     * @return sycl::nd_range<1> the resulting nd range
     */
    inline sycl::nd_range<1> make_ndrange(size_t wg_size, size_t nthread) {
        if (nthread == 0) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                sham::format("make_ndrange: nthread must be > 0 (nthread = {})", nthread));
        }
        if (wg_size == 0) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                sham::format("make_ndrange: wg_size must be > 0 (wg_size = {})", wg_size));
        }
        size_t nthread_rounded = ((nthread + wg_size - 1) / wg_size) * wg_size;
        SHAM_ASSERT(nthread_rounded % wg_size == 0);
        SHAM_ASSERT(nthread_rounded > 0);
        return sycl::nd_range<1>(sycl::range<1>(nthread_rounded), sycl::range<1>(wg_size));
    }

    /**
     * @brief Checks that an `nd_range` can be launched on the given device
     *
     * @param in the nd range to check
     * @param d the device the nd range is meant to be launched on
     * @throws std::invalid_argument if the local (work-group) range of `in` exceeds the
     * device's max work-group size
     */
    inline void check_ndrange(const sycl::nd_range<1> &in, DeviceProperties &d) {
        if (in.get_local_range().size() > d.max_work_group_size) {
            throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                "nd_range local size ({}) exceeds the device's max work-group size ({}) on "
                "device {}",
                in.get_local_range().size(),
                d.max_work_group_size,
                d.name));
        }
    }

    /**
     * @brief Builds an `nd_range` with `make_ndrange` and checks that it can be launched on the
     * given device
     *
     * @param wg_size the work-group size, must be > 0
     * @param nthread the number of threads to cover, must be > 0
     * @param d the device the nd range is meant to be launched on
     * @return sycl::nd_range<1> the resulting nd range
     * @throws std::invalid_argument if the resulting local (work-group) range exceeds the
     * device's max work-group size
     */
    inline sycl::nd_range<1> make_check_ndrange(size_t wg_size, size_t nthread, Device &d) {
        auto ret = sham::make_ndrange(wg_size, nthread);
        check_ndrange(ret, d.prop);
        return ret;
    }

    /**
     * @brief Builds an `nd_range` with `make_ndrange` and checks that it can be launched on the
     * device backing the given queue
     *
     * @param wg_size the work-group size, must be > 0
     * @param nthread the number of threads to cover, must be > 0
     * @param d the queue whose device the nd range is meant to be launched on
     * @return sycl::nd_range<1> the resulting nd range
     * @throws std::invalid_argument if the resulting local (work-group) range exceeds the
     * device's max work-group size
     */
    inline sycl::nd_range<1> make_check_ndrange(size_t wg_size, size_t nthread, DeviceQueue &d) {
        return make_check_ndrange(wg_size, nthread, *d.ctx->device);
    }

} // namespace sham
