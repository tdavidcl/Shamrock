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
 * @file sqrt_noerrno.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Correctly rounded square root without the errno side effect.
 *
 * On host backends sycl::sqrt maps to std::sqrt, which may set errno : the compiler then keeps a
 * branch to the libm call and does not vectorize the loops using it. The IEEE square root being
 * correctly rounded, the value is identical to sycl::sqrt.
 */

#include "shambackends/sycl.hpp"

namespace shamrock::sph {

    /// Square root, same value as sycl::sqrt, without setting errno
    template<class T>
    inline T sqrt_noerrno(T x) {
#if defined(__has_builtin)
    #if __has_builtin(__builtin_elementwise_sqrt)
        return __builtin_elementwise_sqrt(x);
    #else
        return sycl::sqrt(x);
    #endif
#else
        return sycl::sqrt(x);
#endif
    }

} // namespace shamrock::sph
