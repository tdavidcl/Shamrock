// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include <sycl/sycl.hpp>

int main(void) {
    sycl::queue q{};
    int *ptr = sycl::malloc_shared<int>(16, q);
    q.parallel_for(sycl::range<1>{16}, [=](sycl::item<1> id) {
         ptr[id.get_linear_id()] = id.get_linear_id();
     }).wait();
    int ret = ptr[15];
    sycl::free(ptr, q);
    return ret == 15 ? 0 : 1;
}
