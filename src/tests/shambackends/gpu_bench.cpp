// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <cstdio>

namespace sham {
    template<class T, int sz>
    class vec {
        static constexpr u32 align          = sizeof(T);
        static constexpr u32 effective_size = sz;
        alignas(align) T _storage[effective_size];
    };
} // namespace sham

template<class T>
inline void saxpy(u32 i, int n, T a, T *__restrict x, T *__restrict y) {
    if (i < n)
        y[i] = a * x[i] + y[i];
}

template<class T>
inline void saxpy_bench(int N, T init_x, T init_y, T a, int load_size) {

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::DeviceBuffer<T> x = {size_t(N), shamsys::instance::get_compute_scheduler_ptr()};
    sham::DeviceBuffer<T> y = {size_t(N), shamsys::instance::get_compute_scheduler_ptr()};

    x.fill(init_x);
    y.fill(init_y);

    x.synchronize();
    y.synchronize();

    sham::EventList depends_list;

    auto x_ptr = x.get_write_access(depends_list);
    auto y_ptr = y.get_write_access(depends_list);

    shambase::Timer t;
    t.start();
    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{size_t(N)}, [=](sycl::item<1> item) {
            // printf("%d\n", item.get_linear_id());
            saxpy(item.get_linear_id(), N, a, x_ptr, y_ptr);
        });
    });
    e.wait();
    t.end();

    x.complete_event_state(sycl::event{});
    y.complete_event_state(sycl::event{});

    double milliseconds = t.elasped_sec() * 1e3;

    auto y_res = y.copy_to_stdvec();

    T expected = a * init_x + init_y;

    T maxError = {};
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y_res[i] - expected));
    }

    logger::raw_ln("Max error:", maxError);
    logger::raw_ln("Effective Bandwidth (GB/s): ", double(N) * load_size * 3 / milliseconds / 1e6);
}

// From https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
TestStart(Benchmark, "saxpy", saxpy_base, 1) {
    int N = 20 * (1 << 23);

    int float_size = sizeof(float);
    saxpy_bench<float>(N, 1.0f, 2.0f, 2.0f, float_size);

    using vec4     = sycl::vec<float, 4>;
    int vec_4_size = sizeof(vec4);
    saxpy_bench<vec4>(
        N,
        {1.0f, 1.0f, 1.0f, 1.0f},
        {2.0f, 2.0f, 2.0f, 2.0f},
        {2.0f, 2.0f, 2.0f, 2.0f},
        vec_4_size);

    using vec3     = sycl::vec<float, 3>;
    int vec_3_size = sizeof(vec3);
    saxpy_bench<vec3>(N, {1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}, {2.0f, 2.0f, 2.0f}, vec_3_size);
}
