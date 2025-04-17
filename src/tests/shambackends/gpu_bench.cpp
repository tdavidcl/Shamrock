// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/SourceLocation.hpp"
#include "shambase/integer.hpp"
#include "shambase/time.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <shambackends/sycl.hpp>
#include <cstdio>

namespace sham {
    template<class T, int sz>
    class aligned_storage {

        static constexpr auto align_func() {
            if constexpr (sz == 1) {
                return 1;
            } else if constexpr (sz == 2) {
                return 2;
            } else if constexpr (sz == 3) {
                return 4;
            } else if constexpr (sz == 4) {
                return 4;
            } else if constexpr (sz == 8) {
                return 8;
            } else if constexpr (sz == 16) {
                return 16;
            } else {
                return shambase::round_up_multiple(sz, 4);
            }
        };

        static constexpr u32 effective_size = align_func(sz);
        static constexpr u32 align          = sizeof(T) * effective_size;

        public:
        alignas(align) T _st[effective_size];
    };

    template<class T, class accessor>
    class bind_vec {
        public:
        accessor acc;
        int & size;
        
        T& operator()(int i) { return acc[i]; }
        const T& operator()(int i) const { return acc[i]; }

    };

    template<class T, class accessor>
    class bind_mat {
        public:
        accessor acc;
        int & cols;
        int & rows;
        
        T& operator()(int i, int j) { return acc[i*cols+j]; }
        const T& operator()(int i, int j) const { return acc[i*cols+j]; }

    };

    // To be replaced by mdspan asap
    namespace mdspan {
        template<typename T>
        class owning_pointer {
            public:
            T* acc;
            T& operator[](int i) { return acc[i]; }
            const T& operator[](int i) const { return acc[i]; }
        };

        template<typename T>
        class ref_pointer {
            public:
            T*  & acc;
            T& operator[](int i) { return acc[i]; }
            const T& operator[](int i) const { return acc[i]; }
        };

        template<typename T, class accessor>
        class ref_custom_accessor {
            public:
            accessor & acc;
            T& operator[](int i) { return acc[i]; }
            const T& operator[](int i) const { return acc[i]; }
        };

        
    }
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

    sham::EventList depends_list;

    auto x_ptr = x.get_write_access(depends_list);
    auto y_ptr = y.get_write_access(depends_list);

    depends_list.wait();

    sham::EventList empty_list{};

    shambase::Timer t;
    t.start();
    auto e = q.submit(empty_list, [&](sycl::handler &cgh) {
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
        T delt = y_res[i] - expected;

        if constexpr (std::is_same_v<T, sycl::marray<float, 3>>) {
            maxError[0] = sham::max(maxError[0], sham::abs(delt[0]));
            maxError[1] = sham::max(maxError[1], sham::abs(delt[1]));
            maxError[2] = sham::max(maxError[2], sham::abs(delt[2]));
        } else if constexpr (std::is_same_v<T, sycl::marray<float, 4>>) {
            maxError[0] = sham::max(maxError[0], sham::abs(delt[0]));
            maxError[1] = sham::max(maxError[1], sham::abs(delt[1]));
            maxError[2] = sham::max(maxError[2], sham::abs(delt[2]));
            maxError[3] = sham::max(maxError[3], sham::abs(delt[3]));
        } else {
            maxError = sham::max(maxError, sham::abs(delt));
        }
    }

    logger::raw_ln("--------------- saxpy ---------------");
    logger::raw_ln(SourceLocation{}.loc.function_name());
    logger::raw_ln("GPU time (ms):", milliseconds);
    logger::raw_ln("Max error:", maxError);
    logger::raw_ln("Effective Bandwidth (GB/s): ", double(N) * load_size * 3 / milliseconds / 1e6);
}

// From https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
TestStart(Benchmark, "saxpy", saxpy_base, 1) {
    int N = (1 << 27);

    int float_size = sizeof(float);
    saxpy_bench<float>(N, 1.0f, 2.0f, 2.0f, float_size);

    using vec3    = sycl::vec<float, 3>;
    int vec3_size = sizeof(vec3);
    saxpy_bench<vec3>(N, {1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}, {2.0f, 2.0f, 2.0f}, vec3_size);

    using vec4    = sycl::vec<float, 4>;
    int vec4_size = sizeof(vec4);
    saxpy_bench<vec4>(
        N, {1.0f, 1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f, 2.0f}, {2.0f, 2.0f, 2.0f, 2.0f}, vec4_size);

    using vec3_2    = sycl::marray<float, 3>;
    int vec3_2_size = sizeof(vec3_2);
    saxpy_bench<vec3_2>(N, {1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}, {2.0f, 2.0f, 2.0f}, vec3_2_size);

    using vec4_2    = sycl::marray<float, 4>;
    int vec4_2_size = sizeof(vec4_2);
    saxpy_bench<vec4_2>(
        N,
        {1.0f, 1.0f, 1.0f, 1.0f},
        {2.0f, 2.0f, 2.0f, 2.0f},
        {2.0f, 2.0f, 2.0f, 2.0f},
        vec4_2_size);
}
