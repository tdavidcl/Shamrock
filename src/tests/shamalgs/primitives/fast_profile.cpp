// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "omp.h"
#include "shamalgs/primitives/equals.hpp"
#include "shamalgs/primitives/mock_value.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/sphkernels.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <random>
#include <vector>

template<class T>
inline bool compare(const std::vector<T> &v1, const std::vector<T> &v2, T tol) {

    if (v1.size() != v2.size()) {
        return false;
    }

    for (size_t i = 0; i < v1.size(); i++) {
        if (sham::abs(v1[i] - v2[i]) > tol) {
            return false;
        }
    }

    return true;
}

template<class Tscal>
void __attribute__((noinline)) trap_result(const std::vector<Tscal> &result) {
    volatile Tscal tmp = result[0];
    (void) tmp;
}

TestStart(Unittest, "tmp_fast_profile", tmp_fast_profile, 1) {
    using Tscal = f32;

    std::vector<Tscal> positions{};
    std::vector<Tscal> sizes{};

    size_t N      = 30e6;
    size_t Narray = 2048;

    std::mt19937_64 eng(0x42);

    for (size_t i = 0; i < N; i++) {
        positions.push_back(shamalgs::primitives::mock_value<Tscal>(eng, 0, 1));
        sizes.push_back(0.01);
    }

    std::vector<Tscal> arr_pos{};
    for (size_t i = 0; i < Narray; i++) {
        arr_pos.push_back((f64(i) + 0.5) / f64(Narray));
    }

    using Kernel = shammath::M4<Tscal>;

    std::vector<Tscal> reference(Narray, 0);
    f64 time_baseline = shambase::timeit([&]() {
        for (size_t ipoint = 0; ipoint < Narray; ipoint++) {
            Tscal p   = arr_pos[ipoint];
            Tscal sum = 0;
            for (size_t i = 0; i < N; i++) {
                Tscal dist = sham::abs(positions[i] - p);
                Tscal sz   = sizes[i];
                if (dist < sz * Kernel::Rkern) {
                    sum += Kernel::W_1d(dist, sz);
                }
            }
            reference[ipoint] = sum;
        }
    });
    trap_result(reference);
    logger::raw_ln("reference :", reference);

    logger::raw_ln("baseline :", shambase::nanosec_to_time_str(time_baseline * 1e9));

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<Tscal> positions_buf(N, dev_sched);
    positions_buf.copy_from_stdvec(positions);

    sham::DeviceBuffer<Tscal> sizes_buf(N, dev_sched);
    sizes_buf.copy_from_stdvec(sizes);

    sham::DeviceBuffer<Tscal> arr_pos_buf(Narray, dev_sched);
    arr_pos_buf.copy_from_stdvec(arr_pos);

    sham::DeviceBuffer<Tscal> result_buf(Narray, dev_sched);
    result_buf.fill(0);

    positions_buf.synchronize();
    sizes_buf.synchronize();
    arr_pos_buf.synchronize();
    result_buf.synchronize();

    f64 time_result_trivial_gpu = shambase::timeit([&]() {
        sham::kernel_call(
            dev_sched->get_queue(),
            sham::MultiRef{positions_buf, sizes_buf, arr_pos_buf},
            sham::MultiRef{result_buf},
            Narray,
            [N](size_t ipoint,
                const Tscal *__restrict positions,
                const Tscal *__restrict sizes,
                const Tscal *__restrict arr_pos,
                Tscal *__restrict result) {
                Tscal p   = arr_pos[ipoint];
                Tscal sum = 0;
                for (size_t i = 0; i < N; i++) {
                    Tscal dist = sham::abs(positions[i] - p);
                    Tscal sz   = sizes[i];
                    if (dist < sz * Kernel::Rkern) {
                        sum += Kernel::W_1d(dist, sz);
                    }
                }
                result[ipoint] = sum;
            });
        result_buf.synchronize();
    });

    REQUIRE(compare(reference, result_buf.copy_to_stdvec(), Tscal{1e-12}));

    logger::raw_ln(
        "result_trivial_gpu :", shambase::nanosec_to_time_str(time_result_trivial_gpu * 1e9));

    result_buf.fill(0);

    positions_buf.synchronize();
    sizes_buf.synchronize();
    arr_pos_buf.synchronize();
    result_buf.synchronize();

    f64 time_result_trivial_gpu_teams = shambase::timeit([&]() {
        sham::kernel_call_hndl(
            dev_sched->get_queue(),
            sham::MultiRef{positions_buf, sizes_buf, arr_pos_buf},
            sham::MultiRef{result_buf},
            Narray,
            [N](u32 Narray,
                const Tscal *__restrict positions,
                const Tscal *__restrict sizes,
                const Tscal *__restrict arr_pos,
                Tscal *__restrict result) {
                return [=](sycl::handler &cgh) {
                    u32 group_size = 32;
                    u32 group_cnt  = shambase::group_count(Narray, group_size);

                    group_cnt         = group_cnt + (group_cnt % 4);
                    u32 corrected_len = group_cnt * group_size;

                    sycl::local_accessor<Tscal, 1> local_pos{group_size, cgh};
                    sycl::local_accessor<Tscal, 1> local_sizes{group_size, cgh};

                    cgh.parallel_for(
                        sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> item) {
                            u32 local_id      = item.get_local_id(0);
                            u32 group_tile_id = item.get_group_linear_id();
                            u32 point_id      = group_tile_id * group_size + local_id;

                            Tscal local_p = arr_pos[point_id];

                            bool is_valid_point = (point_id < Narray);

                            Tscal local_sum = 0;

                            for (size_t i = 0; i < N; i += group_size) {
                                if (i + local_id < N) {
                                    local_pos[local_id]   = positions[i + local_id];
                                    local_sizes[local_id] = sizes[i + local_id];
                                }
                                item.barrier(sycl::access::fence_space::local_space);
                                if (i + local_id < N && is_valid_point) {
                                    for (size_t i = 0; i < group_size; i++) {
                                        Tscal dist = sham::abs(local_pos[i] - local_p);
                                        Tscal sz   = local_sizes[i];
                                        if (dist < sz * Kernel::Rkern) {
                                            local_sum += Kernel::W_1d(
                                                sham::abs(local_pos[i] - local_p), local_sizes[i]);
                                        }
                                    }
                                }
                            }

                            if (is_valid_point) {
                                result[point_id] = local_sum;
                            }
                        });
                };
            });
        result_buf.synchronize();
    });

    logger::raw_ln(
        "result_trivial_gpu_teams :",
        shambase::nanosec_to_time_str(time_result_trivial_gpu_teams * 1e9));

    REQUIRE(compare(reference, result_buf.copy_to_stdvec(), Tscal{1e-12}));
}
