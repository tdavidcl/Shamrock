// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/details/numeric/exclusiveScanAtomic.hpp"
#include "shamalgs/details/numeric/exclusiveScanGPUGems39.hpp"
#include "shamalgs/details/numeric/numericFallback.hpp"
#include "shamalgs/details/numeric/scanDecoupledLookback.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/numeric.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/shamtest.hpp"
#include <vector>

template<class T>
inline bool validate_scan(
    sham::DeviceBuffer<T> &result,
    sham::DeviceBuffer<T> &ref,
    sham::DeviceBuffer<u32> &result_flag) {

    result_flag.fill(1_u32);

    sham::kernel_call(
        result.get_dev_scheduler_ptr()->get_queue(),
        sham::MultiRef{result, ref},
        sham::MultiRef{result_flag},
        result.get_size(),
        [](u32 i, const T *result, const T *ref, u32 *result_flag) {
            if (result[i] != ref[i]) {
                sycl::atomic_ref<
                    u32,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::global_space>
                    atom(result_flag[0]);
                atom.store(0);
            }
        });

    return result_flag.get_val_at_idx(0) == 1;
}

template<class T>
inline sham::DeviceBuffer<T> make_ref(sham::DeviceBuffer<T> &in) {
    std::vector<T> host_data = in.copy_to_stdvec();
    std::exclusive_scan(host_data.begin(), host_data.end(), host_data.begin(), 0);
    sham::DeviceBuffer<T> ref(host_data.size(), in.get_dev_scheduler_ptr());
    ref.copy_from_stdvec(host_data);
    return ref;
}

template<class T>
inline u64 burn_scan(sham::DeviceBuffer<T> &in) {

    auto dev_sched = in.get_dev_scheduler_ptr();

    auto ref = make_ref(in);

    auto source_buf  = sham::DeviceBuffer<T>(in.get_size(), dev_sched);
    auto result_flag = sham::DeviceBuffer<u32>(1, dev_sched);

    shambase::Timer timer;
    timer.start();
    timer.end();

    f64 last_print = 0;
    u32 i          = 0;
    for (; timer.elasped_sec() < 5; i++) {
        source_buf.copy_from(in);

        auto result   = shamalgs::numeric::scan_exclusive(dev_sched, source_buf, in.get_size());
        bool is_valid = validate_scan(result, ref, result_flag);
        if (!is_valid) {
            throw std::runtime_error("Scan is not valid");
        }

        if (i % 10 == 0) {
            timer.end();
        }
    }

    timer.end();
    // logger::raw_ln(
    //     shambase::format(
    //         "Burn scan test on world rank {} done in {}s, run {} times (size={})",
    //         shamcomm::world_rank(),
    //         timer.elasped_sec(),
    //         i,
    //         in.get_size()));

    return i;
}

template<class T>
inline u64 fuzz_burn(std::mt19937_64 &eng) {

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    u32 random_size = shamalgs::primitives::mock_value<u32>(eng, 1, 1e8);
    auto in         = shamalgs::primitives::mock_vector<T>(eng(), random_size, 0, 10);

    std::string log_str = shambase::format(
        "Burn scan test on world rank {}, seed={}, size={}",
        shamcomm::world_rank(),
        eng(),
        random_size);
    // logger::raw_ln(log_str);

    sham::DeviceBuffer<T> in_buf(in.size(), dev_sched);
    in_buf.copy_from_stdvec(in);

    u64 run_count = burn_scan(in_buf);

    return shamalgs::collective::allreduce_sum(run_count);
}

TestStart(Unittest, "burn_scan", test_burn_scan, -1) {

    std::mt19937_64 eng(0x111 + shamcomm::world_rank() * 1000);

    u64 total_run_count = 0;
    for (u32 i = 0; true; i++) {
        shamcomm::mpi::Barrier(MPI_COMM_WORLD);
        if (shamcomm::world_rank() == 0) {
            logger::raw_ln(
                shambase::format("----- run {} | total run count = {} -----", i, total_run_count));
        }
        shamcomm::mpi::Barrier(MPI_COMM_WORLD);
        total_run_count += fuzz_burn<u32>(eng);
    }
}
