// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/details/reduction/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/USMPtrHolder.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <vector>

void fill(sham::DeviceQueue &q, sham::DeviceBuffer<int> &f_a, int value) {

    sham::EventList depends_list;

    int *a = f_a.get_write_access(depends_list);

    sycl::event e = q.submit(depends_list, [&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(1000), [=](sycl::id<1> indx) {
            a[indx] = value;
        });
    });

    f_a.complete_event_state(e);
}

void add(
    sham::DeviceQueue &q,
    sham::DeviceBuffer<int> &f_a,
    sham::DeviceBuffer<int> &f_b,
    sham::DeviceBuffer<int> &f_c) {

    sham::EventList depends_list;

    const int *a = f_a.get_read_access(depends_list);
    const int *b = f_b.get_read_access(depends_list);
    int *c       = f_c.get_write_access(depends_list);

    sycl::event e = q.submit(depends_list, [&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(1000), [=](sycl::id<1> indx) {
            c[indx] = b[indx] + a[indx];
        });
    });

    f_a.complete_event_state(e);
    f_b.complete_event_state(e);
    f_c.complete_event_state(e);
}

TestStart(Unittest, "shambackends/DeviceBuffer:smalltaskgraph", DeviceBuffer_small_task_graph, 1) {

    std::shared_ptr<sham::DeviceScheduler> dev_sched
        = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<int> a{1000, dev_sched};
    sham::DeviceBuffer<int> b{1000, dev_sched};
    sham::DeviceBuffer<int> c{1000, dev_sched};

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    fill(q, a, 77);
    fill(q, b, 33);

    add(q, a, b, c);
}

TestStart(Unittest, "shambackends/DeviceBuffer:copy_to_stdvec", devbuf_testcopy_to_stdvec, 1) {
    using T = int;

    std::shared_ptr<sham::DeviceScheduler> dev_sched
        = shamsys::instance::get_compute_scheduler_ptr();

    // mock a vector
    std::vector<T> v1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    sham::DeviceBuffer<T> b(v1.size(), dev_sched);
    b.copy_from_stdvec(v1);

    std::vector<T> v2 = b.copy_to_stdvec();

    REQUIRE_EQUAL(b.get_size(), v1.size());
    REQUIRE_EQUAL(v2.size(), v1.size());
    for (size_t i = 0; i < b.get_size(); ++i) {
        REQUIRE_EQUAL(v2[i], v1[i]);
    }
}

TestStart(
    Unittest, "shambackends/DeviceBuffer:copy_to_sycl_buffer", devbuf_testcopy_to_sycl_buffer, 1) {
    using T = int;

    std::shared_ptr<sham::DeviceScheduler> dev_sched
        = shamsys::instance::get_compute_scheduler_ptr();

    sycl::queue &q = dev_sched->get_queue().q;

    // mock a sycl buffer
    sycl::buffer<T> b1(10);
    {
        sycl::host_accessor acc(b1, sycl::write_only, sycl::no_init);
        for (size_t i = 0; i < 10; ++i) {
            acc[i] = i;
        }
    }

    sham::DeviceBuffer<T> b(10, dev_sched);
    b.copy_from_sycl_buffer(b1);

    sycl::buffer<T> b2 = b.copy_to_sycl_buffer();

    REQUIRE_EQUAL(b1.size(), b2.size());
    REQUIRE_EQUAL(b.get_size(), b1.size());

    shamalgs::reduction::equals(q, b1, b2);
}

TestStart(Unittest, "shambackends/DeviceBuffer:fill", DeviceBuffer_fill1, 1) {
    std::shared_ptr<sham::DeviceScheduler> dev_sched
        = shamsys::instance::get_compute_scheduler_ptr();

    // Create a device buffer with size 10
    sham::DeviceBuffer<int> buffer(10, dev_sched);

    // Fill the buffer with value 5
    buffer.fill(5);

    {
        std::vector<int> b = buffer.copy_to_stdvec();
        REQUIRE_EQUAL(b.size(), 10);

        // Check that the buffer is filled with the correct value
        for (int i = 0; i < 10; i++) {
            REQUIRE_EQUAL(b[i], 5);
        }
    }
}

TestStart(Unittest, "shambackends/DeviceBuffer:fill(with count)", DeviceBuffer_fill2, 1) {
    std::shared_ptr<sham::DeviceScheduler> dev_sched
        = shamsys::instance::get_compute_scheduler_ptr();

    // Create a device buffer with size 10
    sham::DeviceBuffer<int> buffer(10, dev_sched);

    // Fill the buffer with value 5
    buffer.fill(0);

    // Fill the buffer with value 5, starting from index 2, with count 5
    buffer.fill(5, {2, 7});

    {
        std::vector<int> b = buffer.copy_to_stdvec();
        REQUIRE(b.size() == 10);

        // Check that the buffer is filled with the correct value
        for (int i = 0; i < 10; i++) {
            if (i >= 2 && i < 7) {
                REQUIRE_EQUAL(b[i], 5);
            } else {
                REQUIRE_EQUAL(b[i], 0);
            }
        }
    }
}

TestStart(Unittest, "shambackends/DeviceBuffer:fill(exception)", DeviceBuffer_fill3, 1) {
    std::shared_ptr<sham::DeviceScheduler> dev_sched
        = shamsys::instance::get_compute_scheduler_ptr();

    // Create a device buffer with size 10
    sham::DeviceBuffer<int> buffer(10, dev_sched);

    // Try to fill the buffer with value 5, starting from index 15, with count 5
    REQUIRE_EXCEPTION_THROW(buffer.fill(5, {15, 20}), std::invalid_argument);
}

TestStart(Unittest, "shambackends/DeviceBuffer:resize", DeviceBuffer_resize, 1) {

    std::shared_ptr<sham::DeviceScheduler> dev_sched
        = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<int> a{1000, dev_sched};
    a.fill(77);

    a.resize(2000);

    REQUIRE(a.get_size() == 2000);
    REQUIRE(a.get_mem_usage() >= a.to_bytesize(2000));

    {
        std::vector<int> b = a.copy_to_stdvec();
        REQUIRE(b.size() == 2000);

        for (int i = 0; i < 1000; i++) {
            REQUIRE(b[i] == 77);
        }
    }
}

TestStart(
    Unittest,
    "shambackends/DeviceBuffer:copy_to_stdvec_idx_range",
    DeviceBuffer_copy_to_stdvec_idx_range,
    1) {

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<int> buffer(10, dev_sched);
    std::vector<int> v1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    buffer.copy_from_stdvec(v1);

    {
        std::vector<int> v2 = buffer.copy_to_stdvec_idx_range(2, 5);
        REQUIRE_EQUAL(v2.size(), 3);
        REQUIRE_EQUAL(v2[0], 2);
        REQUIRE_EQUAL(v2[1], 3);
        REQUIRE_EQUAL(v2[2], 4);
    }

    {
        std::vector<int> v2 = buffer.copy_to_stdvec_idx_range(0, 5);
        REQUIRE_EQUAL(v2.size(), 5);
        REQUIRE_EQUAL(v2[0], 0);
        REQUIRE_EQUAL(v2[1], 1);
        REQUIRE_EQUAL(v2[2], 2);
        REQUIRE_EQUAL(v2[3], 3);
        REQUIRE_EQUAL(v2[4], 4);
    }

    REQUIRE_EXCEPTION_THROW(
        [[maybe_unused]] auto _ = buffer.copy_to_stdvec_idx_range(5, 2), std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(
        [[maybe_unused]] auto _ = buffer.copy_to_stdvec_idx_range(0, 11), std::invalid_argument);
}

TestStart(Unittest, "shambackends/DeviceBuffer:get_val_at_idx", DeviceBuffer_get_val_at_idx, 1) {
    std::shared_ptr<sham::DeviceScheduler> dev_sched
        = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<int> buffer(10, dev_sched);
    std::vector<int> init_values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    buffer.copy_from_stdvec(init_values);

    for (size_t i = 0; i < init_values.size(); ++i) {
        REQUIRE_EQUAL(buffer.get_val_at_idx(i), init_values[i]);
    }

    REQUIRE_EXCEPTION_THROW(buffer.get_val_at_idx(10), std::invalid_argument);
}

TestStart(Unittest, "shambackends/DeviceBuffer:set_val_at_idx", DeviceBuffer_set_val_at_idx, 1) {
    std::shared_ptr<sham::DeviceScheduler> dev_sched
        = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<int> buffer(10, dev_sched);
    std::vector<int> init_values     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> expected_values = {0, 5, 2, 3, 4, 5, 6, 7, 8, 9};

    buffer.copy_from_stdvec(init_values);

    buffer.set_val_at_idx(1, 5);

    REQUIRE_EQUAL(buffer.copy_to_stdvec(), expected_values);

    REQUIRE_EXCEPTION_THROW(buffer.set_val_at_idx(10, 5), std::invalid_argument);
}
