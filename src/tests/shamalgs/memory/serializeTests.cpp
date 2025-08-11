// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/random.hpp"
#include "shamalgs/serialize.hpp"
#include "shambackends/math.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

template<class T>
inline void check_buf(std::string prefix, sycl::buffer<T> &b1, sycl::buffer<T> &b2) {

    REQUIRE_EQUAL_NAMED(prefix + std::string("same size"), b1.size(), b2.size());

    {
        sycl::host_accessor acc1{b1};
        sycl::host_accessor acc2{b2};

        std::string id_err_list = "errors in id : ";

        bool eq = true;
        for (u32 i = 0; i < b1.size(); i++) {
            if (!sham::equals(acc1[i], acc2[i])) {
                eq = false;
                // id_err_list += std::to_string(i) + " ";
            }
        }

        if (eq) {
            REQUIRE_NAMED("same content", eq);
        } else {
            shamtest::asserts().assert_add_comment("same content", eq, id_err_list);
        }
    }
}

TestStart(Unittest, "shamalgs/memory/SerializeHelper:base_test", test_serialize_helper_base, 1) {

    u32 n1                     = 100;
    sycl::buffer<u8> buf_comp1 = shamalgs::random::mock_buffer<u8>(0x111, n1);

    f64_16 test_val = f64_16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    std::string test_str = "physics phd they said";

    u32 n2                        = 100;
    sycl::buffer<u32_3> buf_comp2 = shamalgs::random::mock_buffer<u32_3>(0x121, n2);

    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());

    shamalgs::SerializeSize bytelen
        = ser.serialize_byte_size<u8>(n1) + ser.serialize_byte_size<f64_16>()
          + ser.serialize_byte_size<u32_3>(n2) + ser.serialize_byte_size(test_str);

    ser.allocate(bytelen);
    ser.write_buf(buf_comp1, n1);
    ser.write(test_val);
    ser.write(test_str);
    ser.write_buf(buf_comp2, n2);

    logger::raw_ln("writing done");

    auto recov = ser.finalize();

    {
        sycl::buffer<u8> buf1(n1);
        f64_16 val;
        std::string recv_str;
        sycl::buffer<u32_3> buf2(n2);

        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));

        logger::raw_ln("load 1 ");
        ser2.load_buf(buf1, n1);
        logger::raw_ln("load 1 done");
        ser2.load(val);
        logger::raw_ln("load 2 done");
        ser2.load(recv_str);
        logger::raw_ln("load 3 done");
        ser2.load_buf(buf2, n2);
        logger::raw_ln("load 4 done");

        // shamalgs::memory::print_buf(buf_comp1, n1, 16, "{} ");
        // shamalgs::memory::print_buf(buf1, n1, 16, "{} ");

        REQUIRE_NAMED("same", sham::equals(val, test_val));
        REQUIRE_NAMED("same", test_str == recv_str);
        check_buf("buf 1", buf_comp1, buf1);
        check_buf("buf 2", buf_comp2, buf2);
    }
}

TestStart(Unittest, "shamalgs/memory/SerializeHelper:primitive_types", test_serialize_helper_primitive_types, 1) {

        shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
        
        // Test all primitive types
        u8 test_u8 = 42_u8;
        u16 test_u16 = 12345_u16;
        u32 test_u32 = 1234567890_u32;
        u64 test_u64 = 12345678901234567890_u64;
        i8 test_i8 = -42_i8;
        i16 test_i16 = -12345_i16;
        i32 test_i32 = -1234567890_i32;
        i64 test_i64 = -1234567890123456789_i64;
        f16 test_f16 = f16(3.14);
        f32 test_f32 = 3.14159_f32;
        f64 test_f64 = 3.14159265359;
        
        shamalgs::SerializeSize bytelen = ser.serialize_byte_size<u8>() + ser.serialize_byte_size<u16>() +
                                          ser.serialize_byte_size<u32>() + ser.serialize_byte_size<u64>() +
                                          ser.serialize_byte_size<i8>() + ser.serialize_byte_size<i16>() +
                                          ser.serialize_byte_size<i32>() + ser.serialize_byte_size<i64>() +
                                          ser.serialize_byte_size<f16>() + ser.serialize_byte_size<f32>() +
                                          ser.serialize_byte_size<f64>();
        
        ser.allocate(bytelen);
        ser.write(test_u8);
        ser.write(test_u16);
        ser.write(test_u32);
        ser.write(test_u64);
        ser.write(test_i8);
        ser.write(test_i16);
        ser.write(test_i32);
        ser.write(test_i64);
        ser.write(test_f16);
        ser.write(test_f32);
        ser.write(test_f64);
        
        auto recov = ser.finalize();
        
        {
            shamalgs::SerializeHelper ser2(shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));
            
            u8 recv_u8;
            u16 recv_u16;
            u32 recv_u32;
            u64 recv_u64;
            i8 recv_i8;
            i16 recv_i16;
            i32 recv_i32;
            i64 recv_i64;
            f16 recv_f16;
            f32 recv_f32;
            f64 recv_f64;
            
            ser2.load(recv_u8);
            ser2.load(recv_u16);
            ser2.load(recv_u32);
            ser2.load(recv_u64);
            ser2.load(recv_i8);
            ser2.load(recv_i16);
            ser2.load(recv_i32);
            ser2.load(recv_i64);
            ser2.load(recv_f16);
            ser2.load(recv_f32);
            ser2.load(recv_f64);
            
            REQUIRE_NAMED("u8 same", recv_u8 == test_u8);
            REQUIRE_NAMED("u16 same", recv_u16 == test_u16);
            REQUIRE_NAMED("u32 same", recv_u32 == test_u32);
            REQUIRE_NAMED("u64 same", recv_u64 == test_u64);
            REQUIRE_NAMED("i8 same", recv_i8 == test_i8);
            REQUIRE_NAMED("i16 same", recv_i16 == test_i16);
            REQUIRE_NAMED("i32 same", recv_i32 == test_i32);
            REQUIRE_NAMED("i64 same", recv_i64 == test_i64);
            REQUIRE_NAMED("f16 same", sham::equals(recv_f16, test_f16));
            REQUIRE_NAMED("f32 same", sham::equals(recv_f32, test_f32));
            REQUIRE_NAMED("f64 same", sham::equals(recv_f64, test_f64));
        }
    
}

TestStart(Unittest, "shamalgs/memory/SerializeHelper:vector_types", test_serialize_helper_vector_types, 1) {

        shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
        
        // Test various vector types
        u32_2 test_u32_2 = {42_u32, 123_u32};
        u32_3 test_u32_3 = {1_u32, 2_u32, 3_u32};
        u32_4 test_u32_4 = {10_u32, 20_u32, 30_u32, 40_u32};
        u32_8 test_u32_8 = {100_u32, 200_u32, 300_u32, 400_u32, 500_u32, 600_u32, 700_u32, 800_u32};
        u32_16 test_u32_16 = {1000_u32, 2000_u32, 3000_u32, 4000_u32, 5000_u32, 6000_u32, 7000_u32, 8000_u32,
                              9000_u32, 10000_u32, 11000_u32, 12000_u32, 13000_u32, 14000_u32, 15000_u32, 16000_u32};
        
        f64_2 test_f64_2 = {3.14, 2.718};
        f64_3 test_f64_3 = {1.0, 2.0, 3.0};
        f64_4 test_f64_4 = {1.1, 2.2, 3.3, 4.4};
        f64_8 test_f64_8 = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
        f64_16 test_f64_16 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6};
        
        shamalgs::SerializeSize bytelen = ser.serialize_byte_size<u32_2>() + ser.serialize_byte_size<u32_3>() +
                                          ser.serialize_byte_size<u32_4>() + ser.serialize_byte_size<u32_8>() +
                                          ser.serialize_byte_size<u32_16>() + ser.serialize_byte_size<f64_2>() +
                                          ser.serialize_byte_size<f64_3>() + ser.serialize_byte_size<f64_4>() +
                                          ser.serialize_byte_size<f64_8>() + ser.serialize_byte_size<f64_16>();
        
        ser.allocate(bytelen);
        ser.write(test_u32_2);
        ser.write(test_u32_3);
        ser.write(test_u32_4);
        ser.write(test_u32_8);
        ser.write(test_u32_16);
        ser.write(test_f64_2);
        ser.write(test_f64_3);
        ser.write(test_f64_4);
        ser.write(test_f64_8);
        ser.write(test_f64_16);
        
        auto recov = ser.finalize();
        
        {
            shamalgs::SerializeHelper ser2(shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));
            
            u32_2 recv_u32_2;
            u32_3 recv_u32_3;
            u32_4 recv_u32_4;
            u32_8 recv_u32_8;
            u32_16 recv_u32_16;
            f64_2 recv_f64_2;
            f64_3 recv_f64_3;
            f64_4 recv_f64_4;
            f64_8 recv_f64_8;
            f64_16 recv_f64_16;
            
            ser2.load(recv_u32_2);
            ser2.load(recv_u32_3);
            ser2.load(recv_u32_4);
            ser2.load(recv_u32_8);
            ser2.load(recv_u32_16);
            ser2.load(recv_f64_2);
            ser2.load(recv_f64_3);
            ser2.load(recv_f64_4);
            ser2.load(recv_f64_8);
            ser2.load(recv_f64_16);
            
            REQUIRE_NAMED("u32_2 same", sham::equals(recv_u32_2, test_u32_2));
            REQUIRE_NAMED("u32_3 same", sham::equals(recv_u32_3, test_u32_3));
            REQUIRE_NAMED("u32_4 same", sham::equals(recv_u32_4, test_u32_4));
            REQUIRE_NAMED("u32_8 same", sham::equals(recv_u32_8, test_u32_8));
            REQUIRE_NAMED("u32_16 same", sham::equals(recv_u32_16, test_u32_16));
            REQUIRE_NAMED("f64_2 same", sham::equals(recv_f64_2, test_f64_2));
            REQUIRE_NAMED("f64_3 same", sham::equals(recv_f64_3, test_f64_3));
            REQUIRE_NAMED("f64_4 same", sham::equals(recv_f64_4, test_f64_4));
            REQUIRE_NAMED("f64_8 same", sham::equals(recv_f64_8, test_f64_8));
            REQUIRE_NAMED("f64_16 same", sham::equals(recv_f64_16, test_f64_16));
        }

}

TestStart(Unittest, "shamalgs/memory/SerializeHelper:device_buffer_operations", test_serialize_helper_device_buffer_operations, 1) {

        shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
        
        // Test DeviceBuffer operations
        u32 n1 = 50;
        u32 n2 = 75;
        sham::DeviceBuffer<f32> dev_buf1(n1, ser.get_device_scheduler());
        sham::DeviceBuffer<u64_3> dev_buf2(n2, ser.get_device_scheduler());
        
        // Fill device buffers with test data
        {
            sycl::buffer<f32> host_buf1 = shamalgs::random::mock_buffer<f32>(0x222, n1);
            sycl::buffer<u64_3> host_buf2 = shamalgs::random::mock_buffer<u64_3>(0x333, n2);
            
            sham::EventList depends_list;
            f32* dev_acc1 = dev_buf1.get_write_access(depends_list);
            u64_3* dev_acc2 = dev_buf2.get_write_access(depends_list);
            
            auto e = ser.get_device_scheduler()->get_queue().submit(depends_list, [&](sycl::handler& cgh) {
                sycl::accessor host_acc1{host_buf1, cgh, sycl::read_only};
                sycl::accessor host_acc2{host_buf2, cgh, sycl::read_only};
                
                cgh.parallel_for(sycl::range<1>{n1}, [=](sycl::item<1> id) {
                    dev_acc1[id] = host_acc1[id];
                });
                
                cgh.parallel_for(sycl::range<1>{n2}, [=](sycl::item<1> id) {
                    dev_acc2[id] = host_acc2[id];
                });
            });
            
            dev_buf1.complete_event_state(e);
            dev_buf2.complete_event_state(e);
            e.wait();
        }
        
        shamalgs::SerializeSize bytelen = ser.serialize_byte_size<f32>(n1) + ser.serialize_byte_size<u64_3>(n2);
        
        ser.allocate(bytelen);
        ser.write_buf(dev_buf1, n1);
        ser.write_buf(dev_buf2, n2);
        
        auto recov = ser.finalize();
        
        {
            shamalgs::SerializeHelper ser2(shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));
            
            sham::DeviceBuffer<f32> recv_dev_buf1(n1, ser2.get_device_scheduler());
            sham::DeviceBuffer<u64_3> recv_dev_buf2(n2, ser2.get_device_scheduler());
            
            ser2.load_buf(recv_dev_buf1, n1);
            ser2.load_buf(recv_dev_buf2, n2);
            
            // Compare the device buffers by copying to host
            {
                sycl::buffer<f32> host_buf1(n1);
                sycl::buffer<f32> host_buf2(n1);
                sycl::buffer<u64_3> host_buf3(n2);
                sycl::buffer<u64_3> host_buf4(n2);
                
                sham::EventList depends_list;
                const f32* dev_acc1 = dev_buf1.get_read_access(depends_list);
                const f32* dev_acc2 = recv_dev_buf1.get_read_access(depends_list);
                const u64_3* dev_acc3 = dev_buf2.get_read_access(depends_list);
                const u64_3* dev_acc4 = recv_dev_buf2.get_read_access(depends_list);
                
                auto e = ser2.get_device_scheduler()->get_queue().submit(depends_list, [&](sycl::handler& cgh) {
                    sycl::accessor host_acc1{host_buf1, cgh, sycl::write_only, sycl::no_init};
                    sycl::accessor host_acc2{host_buf2, cgh, sycl::write_only, sycl::no_init};
                    sycl::accessor host_acc3{host_buf3, cgh, sycl::write_only, sycl::no_init};
                    sycl::accessor host_acc4{host_buf4, cgh, sycl::write_only, sycl::no_init};
                    
                    cgh.parallel_for(sycl::range<1>{n1}, [=](sycl::item<1> id) {
                        host_acc1[id] = dev_acc1[id];
                        host_acc2[id] = dev_acc2[id];
                    });
                    
                    cgh.parallel_for(sycl::range<1>{n2}, [=](sycl::item<1> id) {
                        host_acc3[id] = dev_acc3[id];
                        host_acc4[id] = dev_acc4[id];
                    });
                });
                
                e.wait();
                
                check_buf("device buffer f32", host_buf1, host_buf2);
                check_buf("device buffer u64_3", host_buf3, host_buf4);
            }
        }

}

TestStart(Unittest, "shamalgs/memory/SerializeHelper:serialize_size_operations", test_serialize_helper_serialize_size_operations, 1) {

        // Test SerializeSize arithmetic operations
        shamalgs::SerializeSize size1{100, 200};
        shamalgs::SerializeSize size2{50, 75};
        
        // Test addition
        shamalgs::SerializeSize sum = size1 + size2;
        REQUIRE_NAMED("head size addition", sum.head_size == 150);
        REQUIRE_NAMED("content size addition", sum.content_size == 275);
        
        // Test compound addition
        shamalgs::SerializeSize size3{25, 125};
        size3 += size1;
        REQUIRE_NAMED("compound head size addition", size3.head_size == 125);
        REQUIRE_NAMED("compound content size addition", size3.content_size == 325);
        
        // Test multiplication
        shamalgs::SerializeSize product = size1 * size2;
        REQUIRE_NAMED("head size multiplication", product.head_size == 5000);
        REQUIRE_NAMED("content size multiplication", product.content_size == 15000);
        
        // Test compound multiplication
        shamalgs::SerializeSize size4{10, 20};
        size4 *= size2;
        REQUIRE_NAMED("compound head size multiplication", size4.head_size == 500);
        REQUIRE_NAMED("compound content size multiplication", size4.content_size == 1500);

}

TestStart(Unittest, "shamalgs/memory/SerializeHelper:edge_cases", test_serialize_helper_edge_cases, 1) {
{
        shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
        
        // Test empty string
        std::string empty_str = "";
        shamalgs::SerializeSize bytelen = ser.serialize_byte_size(empty_str);
        ser.allocate(bytelen);
        ser.write(empty_str);
        
        auto recov = ser.finalize();
        
        {
            shamalgs::SerializeHelper ser2(shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));
            std::string recv_str;
            ser2.load(recv_str);
            REQUIRE_NAMED("empty string preserved", recv_str == empty_str);
        }
    }
    
    {
        shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
        
        // Test single element buffers
        sycl::buffer<f64> single_buf(1);
        {
            sycl::host_accessor acc{single_buf, sycl::write_only, sycl::no_init};
            acc[0] = 42.0;
        }
        
        shamalgs::SerializeSize bytelen = ser.serialize_byte_size<f64>(1);
        ser.allocate(bytelen);
        ser.write_buf(single_buf, 1);
        
        auto recov = ser.finalize();
        
        {
            shamalgs::SerializeHelper ser2(shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));
            sycl::buffer<f64> recv_buf(1);
            ser2.load_buf(recv_buf, 1);
            
            {
                sycl::host_accessor acc1{single_buf, sycl::read_only};
                sycl::host_accessor acc2{recv_buf, sycl::read_only};
                REQUIRE_NAMED("single element preserved", sham::equals(acc1[0], acc2[0]));
            }
        }
    }
}

TestStart(Unittest, "shamalgs/memory/SerializeHelper:mixed_types_comprehensive", test_serialize_helper_mixed_types_comprehensive, 1) {

        shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
        
        // Test a comprehensive mix of types
        u8 val1 = 255_u8;
        u16 val2 = 65535_u16;
        u32 val3 = 4294967295_u32;
        u64 val4 = 18446744073709551615_u64;
        i8 val5 = -128_i8;
        i16 val6 = -32768_i16;
        i32 val7 = -2147483648_i32;
        i64 val8 = -9223372036854775808_i64;
        f16 val9 = f16(65504.0);  // max f16
        f32 val10 = 3.402823e38_f32;  // max f32
        f64 val11 = 1.7976931348623157e308;  // max f64
        
        u32_2 val12 = {42_u32, 123_u32};
        u32_3 val13 = {1_u32, 2_u32, 3_u32};
        f64_4 val14 = {1.1, 2.2, 3.3, 4.4};
        
        std::string val15 = "comprehensive test string with special chars: !@#$%^&*()";
        
        u32 n1 = 25;
        u32 n2 = 50;
        sycl::buffer<f32> buf1 = shamalgs::random::mock_buffer<f32>(0x444, n1);
        sycl::buffer<u64_3> buf2 = shamalgs::random::mock_buffer<u64_3>(0x555, n2);
        
        shamalgs::SerializeSize bytelen = ser.serialize_byte_size<u8>() + ser.serialize_byte_size<u16>() +
                                          ser.serialize_byte_size<u32>() + ser.serialize_byte_size<u64>() +
                                          ser.serialize_byte_size<i8>() + ser.serialize_byte_size<i16>() +
                                          ser.serialize_byte_size<i32>() + ser.serialize_byte_size<i64>() +
                                          ser.serialize_byte_size<f16>() + ser.serialize_byte_size<f32>() +
                                          ser.serialize_byte_size<f64>() + ser.serialize_byte_size<u32_2>() +
                                          ser.serialize_byte_size<u32_3>() + ser.serialize_byte_size<f64_4>() +
                                          ser.serialize_byte_size(val15) + ser.serialize_byte_size<f32>(n1) +
                                          ser.serialize_byte_size<u64_3>(n2);
        
        ser.allocate(bytelen);
        ser.write(val1);
        ser.write(val2);
        ser.write(val3);
        ser.write(val4);
        ser.write(val5);
        ser.write(val6);
        ser.write(val7);
        ser.write(val8);
        ser.write(val9);
        ser.write(val10);
        ser.write(val11);
        ser.write(val12);
        ser.write(val13);
        ser.write(val14);
        ser.write(val15);
        ser.write_buf(buf1, n1);
        ser.write_buf(buf2, n2);
        
        auto recov = ser.finalize();
        
        {
            shamalgs::SerializeHelper ser2(shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));
            
            u8 recv1;
            u16 recv2;
            u32 recv3;
            u64 recv4;
            i8 recv5;
            i16 recv6;
            i32 recv7;
            i64 recv8;
            f16 recv9;
            f32 recv10;
            f64 recv11;
            u32_2 recv12;
            u32_3 recv13;
            f64_4 recv14;
            std::string recv15;
            sycl::buffer<f32> recv_buf1(n1);
            sycl::buffer<u64_3> recv_buf2(n2);
            
            ser2.load(recv1);
            ser2.load(recv2);
            ser2.load(recv3);
            ser2.load(recv4);
            ser2.load(recv5);
            ser2.load(recv6);
            ser2.load(recv7);
            ser2.load(recv8);
            ser2.load(recv9);
            ser2.load(recv10);
            ser2.load(recv11);
            ser2.load(recv12);
            ser2.load(recv13);
            ser2.load(recv14);
            ser2.load(recv15);
            ser2.load_buf(recv_buf1, n1);
            ser2.load_buf(recv_buf2, n2);
            
            REQUIRE_NAMED("u8 max preserved", recv1 == val1);
            REQUIRE_NAMED("u16 max preserved", recv2 == val2);
            REQUIRE_NAMED("u32 max preserved", recv3 == val3);
            REQUIRE_NAMED("u64 max preserved", recv4 == val4);
            REQUIRE_NAMED("i8 min preserved", recv5 == val5);
            REQUIRE_NAMED("i16 min preserved", recv6 == val6);
            REQUIRE_NAMED("i32 min preserved", recv7 == val7);
            REQUIRE_NAMED("i64 min preserved", recv8 == val8);
            REQUIRE_NAMED("f16 max preserved", sham::equals(recv9, val9));
            REQUIRE_NAMED("f32 max preserved", sham::equals(recv10, val10));
            REQUIRE_NAMED("f64 max preserved", sham::equals(recv11, val11));
            REQUIRE_NAMED("u32_2 preserved", sham::equals(recv12, val12));
            REQUIRE_NAMED("u32_3 preserved", sham::equals(recv13, val13));
            REQUIRE_NAMED("f64_4 preserved", sham::equals(recv14, val14));
            REQUIRE_NAMED("string preserved", recv15 == val15);
            check_buf("comprehensive f32", buf1, recv_buf1);
            check_buf("comprehensive u64_3", buf2, recv_buf2);
        }
    
}

TestStart(Benchmark, "shamalgs/memory/SerializeHelper:benchmark", bench_serializer, 1) {

    auto get_perf_knownsize = [](u32 buf_cnt, u32 buf_len) -> std::pair<f64, f64> {
        StackEntry stack{};
        std::vector<sycl::buffer<f64>> bufs;
        std::vector<sycl::buffer<f64>> bufs_ret;

        for (u32 i = 0; i < buf_cnt; i++) {
            bufs.emplace_back(shamalgs::random::mock_buffer<f64>(0x111 + i, buf_len));
        }

        shambase::Timer tser;
        tser.start();

        shamalgs::SerializeHelper ser1(shamsys::instance::get_compute_scheduler_ptr());
        shamalgs::SerializeSize sz = ser1.serialize_byte_size<f64>(buf_cnt * buf_len);
        ser1.allocate(sz);
        for (u32 i = 0; i < buf_cnt; i++) {
            ser1.write_buf(bufs[i], buf_len);
        }
        auto recov = ser1.finalize();
        shamsys::instance::get_compute_queue().wait();

        tser.end();

        shambase::Timer tdeser;
        tdeser.start();

        for (u32 i = 0; i < buf_cnt; i++) {
            bufs_ret.emplace_back(sycl::buffer<f64>(buf_len));
        }

        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));
        for (u32 i = 0; i < buf_cnt; i++) {
            ser2.load_buf(bufs_ret[i], buf_len);
        }
        shamsys::instance::get_compute_queue().wait();

        tdeser.end();

        return {
            sz.get_total_size() / (tser.nanosec / 1e9),
            sz.get_total_size() / (tdeser.nanosec / 1e9)};
    };
    auto get_perf_unknownsize = [](u32 buf_cnt, u32 buf_len) -> std::pair<f64, f64> {
        StackEntry stack{};
        std::vector<sycl::buffer<f64>> bufs;
        std::vector<sycl::buffer<f64>> bufs_ret;

        for (u32 i = 0; i < buf_cnt; i++) {
            bufs.emplace_back(shamalgs::random::mock_buffer<f64>(0x111 + i, buf_len));
        }

        shambase::Timer tser;
        tser.start();

        shamalgs::SerializeHelper ser1(shamsys::instance::get_compute_scheduler_ptr());
        shamalgs::SerializeSize sz = ser1.serialize_byte_size<f64>(buf_cnt * buf_len)
                                     + (ser1.serialize_byte_size<u32>() * buf_cnt);
        ser1.allocate(sz);
        for (u32 i = 0; i < buf_cnt; i++) {
            ser1.write(buf_len);
            ser1.write_buf(bufs[i], buf_len);
        }
        auto recov = ser1.finalize();
        shamsys::instance::get_compute_queue().wait();

        tser.end();

        shambase::Timer tdeser;
        tdeser.start();

        for (u32 i = 0; i < buf_cnt; i++) {
            bufs_ret.emplace_back(sycl::buffer<f64>(buf_len));
        }

        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));
        for (u32 i = 0; i < buf_cnt; i++) {
            u32 tmp;
            ser2.load(tmp);
            ser2.load_buf(bufs_ret[i], buf_len);
        }
        shamsys::instance::get_compute_queue().wait();

        tdeser.end();

        return {
            sz.get_total_size() / (tser.nanosec / 1e9),
            sz.get_total_size() / (tdeser.nanosec / 1e9)};
    };
    {
        std::vector<f64> x;
        std::vector<f64> tser, tdeser;
        std::vector<f64> tser_usz, tdeser_usz;

        for (u32 i = 1; i < 10000; i *= 2) {
            shamlog_debug_ln("Test", "i =", i);

            auto [p1, p2] = get_perf_knownsize(i, 100);
            auto [p3, p4] = get_perf_unknownsize(i, 100);
            x.push_back(i);
            tser.push_back(p1);
            tdeser.push_back(p2);
            tser_usz.push_back(p3);
            tdeser_usz.push_back(p4);
        }

        PyScriptHandle hdnl{};

        hdnl.data()["X"]         = x;
        hdnl.data()["ser_ksz"]   = tser;
        hdnl.data()["deser_ksz"] = tdeser;
        hdnl.data()["ser_usz"]   = tser_usz;
        hdnl.data()["deser_usz"] = tdeser_usz;

        hdnl.exec(R"py(
        import numpy as np
        import matplotlib.pyplot as plt

        plt.style.use('custom_style.mplstyle')

        fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(15,6))

        axs.plot(X,ser_ksz,'-',c = 'black',label = "serialize (kz)")
        axs.plot(X,deser_ksz,':',c = 'black',label = "deserialize (kz)")
        axs.plot(X,ser_usz,'--',c = 'black',label = "serialize (ukz)")
        axs.plot(X,deser_usz,'-.',c = 'black',label = "deserialize (ukz)")

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('buffer count')
        plt.ylabel('Bandwidth (B.s-1)')
        plt.legend()
        plt.tight_layout()

        plt.savefig("tests/figures/benchmark-serialize1.pdf")

    )py");

        TEX_REPORT(R"==(

        \begin{figure}[ht!]
        \center
        \includegraphics[width=0.95\linewidth]{tests/figures/benchmark-serialize1}
        \caption{Test the serializehelper performance (buf size = 100 f64)}
        \end{figure}

    )==")
    }

    {
        std::vector<f64> x;
        std::vector<f64> tser, tdeser;
        std::vector<f64> tser_usz, tdeser_usz;

        for (u32 i = 8; i < 10000; i *= 2) {
            shamlog_debug_ln("Test", "i =", i);

            auto [p1, p2] = get_perf_knownsize(1000, i);
            auto [p3, p4] = get_perf_unknownsize(1000, i);
            x.push_back(i);
            tser.push_back(p1);
            tdeser.push_back(p2);
            tser_usz.push_back(p3);
            tdeser_usz.push_back(p4);
        }

        PyScriptHandle hdnl{};

        hdnl.data()["X"]         = x;
        hdnl.data()["ser_ksz"]   = tser;
        hdnl.data()["deser_ksz"] = tdeser;
        hdnl.data()["ser_usz"]   = tser_usz;
        hdnl.data()["deser_usz"] = tdeser_usz;

        hdnl.exec(R"py(
        import numpy as np
        import matplotlib.pyplot as plt

        plt.style.use('custom_style.mplstyle')

        fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(15,6))

        axs.plot(X,ser_ksz,'-',c = 'black',label = "serialize (kz)")
        axs.plot(X,deser_ksz,':',c = 'black',label = "deserialize (kz)")
        axs.plot(X,ser_usz,'--',c = 'black',label = "serialize (ukz)")
        axs.plot(X,deser_usz,'-.',c = 'black',label = "deserialize (ukz)")

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('buffer size (f64)')
        plt.ylabel('Bandwidth (B.s-1)')
        plt.legend()
        plt.tight_layout()

        plt.savefig("tests/figures/benchmark-serialize2.pdf")

    )py");

        TEX_REPORT(R"==(

        \begin{figure}[ht!]
        \center
        \includegraphics[width=0.95\linewidth]{tests/figures/benchmark-serialize2}
        \caption{Test the serializehelper performance (buf count = 1000)}
        \end{figure}

    )==")
    }
}
