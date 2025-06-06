// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/StlContainerConversion.hpp"
#include "shamalgs/serialize.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <set>

TestStart(
    Unittest, "shamrock/patch/PatchDataField::serialize_buf", testpatchdatafieldserialize, 1) {

    u32 len                     = 1000;
    u32 nvar                    = 2;
    std::string name            = "testfield";
    PatchDataField<u32_3> field = PatchDataField<u32_3>::mock_field(0x111, len, name, nvar);

    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());

    ser.allocate(field.serialize_buf_byte_size());
    field.serialize_buf(ser);

    auto recov = ser.finalize();

    {
        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));

        PatchDataField<u32_3> buf2 = PatchDataField<u32_3>::deserialize_buf(ser2, name, nvar);

        REQUIRE_NAMED("input match out", field.check_field_match(buf2));
    }
}

TestStart(
    Unittest, "shamrock/patch/PatchDataField::serialize_full", testpatchdatafieldserializefull, 1) {

    u32 len                     = 1000;
    u32 nvar                    = 2;
    std::string name            = "testfield";
    PatchDataField<u32_3> field = PatchDataField<u32_3>::mock_field(0x111, len, name, nvar);

    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());

    ser.allocate(field.serialize_full_byte_size());
    field.serialize_full(ser);

    auto recov = ser.finalize();

    {
        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));

        PatchDataField<u32_3> buf2 = PatchDataField<u32_3>::deserialize_full(ser2);

        REQUIRE_NAMED("input match out", field.check_field_match(buf2));
    }
}

inline void check_pdat_get_ids_where(u32 len, u32 nvar, std::string name, f64 vmin, f64 vmax) {

    PatchDataField<f64> field = PatchDataField<f64>::mock_field(0x111, len, name, nvar, 0, 2000);

    std::set<u32> idx_cd = field.get_ids_set_where(
        [](auto access, u32 id, f64 vmin, f64 vmax) {
            f64 tmp = access[id];
            return tmp > vmin && tmp < vmax;
        },
        vmin,
        vmax);

    logger::raw_ln("found : ", idx_cd.size());

    std::vector<u32> idx_cd_vec = field.get_ids_vec_where(
        [](auto access, u32 id, f64 vmin, f64 vmax) {
            f64 tmp = access[id];
            return tmp > vmin && tmp < vmax;
        },
        vmin,
        vmax);

    logger::raw_ln("found : ", idx_cd_vec.size());

    auto idx_cd_sycl = field.get_ids_buf_where(
        [](auto access, u32 id, f64 vmin, f64 vmax) {
            f64 tmp = access[id];
            return tmp > vmin && tmp < vmax;
        },
        vmin,
        vmax);

    logger::raw_ln("found : ", std::get<1>(idx_cd_sycl));

    // compare content
    REQUIRE(bool(std::get<0>(idx_cd_sycl)) == (idx_cd.size() != 0));

    if (std::get<0>(idx_cd_sycl)) {
        REQUIRE(idx_cd == shambase::set_from_vector(idx_cd_vec));
        REQUIRE(
            idx_cd
            == shambase::set_from_vector(
                shamalgs::memory::buf_to_vec(*std::get<0>(idx_cd_sycl), std::get<1>(idx_cd_sycl))));
    }
}

TestStart(Unittest, "shamrock/patch/PatchDataField::get_ids_..._where", testgetelemwithrange, 1) {

    {
        u32 len          = 10000;
        u32 nvar         = 1;
        std::string name = "testfield";
        f64 vmin         = 0;
        f64 vmax         = 1000;

        check_pdat_get_ids_where(len, nvar, name, vmin, vmax);
    }
    {
        u32 len          = 0;
        u32 nvar         = 1;
        std::string name = "testfield";
        f64 vmin         = 0;
        f64 vmax         = 1000;

        check_pdat_get_ids_where(len, nvar, name, vmin, vmax);
    }
}
