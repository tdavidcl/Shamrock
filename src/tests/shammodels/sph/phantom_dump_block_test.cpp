// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/fortran_io.hpp"
#include "shambase/string.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shamtest/shamtest.hpp"
#include <string>
#include <utility>
#include <vector>

// Phantom stores one `dustfrac` array per dust bin, all with the same tag
NEW_TEST(Unittest, "shammodels/sph/phantom-dump-block-repeated-tags", 1) {
    using namespace shammodels::sph;

    PhantomDumpBlock block{};

    u64 id0 = block.add_new_fort_real("dustfrac");
    u64 id1 = block.add_new_fort_real("dustfrac");
    u64 id2 = block.add_new_fort_real("dustfrac");

    REQUIRE_EQUAL(id0, 0_u64);
    REQUIRE_EQUAL(id1, 1_u64);
    REQUIRE_EQUAL(id2, 2_u64);
    REQUIRE_EQUAL(block.blocks_fort_real.size(), 3_u64);

    std::string padded = sham::format("{:16s}", "dustfrac");
    for (auto &arr : block.blocks_fort_real) {
        REQUIRE_EQUAL(arr.tag, padded);
    }

    // with repeated tags get_ref_* returns the first array
    REQUIRE_EQUAL(block.get_ref_fort_real("dustfrac"), id0);
    REQUIRE_EQUAL(block.blocks_fort_real.size(), 3_u64);

    // get_ref_* is still find-or-create
    u64 idx = block.get_ref_fort_real("x");
    REQUIRE_EQUAL(idx, 3_u64);
    REQUIRE_EQUAL(block.get_ref_fort_real("x"), idx);
    REQUIRE_EQUAL(block.blocks_fort_real.size(), 4_u64);

    REQUIRE_EQUAL(block.add_new_f32("h"), 0_u64);
    REQUIRE_EQUAL(block.get_ref_f32("h"), 0_u64);
    REQUIRE_EQUAL(block.blocks_f32.size(), 1_u64);
}

NEW_TEST(Unittest, "shammodels/sph/phantom-dump-block-fetch-all-arrays", 1) {
    using namespace shammodels::sph;

    PhantomDumpBlock block{};

    // the f32 array is added first, but fort_real arrays come first in the dump type order
    u64 id_f32                    = block.add_new_f32("dustfrac");
    block.blocks_f32[id_f32].vals = {7.f, 8.f};

    std::vector<std::vector<f64>> ref = {{1., 2.}, {3., 4.}, {5., 6.}};
    for (auto &vals : ref) {
        u64 id                          = block.add_new_fort_real("dustfrac");
        block.blocks_fort_real[id].vals = vals;
    }
    block.blocks_fort_real[block.get_ref_fort_real("x")].vals = {-1., -2.};

    REQUIRE_EQUAL(block.count_arrays("dustfrac"), 4_u64);
    REQUIRE_EQUAL(block.count_arrays("x"), 1_u64);
    REQUIRE_EQUAL(block.count_arrays("missing"), 0_u64);

    std::vector<std::vector<f64>> dustfrac = block.fetch_all_arrays<f64>("dustfrac");
    REQUIRE_EQUAL(dustfrac.size(), 4_u64);
    REQUIRE_EQUAL(dustfrac[0], ref[0]);
    REQUIRE_EQUAL(dustfrac[1], ref[1]);
    REQUIRE_EQUAL(dustfrac[2], ref[2]);
    REQUIRE_EQUAL(dustfrac[3], (std::vector<f64>{7., 8.}));

    REQUIRE(block.fetch_all_arrays<f64>("missing").empty());

    // fill_vec still concatenates every matching array
    std::vector<f64> concat;
    block.fill_vec("dustfrac", concat);
    REQUIRE_EQUAL(concat, (std::vector<f64>{1., 2., 3., 4., 5., 6., 7., 8.}));
}

NEW_TEST(Unittest, "shammodels/sph/phantom-dump-block-repeated-tags-roundtrip", 1) {
    using namespace shammodels::sph;

    constexpr u64 npart = 5;
    constexpr u32 ndust = 3;

    PhantomDump dump{};
    dump.override_magic_number();
    dump.iversion = 1;
    dump.fileid   = sham::format("{:100s}", "FT:Phantom Shamrock test");

    PhantomDumpBlock block{};
    block.tot_count = npart;

    u64 xid = block.get_ref_fort_real("x");
    u64 yid = block.get_ref_fort_real("y");
    u64 zid = block.get_ref_fort_real("z");
    u64 hid = block.get_ref_f32("h");

    std::vector<std::vector<f64>> dustfrac_ref(ndust);
    for (u64 i = 0; i < npart; i++) {
        block.blocks_fort_real[xid].vals.push_back(f64(i));
        block.blocks_fort_real[yid].vals.push_back(f64(i) + 0.25);
        block.blocks_fort_real[zid].vals.push_back(f64(i) + 0.5);
        block.blocks_f32[hid].vals.push_back(f32(i) + 0.125f);
    }
    for (u32 j = 0; j < ndust; j++) {
        u64 id = block.add_new_fort_real("dustfrac");
        for (u64 i = 0; i < npart; i++) {
            dustfrac_ref[j].push_back(0.01 * f64(j + 1) + 0.001 * f64(i));
        }
        block.blocks_fort_real[id].vals = dustfrac_ref[j];
    }

    dump.blocks.push_back(std::move(block));

    // serialise then read back in memory
    shambase::FortranIOFile file_out = dump.gen_file();
    auto &buf                        = file_out.get_internal_buf();
    u64 length                       = buf.tellp();
    shambase::FortranIOFile file_in(std::move(buf), length);

    PhantomDump read = PhantomDump::from_file(file_in);
    REQUIRE(file_in.finished_read());

    REQUIRE_EQUAL(read.blocks.size(), 1_u64);
    PhantomDumpBlock &rblock = read.blocks[0];
    REQUIRE_EQUAL(rblock.tot_count, i64(npart));

    REQUIRE_EQUAL(rblock.count_arrays("dustfrac"), u64(ndust));
    std::vector<std::vector<f64>> dustfrac = rblock.fetch_all_arrays<f64>("dustfrac");
    REQUIRE_EQUAL(dustfrac.size(), u64(ndust));
    for (u32 j = 0; j < ndust; j++) {
        REQUIRE_EQUAL(dustfrac[j], dustfrac_ref[j]);
    }

    std::vector<f64> x;
    rblock.fill_vec("x", x);
    REQUIRE_EQUAL(x, dump.blocks[0].blocks_fort_real[xid].vals);

    std::vector<std::vector<f32>> h = rblock.fetch_all_arrays<f32>("h");
    REQUIRE_EQUAL(h.size(), 1_u64);
    REQUIRE_EQUAL(h[0], dump.blocks[0].blocks_f32[hid].vals);
}
