// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamalgs/memory.hpp"
#include "shamalgs/random.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"
#include <numeric>

struct TestStreamCompact {

    using vFunctionCall = std::tuple<std::optional<sycl::buffer<u32>>, u32> (*)(
        sycl::queue &, sycl::buffer<u32> &, u32);

    vFunctionCall fct;

    explicit TestStreamCompact(vFunctionCall arg) : fct(arg) {};

    void check() {
        std::vector<u32> data{1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1};

        u32 len = data.size();

        auto buf = shamalgs::memory::vec_to_buf(data);

        auto [res, res_len] = fct(shamsys::instance::get_compute_queue(), buf, len);

        auto res_check = shamalgs::memory::buf_to_vec(*res, res_len);

        // make check
        std::vector<u32> idxs;
        {
            for (u32 idx = 0; idx < len; idx++) {
                if (data[idx]) {
                    idxs.push_back(idx);
                }
            }
        }

        REQUIRE_EQUAL_NAMED("same length", res_len, u32(idxs.size()));

        for (u32 idx = 0; idx < res_len; idx++) {
            REQUIRE_EQUAL_NAMED("sid_check", res_check[idx], idxs[idx]);
        }
    }
};
