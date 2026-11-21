// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file type_convert_json.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief json (de)serialization of SYCL vector types
 *
 * Kept separate from type_convert.hpp so that including the conversion helpers does not pull
 * <nlohmann/json.hpp> in.
 */

#include "shambackends/type_convert.hpp"
#include <nlohmann/json.hpp>

NLOHMANN_JSON_NAMESPACE_BEGIN
/// json (de)serialization of sycl::vec through std::array
template<typename T, int n>
struct adl_serializer<sycl::vec<T, n>> {
    /// serialize a sycl::vec as a json array
    static void to_json(json &j, const sycl::vec<T, n> &p) { j = sham::sycl_vec_to_array(p); }

    /// deserialize a sycl::vec from a json array
    static void from_json(const json &j, sycl::vec<T, n> &p) {
        p = sham::array_to_sycl_vec(j.get<std::array<T, n>>());
    }
};
NLOHMANN_JSON_NAMESPACE_END
