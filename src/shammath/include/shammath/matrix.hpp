// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file matrix.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/assert.hpp"
#include "shambackends/sycl.hpp"
#include "shammath/matrix_op.hpp"
#include <experimental/mdspan>
#include <array>

namespace shammath {

    template<class T, int m, int n>
    class mat {
        public:
        std::array<T, m * n> data;
        inline constexpr auto get_mdspan() {
            return std::mdspan<T, std::extents<size_t, m, n>>(data.data());
        }

        inline constexpr T &operator()(int i, int j) { return get_mdspan()(i, j); }

        bool operator==(const mat<T, m, n> &other) { return data == other.data; }
    };

    template<class T, int n>
    inline constexpr mat<T, n, n> mat_identity() {
        mat<T, n, n> res{};
        mat_set_identity(res.get_mdspan());
        return res;
    }

    template<class T, int n>
    class vec {
        public:
        std::array<T, n> data;
        inline constexpr auto get_mdspan() {
            return std::mdspan<T, std::extents<size_t, n>>(data.data());
        }

        inline constexpr auto get_mdspan_mat_col() {
            return std::mdspan<T, std::extents<size_t, n, 1>>(data.data());
        }
        inline constexpr auto get_mdspan_mat_row() {
            return std::mdspan<T, std::extents<size_t, 1, n>>(data.data());
        }

        inline constexpr T &operator[](int i) { return get_mdspan()(i); }

        bool operator==(const vec<T, n> &other) { return data == other.data; }
    };

} // namespace shammath
