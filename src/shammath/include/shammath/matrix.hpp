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

#include "shambackends/sycl.hpp"
#include <experimental/mdspan>
#include <array>

// the legendary trick to force a compilation error for missing ;
#define SHAM_ASSERT(x)                                                                             \
    do {                                                                                           \
    } while (false)

namespace shammath {

    template<class T, class Extents, class Layout, class Accessor>
    inline void mat_set_identity(const std::mdspan<T, Extents, Layout, Accessor> &input1) {

        SHAM_ASSERT(input1.extent(0) == input1.extent(1));

        for (int i = 0; i < input1.extent(0); i++) {
            for (int j = 0; j < input1.extent(1); j++) {
                input1(i, j) = (i == j) ? 1 : 0;
            }
        }
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void mat_copy(
        const std::mdspan<T, Extents, Layout, Accessor> &input,
        const std::mdspan<T, Extents, Layout, Accessor> &output) {

        for (int i = 0; i < input.extent(0); i++) {
            for (int j = 0; j < input.extent(1); j++) {
                output(i, j) = input(i, j);
            }
        }
    }

    template<
        class T,
        class Extents1,
        class Extents2,
        class Extents3,
        class Layout1,
        class Layout2,
        class Layout3,
        class Accessor1,
        class Accessor2,
        class Accessor3>
    inline void mat_add(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input1,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &input2,
        const std::mdspan<T, Extents3, Layout3, Accessor3> &output) {

        SHAM_ASSERT(input1.extent(0) == output.extent(0));
        SHAM_ASSERT(input1.extent(1) == output.extent(1));
        SHAM_ASSERT(input1.extent(0) == input2.extent(0));
        SHAM_ASSERT(input1.extent(1) == input2.extent(1));

        for (int i = 0; i < input1.extent(0); i++) {
            for (int j = 0; j < input1.extent(1); j++) {
                output(i, j) = input1(i, j) + input2(i, j);
            }
        }
    }

    template<
        class T,
        class Extents1,
        class Extents2,
        class Extents3,
        class Layout1,
        class Layout2,
        class Layout3,
        class Accessor1,
        class Accessor2,
        class Accessor3>
    inline void mat_sub(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input1,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &input2,
        const std::mdspan<T, Extents3, Layout3, Accessor3> &output) {

        SHAM_ASSERT(input1.extent(0) == output.extent(0));
        SHAM_ASSERT(input1.extent(1) == output.extent(1));
        SHAM_ASSERT(input1.extent(0) == input2.extent(0));
        SHAM_ASSERT(input1.extent(1) == input2.extent(1));

        for (int i = 0; i < input1.extent(0); i++) {
            for (int j = 0; j < input1.extent(1); j++) {
                output(i, j) = input1(i, j) - input2(i, j);
            }
        }
    }

    template<
        class T,
        class Extents1,
        class Extents2,
        class Extents3,
        class Layout1,
        class Layout2,
        class Layout3,
        class Accessor1,
        class Accessor2,
        class Accessor3>
    inline void mat_prod(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input1,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &input2,
        const std::mdspan<T, Extents3, Layout3, Accessor3> &output) {

        SHAM_ASSERT(input1.extent(0) == output.extent(0));
        SHAM_ASSERT(input1.extent(1) == input2.extent(0));
        SHAM_ASSERT(input2.extent(1) == output.extent(1));

        // output_ij = mat1_ik mat2_jk
        for (int i = 0; i < input1.extent(0); i++) {
            for (int j = 0; j < input2.extent(1); j++) {
                T sum = 0;
                for (int k = 0; k < input1.extent(1); k++) {
                    sum += input1(i, k) * input2(k, j);
                }
                output(i, j) = sum;
            }
        }
    }

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

    template<class T, class SizeType, class Layout, class Accessor>
    inline void mat_inv_33(
        const std::mdspan<T, std::extents<SizeType, 3, 3>, Layout, Accessor> &input,
        const std::mdspan<T, std::extents<SizeType, 3, 3>, Layout, Accessor> &output) {

        T &a00 = input(0, 0);
        T &a10 = input(1, 0);
        T &a20 = input(2, 0);

        T &a01 = input(0, 1);
        T &a11 = input(1, 1);
        T &a21 = input(2, 1);

        T &a02 = input(0, 2);
        T &a12 = input(1, 2);
        T &a22 = input(2, 2);

        T det
            = (-a02 * a11 * a20 + a01 * a12 * a20 + a02 * a10 * a21 - a00 * a12 * a21
               - a01 * a10 * a22 + a00 * a11 * a22);

        output(0, 0) = (-a12 * a21 + a11 * a22) / det;
        output(1, 0) = (a12 * a20 - a10 * a22) / det;
        output(2, 0) = (-a11 * a20 + a10 * a21) / det;

        output(0, 1) = (a02 * a21 - a01 * a22) / det;
        output(1, 1) = (-a02 * a20 + a00 * a22) / det;
        output(2, 1) = (a01 * a20 - a00 * a21) / det;

        output(0, 2) = (-a02 * a11 + a01 * a12) / det;
        output(1, 2) = (a02 * a10 - a00 * a12) / det;
        output(2, 2) = (-a01 * a10 + a00 * a11) / det;
    }

    template<class T>
    inline auto
    compute_inv_33(std::array<sycl::vec<T, 3>, 3> mat) -> std::array<sycl::vec<T, 3>, 3> {

        using vec = sycl::vec<T, 3>;

        T a00 = mat[0].x();
        T a10 = mat[1].x();
        T a20 = mat[2].x();

        T a01 = mat[0].y();
        T a11 = mat[1].y();
        T a21 = mat[2].y();

        T a02 = mat[0].z();
        T a12 = mat[1].z();
        T a22 = mat[2].z();

        T det
            = (-a02 * a11 * a20 + a01 * a12 * a20 + a02 * a10 * a21 - a00 * a12 * a21
               - a01 * a10 * a22 + a00 * a11 * a22);

        return {
            (vec{-a12 * a21 + a11 * a22, a02 * a21 - a01 * a22, -a02 * a11 + a01 * a12} / det),
            (vec{a12 * a20 - a10 * a22, -a02 * a20 + a00 * a22, a02 * a10 - a00 * a12} / det),
            (vec{-a11 * a20 + a10 * a21, a01 * a20 - a00 * a21, -a01 * a10 + a00 * a11} / det)};
    }

    template<class T>
    inline auto
    mat_prod_33(std::array<sycl::vec<T, 3>, 3> mat_a, std::array<sycl::vec<T, 3>, 3> mat_b)
        -> std::array<sycl::vec<T, 3>, 3> {

        using vec = sycl::vec<T, 3>;

        T a00 = mat_a[0].x();
        T a10 = mat_a[1].x();
        T a20 = mat_a[2].x();

        T a01 = mat_a[0].y();
        T a11 = mat_a[1].y();
        T a21 = mat_a[2].y();

        T a02 = mat_a[0].z();
        T a12 = mat_a[1].z();
        T a22 = mat_a[2].z();

        T b00 = mat_b[0].x();
        T b10 = mat_b[1].x();
        T b20 = mat_b[2].x();

        T b01 = mat_b[0].y();
        T b11 = mat_b[1].y();
        T b21 = mat_b[2].y();

        T b02 = mat_b[0].z();
        T b12 = mat_b[1].z();
        T b22 = mat_b[2].z();

        return {
            vec{a00 * b00 + a01 * b10 + a02 * b20,
                a00 * b01 + a01 * b11 + a02 * b21,
                a00 * b02 + a01 * b12 + a02 * b22},
            vec{a10 * b00 + a11 * b10 + a12 * b20,
                a10 * b01 + a11 * b11 + a12 * b21,
                a10 * b02 + a11 * b12 + a12 * b22},
            vec{a20 * b00 + a21 * b10 + a22 * b20,
                a20 * b01 + a21 * b11 + a22 * b21,
                a20 * b02 + a21 * b12 + a22 * b22}};
    }

} // namespace shammath
