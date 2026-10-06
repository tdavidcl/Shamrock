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
 * @file mdspan_func_accessor.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Read-only `std::mdspan` whose elements are computed by a callable on access
 *
 * Useful to hand a value that is cheap to recompute to a function taking an mdspan, instead of
 * storing it in memory first.
 */

#include <experimental/mdspan>
#include <type_traits>
#include <cstddef>

namespace shambase {

    /**
     * @brief mdspan accessor policy evaluating `func(offset + i)` for the element of linear
     * index `i`
     *
     * @tparam Func Callable invoked with a `std::size_t` returning the element by value
     */
    template<class Func>
    struct FuncAccessor {

        /// The data handle of the mdspan: the callable and an offset (for submdspan)
        struct Handle {
            Func func;
            std::size_t offset;
        };

        using reference        = std::invoke_result_t<const Func &, std::size_t>;
        using element_type     = const std::remove_cvref_t<reference>;
        using data_handle_type = Handle;
        using offset_policy    = FuncAccessor;

        constexpr reference access(const data_handle_type &h, std::size_t i) const {
            return h.func(h.offset + i);
        }

        constexpr data_handle_type offset(const data_handle_type &h, std::size_t i) const {
            return {h.func, h.offset + i};
        }
    };

    /**
     * @brief Rank-1 read-only mdspan of extent @p n whose element `i` is `func(i)`
     *
     * @code{.cpp}
     * auto squares = shambase::make_func_mdspan_rank_1(u32(10), [](std::size_t i) {
     *     return double(i * i);
     * });
     * double v = squares[3]; // 9
     * @endcode
     */
    template<class IndexType, class Func>
    inline auto make_func_mdspan_rank_1(IndexType n, Func func) {
        using Acc = FuncAccessor<Func>;
        using Ext = std::dextents<IndexType, 1>;
        return std::mdspan<typename Acc::element_type, Ext, std::layout_right, Acc>(
            typename Acc::data_handle_type{func, 0},
            std::layout_right::mapping<Ext>(Ext(n)),
            Acc{});
    }

} // namespace shambase
