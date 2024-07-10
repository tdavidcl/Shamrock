// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file type_traits.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Traits for C++ types
 * 
 * This file contains utility traits classes for C++ types.
 * 
 */

 
#include "shambase/aliases_int.hpp"
#include <climits>
#include <type_traits>

namespace shambase {

    /**
     * @brief Number of bits in a type T
     *
     * @tparam T type for which the number of bits is computed
     */
    template<class T>
    inline constexpr u64 bitsizeof = sizeof(T) * CHAR_BIT;

    /**
     * @brief Check if a type has a certain number of bits
     *
     * This struct template provides a static constexpr bool member `value` which
     * is true if the type `T` has `num` bits.
     *
     * @tparam T type to check
     * @tparam num number of bits
     */
    template<typename T, int num>
    struct has_bitlen {
        static constexpr bool value = bitsizeof<T> == num;
    };

    /**
     * @brief Helper variable template for has_bitlen
     *
     * This variable template is true if `T` has `num` bits.
     *
     * @tparam T type to check
     * @tparam num number of bits
     */
    template<typename T, int num>
    inline constexpr bool has_bitlen_v = has_bitlen<T, num>::value;


    template<typename T>
    class has_operator_self_geq {
        using found     = std::true_type;
        using not_found = std::false_type;

        template<typename A>
        static auto test(A tested) -> decltype(tested >= tested, found());
        template<typename A>
        static not_found test(...);

        public:
        static constexpr bool value = std::is_same<decltype(test<T>(0)), found>::value;
    };

    template<typename T>
    class has_operator_self_leq {
        using found     = std::true_type;
        using not_found = std::false_type;

        template<typename A>
        static auto test(A tested) -> decltype(tested <= tested, found());
        template<typename A>
        static not_found test(...);

        public:
        static constexpr bool value = std::is_same<decltype(test<T>(0)), found>::value;
    };

    template<typename T>
    class has_operator_self_greater_than {
        using found     = std::true_type;
        using not_found = std::false_type;

        template<typename A>
        static auto test(A tested) -> decltype(tested > tested, found());
        template<typename A>
        static not_found test(...);

        public:
        static constexpr bool value = std::is_same<decltype(test<T>(0)), found>::value;
    };

    template<typename T>
    class has_operator_self_less_than {
        using found     = std::true_type;
        using not_found = std::false_type;

        template<typename A>
        static auto test(A tested) -> decltype(tested < tested, found());
        template<typename A>
        static not_found test(...);

        public:
        static constexpr bool value = std::is_same<decltype(test<T>(0)), found>::value;
    };


    template<class>
    inline constexpr bool always_false_v = false;

} // namespace shambase