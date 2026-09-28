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
 * @file TestAssertList.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/aliases_float.hpp"
#include "shambase/string.hpp"
#include "TestAssert.hpp"
#include <utility>

namespace shamtest::details {

    /**
     *@brief Format a string that is an assert name
     *
     * If the string is empty, returns an empty string
     * Otherwise, returns the string with " | " appended
     */
    inline std::string format_assert_name(std::string s) {
        if (s == "") {
            return "";
        }
        return "\"" + s + "\" : ";
    }

    /// Class to hold the list of assertion related to a test
    struct TestAssertList {

        /// List of assertion held by the class
        std::vector<TestAssert> asserts;

        // define member function here
        // to register asserts

        /// Register an assertion with the supplied log
        void assert_bool_with_log(std::string assert_name, bool v, std::string log);

        // The register_require* functions implement the REQUIRE* macros of shamtest.hpp, they are
        // kept out of the macros so that each assert expands to a single call.

        /// Register the result of a `REQUIRE_NAMED` assert
        void register_require(std::string name, const char *expr, bool eval, SourceLocation loc);

        /// Register the result of a `REQUIRE_EQUAL_CUSTOM_COMP_NAMED` assert
        template<class Ta, class Tb>
        inline void register_require_equal(
            std::string name,
            const char *assert_expr,
            bool eval,
            const char *expr_a,
            Ta &a,
            const char *expr_b,
            Tb &b,
            SourceLocation loc) {
            std::string assert_name = format_assert_name(std::move(name)) + assert_expr;
            if (eval) {
                assert_bool_with_log(assert_name, eval, "");
            } else {
                assert_bool_with_log(
                    assert_name,
                    eval,
                    assert_name + " evaluated to false\n\n" + " -> " + expr_a
                        + sham::format(" = {}", a) + "\n" + " -> " + expr_b
                        + sham::format(" = {}", b) + "\n"
                        + " -> location : " + loc.format_one_line());
            }
        }

        /// Register the result of a `REQUIRE_FLOAT_EQUAL_CUSTOM_DIST_NAMED` assert
        template<class Ta, class Tb, class Tprec>
        inline void register_require_float_equal(
            std::string name,
            const char *assert_expr,
            bool eval,
            const char *expr_a,
            Ta &a,
            const char *expr_b,
            Tb &b,
            const char *expr_prec,
            const Tprec &prec,
            SourceLocation loc) {
            std::string assert_name = format_assert_name(std::move(name)) + assert_expr;
            if (eval) {
                assert_bool_with_log(assert_name, eval, "");
            } else {
                assert_bool_with_log(
                    assert_name,
                    eval,
                    assert_name + " evaluated to false\n\n" + " -> " + expr_a
                        + sham::format(" = {}", a) + "\n" + " -> " + expr_b
                        + sham::format(" = {}", b) + "\n" + " -> " + expr_prec
                        + sham::format(" = {}", prec) + "\n"
                        + " -> location : " + loc.format_one_line());
            }
        }

        /// Append the source location to the the supplied string to generate a comment
        inline static std::string gen_comment(std::string s, SourceLocation loc) {
            return s + "\n" + loc.format_multiline();
        }

        /// Test if the supplied boolean is true
        [[deprecated("Please use the supplied testing macros instead")]]
        inline void assert_bool(
            std::string assert_name, bool v, SourceLocation loc = SourceLocation{}) {

            asserts.push_back(
                TestAssert{
                    .value   = v,
                    .name    = std::move(assert_name),
                    .comment = "failed assert location : " + loc.format_one_line()});
        }

        /// Test for an equality
        template<class T1, class T2>
        [[deprecated("Please use the supplied testing macros instead")]]
        inline void assert_equal(
            std::string assert_name, T1 a, T2 b, SourceLocation loc = SourceLocation{}) {

            bool t              = a == b;
            std::string comment = "";

            if (!t) {
                comment = "left=" + std::to_string(a) + " right=" + std::to_string(b);
            }

            asserts.push_back(
                TestAssert{
                    .value   = t,
                    .name    = std::move(assert_name),
                    .comment = gen_comment(comment, loc)});
        }

        /// Assert equal on an array of values
        template<class Acca, class Accb>
        [[deprecated("Please use the supplied testing macros instead")]]
        inline void assert_equal_array(
            std::string assert_name,
            Acca &acc_a,
            Accb &acc_b,
            u32 len,
            SourceLocation loc = SourceLocation{}) {

            bool t              = true;
            std::string comment = "";

            for (u32 i = 0; i < len; i++) {
                t = t && (acc_a[i] == acc_b[i]);
            }

            if (!t) {
                comment += "left : \n";
                comment += shambase::format_array(acc_a, len, 16, "{} ");
                comment += "right : \n";
                comment += shambase::format_array(acc_b, len, 16, "{} ");
            }

            asserts.push_back(
                TestAssert{
                    .value   = t,
                    .name    = std::move(assert_name),
                    .comment = gen_comment(comment, loc)});
        }

        /**
         * @brief Add an assertion testing a floating point equality up to precision eps
         *
         * @param assert_name name of the assertion
         * @param a value a
         * @param b value b
         * @param eps precision of the test
         * @param loc source location of the call
         */
        [[deprecated("Please use the supplied testing macros instead")]]
        inline void assert_float_equal(
            std::string assert_name, f64 a, f64 b, f64 eps, SourceLocation loc = SourceLocation{}) {
            f64 diff = std::fabs(a - b);

            bool t              = diff < eps;
            std::string comment = "";

            if (!t) {
                comment = "left=" + std::to_string(a) + " right=" + std::to_string(b)
                          + " diff=" + std::to_string(diff);
            }

            asserts.push_back(
                TestAssert{
                    .value   = t,
                    .name    = std::move(assert_name),
                    .comment = gen_comment(comment, loc)});
        }

        /// add an assertion with a comment
        inline void assert_add_comment(
            std::string assert_name,
            bool v,
            std::string comment,
            SourceLocation loc = SourceLocation{}) {
            asserts.push_back(
                TestAssert{
                    .value   = v,
                    .name    = std::move(assert_name),
                    .comment = gen_comment(std::move(comment), loc)});
        }

        /// Serialize the assertion in JSON
        std::string serialize_json();
        /// Serialize the assertion in binary format
        void serialize(std::basic_stringstream<byte> &stream);
        /// DeSerialize the assertion from binary format
        static TestAssertList deserialize(std::basic_stringstream<byte> &reader);

        /// Get number of assertion in the list
        inline u32 get_assert_count() { return asserts.size(); }

        /// Get the number of successful assertions
        inline u32 get_assert_success_count() {
            u32 cnt = 0;
            for (TestAssert &a : asserts) {
                if (a.value) {
                    cnt++;
                }
            }
            return cnt;
        }
    };
} // namespace shamtest::details
