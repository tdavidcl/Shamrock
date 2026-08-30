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
 * @file modern_gpu_merge_sort.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace shamalgs::primitives::device::details {

    namespace debug {

        inline constexpr const char *ansi_green = "\033[32m";
        inline constexpr const char *ansi_red   = "\033[31m";
        inline constexpr const char *ansi_reset = "\033[0m";

        /// Flags indices that continue an ascending run from the previous element (mirrors
        /// `_sorted_run_mask` in examples/benchmarks/run_sort_by_keys_tmp_dev.py).
        template<class T>
        inline std::vector<bool> sorted_run_mask(const std::vector<T> &values) {
            std::vector<bool> mask(values.size(), false);
            for (size_t i = 0; i < values.size(); ++i) {
                mask[i] = (i == 0) || (values[i - 1] < values[i]);
            }
            return mask;
        }

        template<class T>
        inline std::string to_str(const T &v) {
            std::ostringstream oss;
            oss << v;
            return oss.str();
        }

    } // namespace debug

    /**
     * @brief Debug pretty-printer for key/value arrays, mirroring `print_key_val_table` in
     * examples/benchmarks/run_sort_by_keys_tmp_dev.py.
     *
     * Prints the indices, the keys (highlighted green if ascending from the previous key, red
     * otherwise), and the values as an aligned table.
     *
     * @tparam Tkey Type of the keys.
     * @tparam Tval Type of the values.
     * @param keys Host-side keys to print.
     * @param vals Host-side values to print.
     * @param idx_label Label of the index row.
     * @param key_label Label of the key row.
     * @param val_label Label of the value row.
     */
    template<class Tkey, class Tval>
    inline void print_key_val_table(
        const std::vector<Tkey> &keys,
        const std::vector<Tval> &vals,
        const std::string &idx_label = "i",
        const std::string &key_label = "key",
        const std::string &val_label = "val") {

        using namespace debug;

        size_t n = keys.size();

        std::vector<std::string> idx_str(n), key_str(n), val_str(n);
        size_t col_width = 0;
        for (size_t i = 0; i < n; ++i) {
            idx_str[i] = to_str(i);
            key_str[i] = to_str(keys[i]);
            val_str[i] = to_str(vals[i]);
            col_width
                = std::max({col_width, idx_str[i].size(), key_str[i].size(), val_str[i].size()});
        }

        size_t label_width = std::max({idx_label.size(), key_label.size(), val_label.size()});

        auto pad_label = [&](const std::string &label) {
            std::string s = label;
            s.resize(label_width, ' ');
            return s;
        };

        auto pad_cell = [&](const std::string &s) {
            return std::string(col_width - s.size(), ' ') + s;
        };

        auto fmt_row = [&](const std::string &label,
                           const std::vector<std::string> &values,
                           const std::vector<bool> *mask) {
            std::string line = pad_label(label) + " | ";
            for (size_t i = 0; i < values.size(); ++i) {
                if (i > 0) {
                    line += "  ";
                }
                std::string cell = pad_cell(values[i]);
                if (mask != nullptr) {
                    cell = std::string((*mask)[i] ? ansi_green : ansi_red) + cell + ansi_reset;
                }
                line += cell;
            }
            return line;
        };

        std::vector<bool> mask = sorted_run_mask(keys);

        std::string idx_line = fmt_row(idx_label, idx_str, nullptr);
        std::string key_line = fmt_row(key_label, key_str, &mask);
        std::string val_line = fmt_row(val_label, val_str, nullptr);

        std::cout << std::string(idx_line.size(), '-') << "\n";
        std::cout << idx_line << "\n";
        std::cout << std::string(idx_line.size(), '-') << "\n";
        std::cout << key_line << "\n";
        std::cout << val_line << "\n";
    }

    template<class Tkey, class Tval>
    inline void sort_by_keys_modern_gpu_mergesort(
        sham::DeviceBuffer<Tkey> &buf_key, sham::DeviceBuffer<Tval> &buf_values, u32 len) {
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << "------- sort_by_keys_modern_gpu_mergesort -------" << std::endl;
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << "init state:" << std::endl;
        print_key_val_table(buf_key.copy_to_stdvec(), buf_values.copy_to_stdvec());
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << "------- sort_by_keys_modern_gpu_mergesort end -------" << std::endl;
        std::cout << "-------------------------------------------------" << std::endl;
    }

} // namespace shamalgs::primitives::device::details
