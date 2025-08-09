// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file DistributedData.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/exception.hpp"
#include "shambase/sets.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include <functional>
#include <map>
#include <utility>

namespace shambase {

    /**
     * @brief Describe an object common to two patches, typically interface (sender,receiver)
     *
     * @tparam T
     */
    template<class T>
    class DistributedDataShared {

        std::multimap<std::pair<u64, u64>, T> data;

        using iterator = typename std::multimap<std::pair<u64, u64>, T>::iterator;

        public:
        inline std::multimap<std::pair<u64, u64>, T> &get_native() { return data; }

        inline iterator add_obj(u64 left_id, u64 right_id, T &&obj) {
            std::pair<u64, u64> tmp = {left_id, right_id};
            return data.emplace(std::move(tmp), std::forward<T>(obj));
        }

        inline void for_each(std::function<void(u64, u64, T &)> &&f) {
            for (auto &[id, obj] : data) {
                f(id.first, id.second, obj);
            }
        }

        inline void tranfer_all(std::function<bool(u64, u64)> cd, DistributedDataShared &other) {

            std::vector<std::pair<u64, u64>> occurences;

            // whoa i forgot the & here and triggered the copy constructor of every patch
            // like do not forget it or it will be a disaster waiting to come
            // i did throw up a 64 GPUs run because of that
            for (auto &[k, v] : data) {
                if (cd(k.first, k.second)) {
                    occurences.push_back(k);
                }
            }

            for (auto p : occurences) {
                auto ext = data.extract(p);
                other.data.insert(std::move(ext));
            }
        }

        inline bool has_key(u64 left_id, u64 right_id) {
            return (data.find({left_id, right_id}) != data.end());
        }

        inline u64 get_element_count() { return data.size(); }

        template<class Tmap>
        inline DistributedDataShared<Tmap> map(std::function<Tmap(u64, u64, T &)> map_func) {
            DistributedDataShared<Tmap> ret;
            for_each([&](u64 left, u64 right, T &ref) {
                ret.add_obj(left, right, map_func(left, right, ref));
            });
            return ret;
        }

        inline void reset() { data.clear(); }

        inline bool is_empty() { return data.empty(); }
    };

} // namespace shambase
