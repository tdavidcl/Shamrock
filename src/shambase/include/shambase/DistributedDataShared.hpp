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
 * @file DistributedDataShared.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Container for objects shared between two distributed data patches
 *
 */

#include "shambase/aliases_int.hpp"
#include <functional>
#include <map>
#include <utility>
#include <vector>

namespace shambase {

    /**
     * @brief Container for objects shared between two distributed data elements
     *
     * This class manages objects that are associated with pairs of patch identifiers,
     * typically representing interfaces or shared boundaries between computational domains.
     * It uses a multimap to allow multiple objects for the same patch pair
     * (typically for ghost zones with periodic boundaries).
     *
     * @tparam T Type of objects to be stored in the container
     *
     * @code{.cpp}
     * // Create a container for shared boundary data
     * DistributedDataShared<BoundaryInfo> boundaries;
     *
     * // Add objects for patch pairs
     * boundaries.add_obj(1, 2, boundary_data);
     * boundaries.add_obj(2, 3, other_boundary);
     *
     * // Iterate over all shared objects
     * boundaries.for_each([](u64 left, u64 right, BoundaryInfo& data) {
     *     // Process boundary data between patches left and right
     * });
     * @endcode
     */
    template<class T>
    class DistributedDataShared {

        std::multimap<std::pair<u64, u64>, T> data;

        using iterator = typename std::multimap<std::pair<u64, u64>, T>::iterator;

        public:
        /**
         * @brief Get direct access to the underlying multimap container
         *
         * @return Reference to the internal multimap storing patch pair to object mappings
         */
        inline std::multimap<std::pair<u64, u64>, T> &get_native() { return data; }

        /// const version
        inline const std::multimap<std::pair<u64, u64>, T> &get_native() const { return data; }

        /// iterator forwarding
        inline auto begin() { return data.begin(); }
        /// iterator forwarding
        inline auto end() { return data.end(); }
        /// iterator forwarding
        inline auto begin() const { return data.begin(); }
        /// iterator forwarding
        inline auto end() const { return data.end(); }
        /// iterator forwarding
        inline auto cbegin() const { return data.cbegin(); }
        /// iterator forwarding
        inline auto cend() const { return data.cend(); }

        /**
         * @brief Add an object associated with a patch pair
         *
         * @param left_id Identifier of the first patch
         * @param right_id Identifier of the second patch
         * @param obj Object to be associated with the patch pair (moved)
         * @return Iterator pointing to the inserted element
         *
         * @code{.cpp}
         * DistributedDataShared<DataType> container;
         * auto it = container.add_obj(1, 2, std::move(data_object));
         * @endcode
         */
        inline iterator add_obj(u64 left_id, u64 right_id, T &&obj) {
            return data.emplace(std::pair<u64, u64>{left_id, right_id}, std::forward<T>(obj));
        }

        /**
         * @brief Apply a function to all stored objects
         *
         * @param f Function to apply, receives (left_id, right_id, object_reference)
         *
         * @code{.cpp}
         * container.for_each([](u64 left, u64 right, DataType& obj) {
         *     // Process each object with its associated patch pair
         * });
         * @endcode
         */
        inline void for_each(std::function<void(u64, u64, T &)> &&f) {
            for (auto &[id, obj] : data) {
                f(id.first, id.second, obj);
            }
        }

        /// const version
        inline void for_each(std::function<void(u64, u64, const T &)> &&f) const {
            for (auto &[id, obj] : data) {
                f(id.first, id.second, obj);
            }
        }

        /**
         * @brief Transfer objects to another container based on a condition
         *
         * Moves objects from this container to another container if they satisfy
         * the provided condition function. The objects are removed from this container.
         *
         * @param cd Condition function that takes (left_id, right_id) and returns true for objects
         * to transfer
         * @param other Destination container to receive the transferred objects
         *
         * @code{.cpp}
         * DistributedDataShared<DataType> source, destination;
         * // Transfer objects where left_id > right_id
         * source.tranfer_all([](u64 left, u64 right) { return left > right; }, destination);
         * @endcode
         */
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

        /**
         * @brief Check if a patch pair exists in the container
         *
         * @param left_id Identifier of the first patch
         * @param right_id Identifier of the second patch
         * @return True if at least one object exists for the given patch pair
         *
         * @code{.cpp}
         * if (container.has_key(1, 2)) {
         *     // Handle existing patch pair
         * }
         * @endcode
         */
        inline bool has_key(u64 left_id, u64 right_id) const {
            return (data.find({left_id, right_id}) != data.end());
        }

        /**
         * @brief Get the total number of objects in the container
         *
         * @return Number of stored objects across all patch pairs
         */
        inline u64 get_element_count() const { return data.size(); }

        /**
         * @brief Transform all objects to a new type using a mapping function
         *
         * @tparam Tmap Type of the transformed objects
         * @param map_func Transformation function that takes (left_id, right_id, object_reference)
         * and returns transformed object
         * @return New container with transformed objects maintaining the same patch pair
         * associations
         *
         * @code{.cpp}
         * DistributedDataShared<int> int_container;
         * auto string_container = int_container.map<std::string>(
         *     [](u64 left, u64 right, int& value) {
         *         return std::to_string(value);
         *     }
         * );
         * @endcode
         */
        template<class Tmap>
        inline DistributedDataShared<Tmap> map(std::function<Tmap(u64, u64, T &)> map_func) {
            DistributedDataShared<Tmap> ret;
            for_each([&](u64 left, u64 right, T &ref) {
                ret.add_obj(left, right, map_func(left, right, ref));
            });
            return ret;
        }

        /// const version
        template<class Tmap>
        inline DistributedDataShared<Tmap> map(
            std::function<Tmap(u64, u64, const T &)> map_func) const {
            DistributedDataShared<Tmap> ret;
            for_each([&](u64 left, u64 right, const T &ref) {
                ret.add_obj(left, right, map_func(left, right, ref));
            });
            return ret;
        }

        /**
         * @brief Reshape the container to hold exactly the entries of `reference`, reusing the
         * objects already stored instead of creating new ones.
         *
         * This is the DistributedDataShared equivalent of the `ensure_sizes` pattern: a container
         * kept alive between steps keeps its objects (and the allocations they own), only the
         * entries that cannot be reused are created.
         *
         * After the call, the container holds one object per entry of `reference`, with the same
         * keys, the same multiplicity and the same iteration order. The objects are taken, in
         * priority order, from:
         *   1. the object stored under the same key at the same occurrence rank (the k-th object
         *      stored for a key is reused for the k-th entry of that key in `reference`),
         *   2. the objects left over by step 1 (keys absent from `reference`, or surplus
         *      occurrences of a key), recycled in their iteration order,
         *   3. `create()` once no leftover object remains.
         * Leftover objects that were not recycled are destroyed.
         *
         * `update(left_id, right_id, obj, ref_obj)` is then called on every entry, in the
         * iteration order of `reference`. Reused objects keep their previous content, so `update`
         * must set the full state it relies on (e.g. resize and overwrite a buffer).
         *
         * Every step is driven by the iteration order of the two containers, so the result is
         * deterministic.
         *
         * @tparam Tref type of the objects stored in the reference container
         * @param reference container whose entries are reproduced
         * @param create function `() -> T` called when no stored object can be reused
         * @param update function `(u64 left_id, u64 right_id, T &obj, const Tref &ref_obj)`
         *
         * @code{.cpp}
         * DistributedDataShared<std::vector<int>> buffers;
         * DistributedDataShared<u32> sizes;
         * buffers.reshape_like(
         *     sizes,
         *     []() { return std::vector<int>{}; },
         *     [](u64, u64, std::vector<int> &buf, const u32 &size) { buf.resize(size); });
         * @endcode
         */
        template<class Tref, class FuncCreate, class FuncUpdate>
        inline void reshape_like(
            const DistributedDataShared<Tref> &reference,
            FuncCreate &&create,
            FuncUpdate &&update) {

            using map_t  = std::multimap<std::pair<u64, u64>, T>;
            using node_t = typename map_t::node_type;

            // one node per entry of reference, in the iteration order of reference
            std::vector<node_t> slots;
            slots.reserve(reference.get_element_count());

            // indices in slots that are not matched by key
            std::vector<size_t> unmatched;

            // 1. reuse the object with the same key and the same occurrence rank
            // (lower_bound gives the first remaining occurrence of the key in insertion order)
            for (const auto &entry : reference) {
                auto it = data.lower_bound(entry.first);
                if (it != data.end() && it->first == entry.first) {
                    slots.push_back(data.extract(it));
                } else {
                    unmatched.push_back(slots.size());
                    slots.emplace_back();
                }
            }

            // 2. recycle the leftovers, 3. then create the missing objects
            for (size_t i : unmatched) {
                if (!data.empty()) {
                    slots[i] = data.extract(data.begin());
                } else {
                    map_t tmp;
                    tmp.emplace(std::pair<u64, u64>{0, 0}, create());
                    slots[i] = tmp.extract(tmp.begin());
                }
            }

            // drop the leftovers that were not recycled
            data.clear();

            // insert in the iteration order of reference, inserting at the end keeps the
            // relative order of equal keys
            size_t i = 0;
            for (const auto &entry : reference) {
                node_t &node = slots[i++];
                node.key()   = entry.first;
                auto it      = data.insert(data.end(), std::move(node));
                update(entry.first.first, entry.first.second, it->second, entry.second);
            }
        }

        /**
         * @brief Clear all objects from the container
         */
        inline void reset() { data.clear(); }

        /**
         * @brief Check if the container is empty
         *
         * @return True if no objects are stored in the container
         */
        inline bool is_empty() const { return data.empty(); }
    };

} // namespace shambase
