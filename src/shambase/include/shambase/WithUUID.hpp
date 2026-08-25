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
 * @file WithUUID.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include <atomic>
#include <limits>

namespace shambase {

    /**
     * @brief A class that provides unique identifiers (UUID) to instances
     *
     * @tparam T The class type to which the UUID will be assigned
     * @tparam Tint The data type for the UUID
     * @tparam thread_safe Whether the UUID constructor should be thread-safe
     *
     * Example usage:
     * @code {.cpp}
     *    class A1 : public WithUUID<A1, u64> {};
     *    ...
     *    std::cout << "Instance1 UUID: " << A1{}.get_uuid() << std::endl;
     * @endcode
     *
     * Copying would duplicate a supposedly-unique uuid across two live instances, so the copy
     * constructor/assignment are deleted. Moving transfers the uuid to the destination and
     * invalidates the source (setting its uuid to `invalid_uuid`), so a moved-from instance
     * never appears alive with a duplicate of the uuid it no longer owns -- check `is_alive()`
     * rather than assuming a moved-from instance still holds a meaningful uuid.
     */
    template<typename T, class Tint, bool thread_safe = true>
    class WithUUID {

        protected:
        /**
         * @brief The unique identifier of the class
         */
        Tint uuid;

        public:
        /// Sentinel uuid value marking an instance as moved-from / invalidated. Chosen as the
        /// max representable Tint rather than e.g. 0, since 0 is a legitimately-assigned uuid.
        static constexpr Tint invalid_uuid = std::numeric_limits<Tint>::max();

        /**
         * @brief Get the uuid of the class
         *
         * @return The uuid of the class
         */
        inline Tint get_uuid() const { return uuid; }

        /// Whether this instance still holds a valid uuid (false once moved from).
        inline bool is_alive() const { return uuid != invalid_uuid; }

        /// Marks this instance's uuid as invalid. Meant for derived classes that need to give
        /// up their identity outside of a move (e.g. after already having fired a one-time
        /// notification tied to it).
        inline void invalidate() { uuid = invalid_uuid; }

        /**
         * @brief Constructor of the class
         *
         * Assigns a unique identifier to the class
         */
        inline WithUUID() {
            if constexpr (thread_safe) {
                // local atomic static storage for the UUID
                static std::atomic<Tint> _uuid = 0;
                // increment and store the UUID (atomic)
                uuid = _uuid.fetch_add(1, std::memory_order_relaxed);
            } else {
                // we need to redo the static storage in this case otherwise
                // some lock xadd would be emitted as std::atomic is thread safe.
                static Tint _uuid = 0;
                uuid              = _uuid++;
            }
        }

        WithUUID(const WithUUID &)            = delete; ///< would duplicate the uuid
        WithUUID &operator=(const WithUUID &) = delete; ///< would duplicate the uuid

        /// Move constructor: transfers the uuid to `this` and invalidates `other`.
        inline WithUUID(WithUUID &&other) noexcept : uuid(other.uuid) { other.invalidate(); }

        /// Move assignment: transfers the uuid to `this` and invalidates `other`.
        inline WithUUID &operator=(WithUUID &&other) noexcept {
            if (this != &other) {
                uuid = other.uuid;
                other.invalidate();
            }
            return *this;
        }
    };

} // namespace shambase
