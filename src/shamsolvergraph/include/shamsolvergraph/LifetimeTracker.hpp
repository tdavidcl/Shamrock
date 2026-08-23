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
 * @file LifetimeTracker.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Callback API to track the lifetime, state updates and operations of solvergraph objects
 *
 */

#include "shambase/WithUUID.hpp"
#include "shambase/aliases_int.hpp"

namespace shamrock::solvergraph {

    /**
     * @brief Tracks the lifetime of an object of type T and notifies observers through static
     * callbacks.
     *
     * The tracker is meant to be held by the tracked object as a `std::shared_ptr` member (not
     * inherited). This makes lifetime notifications move-safe: when the tracked object is moved,
     * the tracker pointer moves with it and the moved-from object is left with a null tracker,
     * so the destroy notification is emitted exactly once (a plain UUID member would emit a
     * duplicate destroy notification when the moved-from object is destroyed).
     *
     * All callbacks are `nullptr` by default, so the cost when tracking is disabled is a single
     * null pointer check per notification site.
     *
     * @tparam T The tracked object type (e.g. INode, IEdge)
     */
    template<typename T>
    class LifetimeTracker : public shambase::WithUUID<LifetimeTracker<T>, u64> {
        public:
        /// Called when a tracked object is created
        inline static void (*on_create)(u64 uuid) = nullptr;
        /// Called when a tracked object is destroyed
        inline static void (*on_destroy)(u64 uuid) = nullptr;

        /// Called when the state of a tracked object changes (e.g. edges are rebound)
        inline static void (*on_state_update)(T &object) = nullptr;
        /// Called when an operation is performed on a tracked object (e.g. evaluation)
        inline static void (*on_op)(u64 uuid, u64 op_id) = nullptr;

        /// Constructor, notifies the creation of the tracked object
        LifetimeTracker() : shambase::WithUUID<LifetimeTracker, u64>() {
            if (on_create != nullptr) {
                on_create(this->get_uuid());
            }
        };

        LifetimeTracker(const LifetimeTracker &)            = delete; ///< would duplicate the UUID
        LifetimeTracker &operator=(const LifetimeTracker &) = delete; ///< would duplicate the UUID

        /// Move constructor
        LifetimeTracker(LifetimeTracker &&) noexcept = default;
        /// Move assignment
        LifetimeTracker &operator=(LifetimeTracker &&) noexcept = default;

        /// Notify a state update of the tracked object
        inline void notify_update(T &object) {
            if (on_state_update != nullptr) {
                on_state_update(object);
            }
        }

        /// Notify an operation performed on the tracked object
        inline void notify_op(u64 op_id) {
            if (on_op != nullptr) {
                on_op(this->get_uuid(), op_id);
            }
        }

        /// Destructor, notifies the destruction of the tracked object
        ~LifetimeTracker() {
            if (on_destroy != nullptr) {
                on_destroy(this->get_uuid());
            }
        };
    };

} // namespace shamrock::solvergraph
