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
#include <utility>

namespace shamrock::solvergraph {

    /**
     * @brief Tracks the lifetime of an object of type T and notifies observers through static
     * callbacks.
     *
     * The tracker is meant to be held by the tracked object as a plain value member (not
     * inherited, so that trace_state_update()/trace_op() can still take a `T&` reference to the
     * enclosing object rather than to the tracker itself). Holding it by value instead of behind
     * a `std::shared_ptr` avoids a heap allocation per tracked object.
     *
     * Move-safety comes entirely from the base class: WithUUID's move constructor/assignment
     * invalidate the source, so its is_alive() reports false afterward. Every notification
     * method here (including the destructor) checks is_alive() first, so a moved-from instance's
     * own eventual destruction is a silent no-op instead of emitting a duplicate destroy
     * notification for a uuid whose ownership has already moved elsewhere.
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

        private:
        /// Whether trace_state_update() has ever fired for this object. Lets evaluate() (see
        /// INode) lazily fire a first state_update of its own right before the object's first
        /// operation, so meta nodes that own no ro/rw edges (and therefore never go through
        /// __internal_set_ro_edges/__internal_set_rw_edges) still get one without any manual
        /// call in their constructor.
        bool updated = false;

        public:
        /// Whether trace_state_update() has ever fired for this object
        inline bool has_been_updated() const { return updated; }

        /// Constructor, notifies the creation of the tracked object
        LifetimeTracker() : shambase::WithUUID<LifetimeTracker, u64>() {
            if (on_create != nullptr) {
                on_create(this->get_uuid());
            }
        };

        LifetimeTracker(const LifetimeTracker &)            = delete; ///< would duplicate the UUID
        LifetimeTracker &operator=(const LifetimeTracker &) = delete; ///< would duplicate the UUID

        /// Move constructor: delegates entirely to WithUUID's own move constructor, which
        /// transfers the uuid to `this` and invalidates `other`.
        LifetimeTracker(LifetimeTracker &&) noexcept = default;

        /// Move assignment: `this` gives up whatever identity it held (firing its own destroy
        /// notification first, if still alive) before WithUUID's move assignment transfers
        /// `other`'s uuid over and invalidates `other`.
        inline LifetimeTracker &operator=(LifetimeTracker &&other) noexcept {
            if (this != &other) {
                trace_destroy();
                shambase::WithUUID<LifetimeTracker, u64>::operator=(std::move(other));
                updated = other.updated;
            }
            return *this;
        }

        inline void trace_create() {
            if (on_create) {
                on_create(this->uuid);
            }
        }
        /// Fires the destroy notification, if not already fired or moved from. Idempotent: safe
        /// to call explicitly (e.g. to control ordering relative to other teardown logic) and
        /// again later from the destructor.
        inline void trace_destroy() {
            if (this->is_alive()) {
                if (on_destroy) {
                    on_destroy(this->uuid);
                }
                this->invalidate();
            }
        }
        inline void trace_state_update(T &object) {
            if (this->is_alive()) {
                updated = true;
                if (on_state_update) {
                    on_state_update(object);
                }
            }
        }
        inline void trace_op(u64 op_id) {
            if (this->is_alive() && on_op) {
                on_op(this->uuid, op_id);
            }
        }

        /// Destructor, notifies the destruction of the tracked object (unless already notified,
        /// or this instance was moved from).
        ~LifetimeTracker() { trace_destroy(); };
    };

} // namespace shamrock::solvergraph
