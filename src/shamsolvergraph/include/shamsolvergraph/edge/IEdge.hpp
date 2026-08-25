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
 * @file IEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shamsolvergraph/IFreeable.hpp"
#include "shamsolvergraph/LifetimeTracker.hpp"
#include <string>

namespace shamrock::solvergraph {

    class INode;

    class IEdge : public IFreeable {

        /// Tracks the lifetime of the edge and holds its UUID.
        /// Held as a plain value member instead of a base class (so trace_state_update() can
        /// still take a `IEdge&` to this object) and instead of a `std::shared_ptr` member (to
        /// avoid a heap allocation per edge): LifetimeTracker handles move-safety internally, so
        /// no indirection is needed here to avoid a duplicate destroy notification on move.
        LifetimeTracker<IEdge> tracker;

        public:
        IEdge() = default;

        IEdge(const IEdge &)            = delete; /// would duplicate the UUID
        IEdge &operator=(const IEdge &) = delete; /// would duplicate the UUID

        /// Move constructor - transfers identity via LifetimeTracker's own move ctor. Must be
        /// declared explicitly (not just left implicit): a user-declared destructor (below)
        /// suppresses implicit move constructor generation.
        IEdge(IEdge &&) noexcept = default;
        /// Move assignment - transfers identity via LifetimeTracker's own move assignment
        IEdge &operator=(IEdge &&) noexcept = default;

        inline std::string get_label() const { return _impl_get_dot_label(); }
        inline std::string get_tex_symbol() const { return _impl_get_tex_symbol(); }

        virtual std::string _impl_get_dot_label() const  = 0;
        virtual std::string _impl_get_tex_symbol() const = 0;

        /// Get the UUID of the edge
        inline u64 get_uuid() const { return tracker.get_uuid(); }

        inline virtual ~IEdge() {}
    };

} // namespace shamrock::solvergraph
