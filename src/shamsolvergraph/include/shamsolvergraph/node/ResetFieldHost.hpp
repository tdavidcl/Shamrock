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
 * @file ResetFieldHost.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node that resets a host-side (std::vector) field to its default value.
 */

#include "shambase/stacktrace.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <vector>

#define NODE_EDGES(X_RO, X_RW) X_RW(shamrock::solvergraph::IDataEdge<std::vector<T>>, field)

namespace shamrock::solvergraph {

    /**
     * @brief Reset a host-side (std::vector) field to its default value: field[i] = T{}
     *
     * @tparam T The value type stored in the field vector
     */
    template<class T>
    class ResetFieldHost : public INode {

        public:
        ResetFieldHost() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {
            __shamrock_stack_entry();

            auto edges = get_edges();

            std::vector<T> &field = edges.field.data;

            for (size_t i = 0; i < field.size(); i++) {
                field[i] = T{};
            }
        }

        inline virtual std::string _impl_get_label() const { return "ResetFieldHost"; }

        inline virtual std::string _impl_get_tex() const { return "TODO"; }
    };

} // namespace shamrock::solvergraph

#undef NODE_EDGES
