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
 * @file ScalarEdgeSerializable.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"

namespace shamrock::solvergraph {

    template<class T>
    class ScalarEdgeSerializable : public ScalarEdge<T>, public JSonSerializable {
        public:
        using ScalarEdge<T>::ScalarEdge;
        using ScalarEdge<T>::value;
    };

} // namespace shamrock::solvergraph
