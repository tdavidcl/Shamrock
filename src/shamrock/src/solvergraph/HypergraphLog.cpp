// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file INode.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamcomm/logs.hpp"
#include <memory>

#include "shamrock/solvergraph/HypergraphLog.hpp"
#include "shamrock/solvergraph/INode.hpp"

namespace shamrock::solvergraph {

    void log_new_inode(u64 uuid) {
        shamcomm::logs::raw_ln("New inode created: ", uuid);
    }

    void log_del_inode(u64 uuid) {
        shamcomm::logs::raw_ln("Inode deleted: ", uuid);
    }

} // namespace shamrock::solvergraph
