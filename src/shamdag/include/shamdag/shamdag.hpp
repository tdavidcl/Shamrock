// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file shamdag.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <functional>
#include <memory>
#include <vector>

class INode;
class IDataEdge;

void multi_evaluate(
    std::vector<std::shared_ptr<INode>> endpoints,
    std::function<void(INode &)> f,
    std::function<bool(INode &)> condition);

void multi_evaluate_up(
    std::vector<std::shared_ptr<INode>> endpoints,
    std::function<void(INode &)> f,
    std::function<bool(INode &)> condition);

std::vector<std::shared_ptr<INode>> filter_duplicate(std::vector<std::shared_ptr<INode>> nodes);

std::vector<std::shared_ptr<INode>> get_endpoints(std::vector<std::shared_ptr<INode>> nodes);
std::vector<std::shared_ptr<INode>> get_entrypoints(std::vector<std::shared_ptr<INode>> nodes);
