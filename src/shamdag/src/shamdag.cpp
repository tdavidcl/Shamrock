// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file shamdag.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamdag/shamdag.hpp"
#include "shamdag/INode.hpp"
#include <stack>

void multi_evaluate(
    std::vector<std::shared_ptr<INode>> endpoints,
    std::function<void(INode &)> f,
    std::function<bool(INode &)> condition) {

    std::stack<INode *> stack;
    for (auto &n : endpoints) {
        auto &ref = shambase::get_check_ref(n);
        if (condition(ref)) {
            stack.push(&ref);
        }
    }

    auto print_stack = [&]() {
        std::cout << "-----------------" << std::endl;
        std::stack<INode *> stack2 = stack;
        while (!stack2.empty()) {
            std::cout << "Node " << stack2.top() << std::endl;
            stack2.pop();
        }
    };

    std::unordered_map<INode *, bool> unrolled_map  = {};
    std::unordered_map<INode *, bool> evaluated_map = {};

    while (!stack.empty()) {
        // print_stack();
        auto current = stack.top();

        if (!unrolled_map[current]) {
            current->on_childrens([&](auto &nchild) {
                if (condition(nchild)) {
                    std::cout << "push " << &nchild << " eval = " << nchild.is_evaluated()
                              << " cd = " << condition(nchild) << std::endl;
                    stack.push(&nchild);
                }
            });
            unrolled_map[current] = true;
        } else if (!evaluated_map[current]) {

            // std::cout << "eval " << current << std::endl;
            f(*current);
            evaluated_map[current] = true;
        } else {
            stack.pop();
        }
    }
}

void multi_evaluate_up(
    std::vector<std::shared_ptr<INode>> endpoints,
    std::function<void(INode &)> f,
    std::function<bool(INode &)> condition) {

    std::stack<INode *> stack;
    for (auto &n : endpoints) {
        auto &ref = shambase::get_check_ref(n);
        if (condition(ref)) {
            stack.push(&ref);
        }
    }

    auto print_stack = [&]() {
        std::cout << "-----------------" << std::endl;
        std::stack<INode *> stack2 = stack;
        while (!stack2.empty()) {
            std::cout << "Node " << stack2.top() << std::endl;
            stack2.pop();
        }
    };

    std::unordered_map<INode *, bool> unrolled_map  = {};
    std::unordered_map<INode *, bool> evaluated_map = {};

    while (!stack.empty()) {
        // print_stack();
        auto current = stack.top();

        if (!unrolled_map[current]) {
            current->on_parents([&](auto &nchild) {
                if (condition(nchild)) {
                    stack.push(&nchild);
                }
            });
            unrolled_map[current] = true;
        } else if (!evaluated_map[current]) {

            // std::cout << "eval " << current << std::endl;
            f(*current);
            evaluated_map[current] = true;
        } else {
            stack.pop();
        }
    }
}
