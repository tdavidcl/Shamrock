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
 * @file INode.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shamdag/IDataEdge.hpp"
#include "shamdag/shamdag.hpp"
#include <iostream>

// Inode is node between data edges, takes multiple inputs, multiple outputs
class INode : public std::enable_shared_from_this<INode> {

    public:
    std::vector<std::shared_ptr<IDataEdge>> inputs;
    std::vector<std::shared_ptr<IDataEdge>> outputs;
    bool evaluated = false;

    inline std::shared_ptr<INode> getptr_shared() { return shared_from_this(); }
    inline std::weak_ptr<INode> getptr_weak() { return weak_from_this(); }

    inline std::vector<std::shared_ptr<IDataEdge>> &get_inputs() { return inputs; }
    inline std::vector<std::shared_ptr<IDataEdge>> &get_outputs() { return outputs; }

    inline void __internal_set_inputs(std::vector<std::shared_ptr<IDataEdge>> new_inputs) {
        for (auto e : inputs) {
            shambase::get_check_ref(e).parent = {};
        }
        this->inputs = new_inputs;
        for (auto e : inputs) {
            shambase::get_check_ref(e).parent = getptr_weak();
        }
    }

    inline void __internal_set_outputs(std::vector<std::shared_ptr<IDataEdge>> new_outputs) {
        for (auto e : outputs) {
            shambase::get_check_ref(e).child = {};
        }
        this->outputs = new_outputs;
        for (auto e : outputs) {
            shambase::get_check_ref(e).child = getptr_weak();
        }
    }

    template<class Func>
    inline void on_childrens(Func &&f) {
        for (auto &in : inputs) {
            in->on_child([&](auto &child) {
                f(child);
            });
        }
    }

    template<class Func>
    inline void on_parents(Func &&f) {
        for (auto &in : outputs) {
            in->on_parent([&](auto &child) {
                f(child);
                std::cout << "on parent : " << this << " -> " << &child << std::endl;
            });
        }
    }

    virtual ~INode() {
        __internal_set_inputs({});
        __internal_set_outputs({});
    }

    inline void evaluate() {
        multi_evaluate(
            {getptr_shared()},
            [&](auto &current) {
                std::cout << "evaluating " << current.get_label() << std::endl;
                current._impl_evaluate_internal();
                current.evaluated = true;
            },
            [&](auto &n) {
                return !n.evaluated;
            });
    }

    inline void reset_up() {
        multi_evaluate_up(
            {getptr_shared()},
            [&](auto &current) {
                std::cout << "resetting " << current.get_label() << std::endl;
                current._impl_reset_internal();
                current.evaluated = false;
            },
            [&](auto &n) {
                return n.evaluated;
            });
    }
    inline void reset_down() {
        multi_evaluate(
            {getptr_shared()},
            [&](auto &current) {
                std::cout << "resetting " << current.get_label() << std::endl;
                current._impl_reset_internal();
                current.evaluated = false;
            },
            [&](auto &n) {
                return n.evaluated;
            });
    }

    template<class T>
    T &get_input(int slot) {
        return shambase::get_check_ref(std::dynamic_pointer_cast<T>(inputs.at(slot)));
    }

    template<class T>
    T &get_output(int slot) {
        return shambase::get_check_ref(std::dynamic_pointer_cast<T>(outputs.at(slot)));
    }

    inline std::string get_label() { return _impl_get_label(); };
    inline std::string get_node_tex() { return _impl_get_node_tex(); };

    protected:
    virtual void _impl_evaluate_internal()   = 0;
    virtual void _impl_reset_internal()      = 0;
    virtual std::string _impl_get_label()    = 0;
    virtual std::string _impl_get_node_tex() = 0;
};
