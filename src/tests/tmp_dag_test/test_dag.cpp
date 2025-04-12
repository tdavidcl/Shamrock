// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/shamtest.hpp"
#include <unordered_map>
#include <iostream>
#include <memory>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <vector>

namespace shambase {
    template<class T>
    inline T &get_check_ref(const std::shared_ptr<T> &ptr) {
        if (!bool(ptr)) {
            throw std::runtime_error("the ptr does not hold anything");
        }
        return *ptr;
    }
} // namespace shambase

class INode;

template<class Func,class Func2>
void multi_evaluate(std::vector<std::shared_ptr<INode>> endpoints, Func &&f, Func2&& condition);

template<class Func,class Func2>
inline void multi_evaluate_up(std::vector<std::shared_ptr<INode>> endpoints, Func &&f, Func2&& condition);

class IDataEdge {
    public:
    std::weak_ptr<INode> child;
    std::weak_ptr<INode> parent;

    template<class Func>
    inline void on_child(Func &&f) {
        if (auto spt = child.lock()) {
            f(shambase::get_check_ref(spt));
        } else {
            throw "";
        }
    }

    template<class Func>
    inline void on_parent(Func &&f) {
        if (auto spt = parent.lock()) {
            f(shambase::get_check_ref(spt));
        } else {
            throw "";
        }
    }

    template<class Func>
    inline void on_links(Func &&f) {
        auto spt_c = child.lock();
        auto spt_p = parent.lock();

        if (!bool(spt_c)) {
            throw "";
        }
        if (!bool(spt_p)) {
            throw "";
        }

        f(*spt_c, *spt_p);
    }

    inline std::string get_label() { return _impl_get_label(); }

    inline std::string get_tex_symbol() { return _impl_get_tex_symbol(); }

    virtual std::string _impl_get_label()      = 0;
    virtual std::string _impl_get_tex_symbol() = 0;

    virtual ~IDataEdge() {}
};

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
        multi_evaluate({getptr_shared()}, [&](auto &current) {
            std::cout << "evaluating " << current.get_label() << std::endl;
            current._impl_evaluate_internal();
            current.evaluated = true;
        }, [&](auto & n){
            return !n.evaluated;
        });
    }

    inline void reset_up(){
        multi_evaluate_up({getptr_shared()}, [&](auto &current) {
            std::cout << "resetting " << current.get_label() << std::endl;
            current._impl_reset_internal();
            current.evaluated = false;
        }, [&](auto & n){
            return n.evaluated;
        });
    }
    inline void reset_down(){
        multi_evaluate({getptr_shared()}, [&](auto &current) {
            std::cout << "resetting " << current.get_label() << std::endl;
            current._impl_reset_internal();
            current.evaluated = false;
        }, [&](auto & n){
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
    virtual void _impl_reset_internal()   = 0;
    virtual std::string _impl_get_label()    = 0;
    virtual std::string _impl_get_node_tex() = 0;
};

////////////////////////////////////:
// traversal
////////////////////////////////////:

template<class Func,class Func2>
void multi_evaluate(std::vector<std::shared_ptr<INode>> endpoints, Func &&f, Func2&& condition){

    std::stack<INode *> stack;
    for (auto &n : endpoints) {
        auto & ref = shambase::get_check_ref(n);
        if(condition(ref)){
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
        //print_stack();
        auto current = stack.top();

        if (!unrolled_map[current]) {
            current->on_childrens([&](auto &nchild) {
                if(condition(nchild)){
                    std::cout << "push " << &nchild<< " eval = " << nchild.evaluated << " cd = " << condition(nchild) <<std::endl;
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

template<class Func,class Func2>
inline void multi_evaluate_up(std::vector<std::shared_ptr<INode>> endpoints, Func &&f, Func2&& condition){

    std::stack<INode *> stack;
    for (auto &n : endpoints) {
        auto & ref = shambase::get_check_ref(n);
        if(condition(ref)){
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
        //print_stack();
        auto current = stack.top();

        if (!unrolled_map[current]) {
            current->on_parents([&](auto &nchild) {
                if(condition(nchild)){
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

inline std::string get_node_graph_tex(std::vector<std::shared_ptr<INode>> endpoints) {
    std::stringstream output;

    output << "\\documentclass{article}\n";
    output << "\\begin{document}\n";
    multi_evaluate(endpoints, [&](auto &current) {
        output << current.get_node_tex() << "\n";
    }, [](auto & n){
        return true;
    });
    output << "\\end{document}\n";

    return output.str();
}

inline std::string get_node_dot_graph(std::vector<std::shared_ptr<INode>> endpoints) {

    struct Node {
        std::string id;
        std::string label;
    };
    struct Edge {
        std::string child;
        std::string parent;
        std::string label;
    };

    int i = 0;
    std::unordered_map<INode *, Node> node_map;
    multi_evaluate(endpoints, [&](auto &current) {
        node_map[&current] = {.id = std::to_string(i++), .label = current.get_label()};
    }, [](auto & n){
        return true;
    });

    std::unordered_map<IDataEdge *, Edge> edge_map;

    auto register_edge = [&](IDataEdge &edge) {
        edge.on_links([&](auto &child, auto &parent) {
            edge_map[&edge]
                = {.child  = node_map[&child].id,
                   .parent = node_map[&parent].id,
                   .label  = edge.get_label()};
        });
    };

    for (auto &[ptr, n] : node_map) {
        for (auto &in : ptr->inputs) {
            register_edge(shambase::get_check_ref(in));
        }
        for (auto &out : ptr->outputs) {
            register_edge(shambase::get_check_ref(out));
        }
    }

    std::stringstream output;

    output << "digraph G {\n";

    for (auto &[ptr, n] : node_map) {
        output << "node_" << n.id << " [label=\"" << n.label << "\"];\n";
    }

    for (auto &[ptr, e] : edge_map) {
        output << "node_" << e.child << " -> node_" << e.parent << " [label=\"" << e.label
               << "\"];\n";
    }

    output << "}\n";

    return output.str();
}

class Field : public IDataEdge {
    std::string name;
    std::string texsymbol;

    public:
    std::vector<float> field_data = {};
    Field(std::string name, std::string texsymbol)
        : name(name), texsymbol(texsymbol), field_data() {}

    inline std::string _impl_get_label() { return name; };
    inline std::string _impl_get_tex_symbol() { return texsymbol; };
};

class RhoOp : public INode {
    public:
    void _impl_evaluate_internal() {
        auto &h    = get_input<Field>(0);
        auto &mass = get_input<Field>(1);
        auto &rho  = get_output<Field>(0);

        rho.field_data.resize(h.field_data.size());
        for (int i = 0; i < h.field_data.size(); i++) {
            rho.field_data[i] = mass.field_data[i] / h.field_data[i];
        }
    }

    void _impl_reset_internal(){
         get_output<Field>(0).field_data = {};
    }

    inline void set_inputs(std::shared_ptr<Field> h, std::shared_ptr<Field> mass) {
        __internal_set_inputs({h, mass});
    }
    inline void set_outputs(std::shared_ptr<Field> rho) { __internal_set_outputs({rho}); }

    inline std::string _impl_get_label() { return "Compute rho"; }
    inline std::string _impl_get_node_tex() {
        std::string h    = get_input<Field>(0).get_tex_symbol() + "_a";
        std::string mass = get_input<Field>(1).get_tex_symbol() + "_a";
        std::string rho  = get_output<Field>(0).get_tex_symbol() + "_a";

        return "\\[" + rho + " = \\frac{" + mass + "}{" + h + "}\\]";
    }
};
class FieldLoader : public INode {
    public:
    std::vector<float> mirror;

    FieldLoader(std::vector<float> input) : mirror(input) {}
    FieldLoader(float val, int n) : mirror(n, val) {}

    void _impl_evaluate_internal() {
        auto &field      = get_output<Field>(0);
        field.field_data = mirror;
    }

    void _impl_reset_internal(){
        mirror.resize(0);
        auto &field      = get_output<Field>(0);
        field.field_data = {};
    }

    inline void set_inputs() { __internal_set_inputs({}); }
    inline void set_outputs(std::shared_ptr<Field> field) { __internal_set_outputs({field}); }

    inline std::string _impl_get_label() { return "Loader"; }
    inline std::string _impl_get_node_tex() {
        std::string field = get_output<Field>(0).get_tex_symbol() + "_a";

        return "\\[" + field + " \\leftarrow field\\]";
    }
};

class FieldWriter : public INode {
    public:
    std::vector<float> &write_back;

    FieldWriter(std::vector<float> &write_back) : write_back(write_back) {}

    void _impl_evaluate_internal() {
        auto &field = get_input<Field>(0);
        write_back  = field.field_data;
    }


    void _impl_reset_internal(){
        write_back = {};
    }

    inline void set_inputs(std::shared_ptr<Field> field) { __internal_set_inputs({field}); }
    inline void set_outputs() { __internal_set_outputs({}); }

    inline std::string _impl_get_label() { return "Writer"; }
    inline std::string _impl_get_node_tex() {
        std::string field = get_input<Field>(0).get_tex_symbol() + "_a";

        return "\\[" + field + " \\rightarrow field\\]";
    }
};

TestStart(Unittest, "dag_stuff_testing", dag_stuff_testing, 1) {

    int n                         = 3;
    float m_source                = 3;
    std::vector<float> h_source   = {1., 1. / 2., 1. / 4.};
    std::vector<float> rho_result = {};

    std::shared_ptr<FieldLoader> h_load    = std::make_shared<FieldLoader>(h_source);
    std::shared_ptr<FieldLoader> mass_load = std::make_shared<FieldLoader>(m_source, n);
    std::shared_ptr<FieldWriter> rho_write = std::make_shared<FieldWriter>(rho_result);

    std::shared_ptr<Field> h    = std::make_shared<Field>("h", "h");
    std::shared_ptr<Field> mass = std::make_shared<Field>("m", "m");
    std::shared_ptr<Field> rho  = std::make_shared<Field>("rho", "\\rho");

    std::shared_ptr<RhoOp> rho_op = std::make_shared<RhoOp>();

    h_load->set_outputs(h);
    mass_load->set_outputs(mass);
    rho_op->set_inputs(h, mass);
    rho_op->set_outputs(rho);
    rho_write->set_inputs(rho);

    std::cout << get_node_graph_tex({rho_write}) << std::endl;
    std::cout << get_node_dot_graph({rho_write}) << std::endl;

    rho_write->evaluate();
    std::cout << "Rho result" << std::endl;
    std::cout << "size " << rho_result.size() << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << i << " " << rho_result[i] << std::endl;
    }
}

TestStart(Unittest, "dag_stuff_testing_reset", dag_stuff_testingreset, 1) {

    int n                         = 3;
    float m_source                = 3;
    std::vector<float> h_source   = {1., 1. / 2., 1. / 4.};
    std::vector<float> rho_result = {};
    std::vector<float> P_result = {};
    std::vector<float> dt_result = {};

    std::shared_ptr<FieldLoader> h_load    = std::make_shared<FieldLoader>(h_source);
    std::shared_ptr<FieldLoader> mass_load = std::make_shared<FieldLoader>(m_source, n);
    std::shared_ptr<FieldWriter> rho_write = std::make_shared<FieldWriter>(rho_result);

    std::shared_ptr<Field> h    = std::make_shared<Field>("h", "h");
    std::shared_ptr<Field> mass = std::make_shared<Field>("m", "m");
    std::shared_ptr<Field> rho  = std::make_shared<Field>("rho", "\\rho");

    std::shared_ptr<RhoOp> rho_op = std::make_shared<RhoOp>();

    h_load->set_outputs(h);
    mass_load->set_outputs(mass);
    rho_op->set_inputs(h, mass);
    rho_op->set_outputs(rho);
    rho_write->set_inputs(rho);

    std::cout << get_node_graph_tex({rho_write}) << std::endl;
    std::cout << get_node_dot_graph({rho_write}) << std::endl;

    auto print_node_states = [&](){
        std::cout << "- h_load    eval=" << h_load->evaluated << std::endl;
        std::cout << "- mass_load eval=" << mass_load->evaluated << std::endl;
        std::cout << "- rho_op    eval=" << rho_op->evaluated << std::endl;
        std::cout << "- rho_write eval=" << rho_write->evaluated << std::endl;
    };


    print_node_states();
    rho_write->evaluate();
    std::cout << "Rho result" << std::endl;
    std::cout << "size " << rho_result.size() << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << i << " " << rho_result[i] << std::endl;
    }

    print_node_states();

    std::cout << " -- retry evaluate :" << std::endl;
    rho_write->evaluate();
    print_node_states();
    std::cout << " -- reset write :" << std::endl;
    rho_write->reset_up();
    print_node_states();
    std::cout << " -- retry evaluate :" << std::endl;
    rho_write->evaluate();
    print_node_states();
    std::cout << " -- reset only rhoop and up :" << std::endl;
    rho_op->reset_up();
    print_node_states();
    std::cout << " -- retry evaluate :" << std::endl;
    rho_write->evaluate();
    print_node_states();
}
