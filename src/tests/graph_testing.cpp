// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file fmmTests.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/WithUUID.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/string.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class AccessState {
    public:
    enum class access_t { ro, rw, none };
    access_t access_state = access_t::none;

    inline void set_state_ro() {
        if (access_state == access_t::none) {
            access_state = access_t::ro;
        } else {
            shambase::throw_with_loc<std::runtime_error>("State must be none to set ro");
        }
    }
    inline void set_state_rw() {
        if (access_state == access_t::none) {
            access_state = access_t::rw;
        } else {
            shambase::throw_with_loc<std::runtime_error>("State must be none to set rw");
        }
    }
    inline void set_state_none() {
        if (access_state != access_t::none) {
            access_state = access_t::none;
        } else {
            shambase::throw_with_loc<std::runtime_error>("State must not be none to set none");
        }
    }
};

class INode;

class IDataEdge : public AccessState, public shambase::WithUUID<IDataEdge, u64> {
    public:
    // std::weak_ptr<INode> child;
    // std::weak_ptr<INode> parent;

    // std::vector<std::weak_ptr<INode>> read_access_log;

    inline std::string get_label() { return _impl_get_label(); }
    inline std::string get_tex_symbol() { return _impl_get_tex_symbol(); }

    virtual std::string _impl_get_label()      = 0;
    virtual std::string _impl_get_tex_symbol() = 0;

    virtual ~IDataEdge() {}
};

// Inode is node between data edges, takes multiple inputs, multiple outputs
class INode : public std::enable_shared_from_this<INode>, public shambase::WithUUID<INode, u64> {

    std::vector<std::shared_ptr<IDataEdge>> ro_edges;
    std::vector<std::shared_ptr<IDataEdge>> rw_edges;

    public:
    inline std::shared_ptr<INode> getptr_shared() { return shared_from_this(); }
    inline std::weak_ptr<INode> getptr_weak() { return weak_from_this(); }

    inline std::vector<std::shared_ptr<IDataEdge>> &get_ro_edges() { return ro_edges; }
    inline std::vector<std::shared_ptr<IDataEdge>> &get_rw_edges() { return rw_edges; }

    inline void __internal_set_ro_edges(std::vector<std::shared_ptr<IDataEdge>> new_ro_edges) {
        for (auto e : ro_edges) {
            // shambase::get_check_ref(e).parent = {};
        }
        this->ro_edges = new_ro_edges;
        for (auto e : ro_edges) {
            // shambase::get_check_ref(e).parent = getptr_weak();
        }
    }

    inline void __internal_set_rw_edges(std::vector<std::shared_ptr<IDataEdge>> new_rw_edges) {
        for (auto e : rw_edges) {
            // shambase::get_check_ref(e).child = {};
        }
        this->rw_edges = new_rw_edges;
        for (auto e : rw_edges) {
            // shambase::get_check_ref(e).child = getptr_weak();
        }
    }

    template<class Func>
    inline void on_edge_ro_edges(Func &&f) {
        for (auto &in : ro_edges) {
            f(shambase::get_check_ref(in));
        }
    }

    template<class Func>
    inline void on_edge_rw_edges(Func &&f) {
        for (auto &out : rw_edges) {
            f(shambase::get_check_ref(out));
        }
    }

    virtual ~INode() {
        __internal_set_ro_edges({});
        __internal_set_rw_edges({});
    }

    template<class T>
    T &get_ro_edge(int slot) {
        return shambase::get_check_ref(std::dynamic_pointer_cast<T>(ro_edges.at(slot)));
    }

    template<class T>
    T &get_rw_edge(int slot) {
        return shambase::get_check_ref(std::dynamic_pointer_cast<T>(rw_edges.at(slot)));
    }

    inline std::string get_dot_graph() { return get_partial_dot_graph(); };
    inline std::string get_node_tex() { return get_partial_node_tex(); };

    inline std::string get_partial_dot_graph() { return _impl_get_dot_subgraph(); };
    inline std::string get_dot_graph_node_start() { return _impl_get_node_dot_start(); };
    inline std::string get_dot_graph_node_end() { return _impl_get_node_dot_end(); };
    inline std::string get_partial_node_tex() { return _impl_get_node_tex(); };

    inline void evaluate() {
        on_edge_ro_edges([](auto &e) {
            e.set_state_ro();
        });
        on_edge_rw_edges([](auto &e) {
            e.set_state_rw();
        });
        _impl_evaluate_internal();
        on_edge_ro_edges([](auto &e) {
            e.set_state_none();
        });
        on_edge_rw_edges([](auto &e) {
            e.set_state_none();
        });
    }

    inline void reset() {
        on_edge_ro_edges([](auto &e) {
            e.set_state_ro();
        });
        on_edge_rw_edges([](auto &e) {
            e.set_state_rw();
        });
        _impl_reset_internal();
        on_edge_ro_edges([](auto &e) {
            e.set_state_none();
        });
        on_edge_rw_edges([](auto &e) {
            e.set_state_none();
        });
    }

    protected:
    virtual void _impl_evaluate_internal() = 0;
    virtual void _impl_reset_internal()    = 0;
    virtual std::string _impl_get_label()  = 0;

    inline virtual std::string _impl_get_dot_subgraph() {
        std::string node_str
            = shambase::format("n_{} [label=\"{}\"];\n", this->get_uuid(), _impl_get_label());

        std::string edge_str = "";
        for (auto &in : ro_edges) {
            edge_str += shambase::format(
                "e_{} -> n_{} [style=\"dashed\", color=green];\n",
                in->get_uuid(),
                this->get_uuid());
            edge_str += shambase::format(
                "e_{} [label=\"{}\",shape=rect, style=filled];\n", in->get_uuid(), in->get_label());
        }
        for (auto &out : rw_edges) {
            edge_str += shambase::format(
                "n_{} -> e_{} [style=\"dashed\", color=red];\n", this->get_uuid(), out->get_uuid());
            edge_str += shambase::format(
                "e_{} [label=\"{}\",shape=rect, style=filled];\n",
                out->get_uuid(),
                out->get_label());
        }

        return shambase::format("{}{}", node_str, edge_str);
    };

    inline virtual std::string _impl_get_node_dot_start() {
        return shambase::format("n_{}", this->get_uuid());
    }
    inline virtual std::string _impl_get_node_dot_end() {
        return shambase::format("n_{}", this->get_uuid());
    }

    virtual std::string _impl_get_node_tex() = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// test api
////////////////////////////////////////////////////////////////////////////////////////////////////

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
        auto &h    = get_ro_edge<Field>(0);
        auto &mass = get_ro_edge<Field>(1);
        auto &rho  = get_rw_edge<Field>(0);

        rho.field_data.resize(h.field_data.size());
        for (int i = 0; i < h.field_data.size(); i++) {
            rho.field_data[i] = mass.field_data[i] / h.field_data[i];
        }
    }

    void _impl_reset_internal() { get_ro_edge<Field>(0).field_data = {}; }

    inline void set_inputs(std::shared_ptr<Field> h, std::shared_ptr<Field> mass) {
        __internal_set_ro_edges({h, mass});
    }
    inline void set_outputs(std::shared_ptr<Field> rho) { __internal_set_rw_edges({rho}); }

    inline std::string _impl_get_label() { return "Compute rho"; }

    inline std::string _impl_get_node_tex() {
        std::string h    = get_ro_edge<Field>(0).get_tex_symbol() + "_a";
        std::string mass = get_ro_edge<Field>(1).get_tex_symbol() + "_a";
        std::string rho  = get_rw_edge<Field>(0).get_tex_symbol() + "_a";

        return "\\[" + rho + " = \\frac{" + mass + "}{" + h + "}\\]";
    }
};
class FieldLoader : public INode {
    public:
    std::vector<float> mirror;

    FieldLoader(std::vector<float> input) : mirror(input) {}
    FieldLoader(float val, int n) : mirror(n, val) {}

    void _impl_evaluate_internal() {
        auto &field      = get_rw_edge<Field>(0);
        field.field_data = mirror;
    }

    void _impl_reset_internal() {
        mirror.resize(0);
        auto &field      = get_rw_edge<Field>(0);
        field.field_data = {};
    }

    inline void set_inputs() { __internal_set_ro_edges({}); }
    inline void set_outputs(std::shared_ptr<Field> field) { __internal_set_rw_edges({field}); }

    inline std::string _impl_get_label() { return "Loader"; }

    inline std::string _impl_get_node_tex() {
        std::string field = get_rw_edge<Field>(0).get_tex_symbol() + "_a";

        return "\\[" + field + " \\leftarrow field\\]";
    }
};

class FieldWriter : public INode {
    public:
    std::vector<float> &write_back;

    FieldWriter(std::vector<float> &write_back) : write_back(write_back) {}

    void _impl_evaluate_internal() {
        auto &field = get_ro_edge<Field>(0);
        write_back  = field.field_data;
    }

    void _impl_reset_internal() { write_back = {}; }

    inline void set_inputs(std::shared_ptr<Field> field) { __internal_set_ro_edges({field}); }
    inline void set_outputs() { __internal_set_rw_edges({}); }

    inline std::string _impl_get_label() { return "Writer"; }

    inline std::string _impl_get_node_tex() {
        std::string field = get_ro_edge<Field>(0).get_tex_symbol() + "_a";

        return "\\[" + field + " \\rightarrow field\\]";
    }
};

class OperationSequence : public INode {
    std::vector<std::shared_ptr<INode>> nodes;

    public:
    OperationSequence(std::vector<std::shared_ptr<INode>> nodes) : nodes(nodes) {
        if (nodes.size() == 0) {
            shambase::throw_with_loc<std::invalid_argument>(
                "OperationSequence must have at least one node");
        }
    }
    void _impl_evaluate_internal() {
        for (auto &node : nodes) {
            node->evaluate();
        }
    }

    void _impl_reset_internal() {
        for (int i = nodes.size() - 1; i >= 0; i--) {
            nodes[i]->reset();
        }
    }

    std::string _impl_get_label() { return "Sequence"; }

    std::string _impl_get_dot_subgraph() {
        std::stringstream ss;

        ss << "subgraph cluster_" + std::to_string(get_uuid()) + " {\n";
        for (auto &node : nodes) {
            ss << node->get_partial_dot_graph();
        }

        for (int i = 0; i < nodes.size() - 1; i++) {
            ss << nodes[i]->get_dot_graph_node_end() << " -> "
               << nodes[i + 1]->get_dot_graph_node_start() << " [weight=3];\n";
        }

        ss << shambase::format("label = \"{}\";\n", _impl_get_label());
        ss << "}\n";

        return ss.str();
    };

    inline virtual std::string _impl_get_node_dot_start() {
        return nodes[0]->get_dot_graph_node_start();
    }
    inline virtual std::string _impl_get_node_dot_end() {
        return nodes[nodes.size() - 1]->get_dot_graph_node_end();
    }

    std::string _impl_get_node_tex() {
        std::stringstream ss;
        for (auto &node : nodes) {
            ss << node->get_partial_node_tex() << "\n";
        }
        return ss.str();
    }
};

class Looper : public INode {
    std::shared_ptr<INode> to_loop;
    int n;

    public:
    Looper(std::shared_ptr<INode> to_loop, int n) : to_loop(to_loop), n(n) {}

    void _impl_evaluate_internal() {
        for (int i = 0; i < n; i++) {
            to_loop->evaluate();
        }
    }

    void _impl_reset_internal() { to_loop->reset(); }

    std::string _impl_get_label() { return "Looper"; }

    std::string _impl_get_dot_subgraph() {
        std::stringstream ss;

        ss << "subgraph cluster_" + std::to_string(get_uuid()) + " {\n";

        std::string loop_node = _impl_get_node_dot_start();
        ss << loop_node + " [label=\"\", shape=point];\n";

        ss << to_loop->get_partial_dot_graph();

        ss << to_loop->get_dot_graph_node_end() << " -> " << loop_node
           << shambase::format(" [label=\"loop {} times\",weight=0];\n", n);
        ss << loop_node << " -> " << to_loop->get_dot_graph_node_start()
           << shambase::format(" [weight=1];\n", n);

        ss << shambase::format("label = \"{}\";\n", _impl_get_label());
        ss << "}\n";

        return ss.str();
    };

    inline virtual std::string _impl_get_node_dot_start() {
        return shambase::format("loop_{}", to_loop->get_uuid());
    }
    inline virtual std::string _impl_get_node_dot_end() {
        return to_loop->get_dot_graph_node_end();
    }

    std::string _impl_get_node_tex() {
        std::stringstream ss;
        ss << "Loop " << n << " times: {\n";
        ss << to_loop->get_partial_node_tex() << "\n";
        ss << "}\n";
        return ss.str();
    }
};

TestStart(Unittest, "tmp_graph_test", tmp_graph_test, 1) {

    int n                         = 3;
    float m_source                = 3;
    std::vector<float> h_source   = {1., 1. / 2., 1. / 4.};
    std::vector<float> rho_result = {};

    std::shared_ptr<FieldLoader> h_load = std::make_shared<FieldLoader>(h_source);
    std::shared_ptr<Field> h            = std::make_shared<Field>("h", "h");
    h_load->set_outputs(h);

    std::shared_ptr<FieldLoader> mass_load = std::make_shared<FieldLoader>(m_source, n);
    std::shared_ptr<Field> mass            = std::make_shared<Field>("m", "m");
    mass_load->set_outputs(mass);

    std::shared_ptr<FieldWriter> rho_write = std::make_shared<FieldWriter>(rho_result);
    std::shared_ptr<Field> rho             = std::make_shared<Field>("rho", "\\rho");
    rho_write->set_inputs(rho);

    std::shared_ptr<RhoOp> rho_op = std::make_shared<RhoOp>();
    rho_op->set_inputs(h, mass);
    rho_op->set_outputs(rho);

    std::shared_ptr<OperationSequence> load_seq
        = std::make_shared<OperationSequence>(OperationSequence{{h_load, mass_load}});

    std::shared_ptr<Looper> looper = std::make_shared<Looper>(rho_op, 2);

    std::shared_ptr<OperationSequence> seq
        = std::make_shared<OperationSequence>(OperationSequence{{load_seq, looper, rho_write}});

    seq->evaluate();
    std::cout << seq->get_dot_graph() << std::endl;
    std::cout << seq->get_node_tex() << std::endl;

    std::cout << "Rho result" << std::endl;
    std::cout << "size " << rho_result.size() << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << i << " " << rho_result[i] << std::endl;
    }
}
