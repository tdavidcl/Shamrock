// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySolverGraph.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/MemPerfInfos.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include "shampylib/PatchDataToPy.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamsolvergraph/SolverGraph.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/edge/IEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsolvergraph/node/NodeFreeAlloc.hpp"
#include "shamsys/NodeInstance.hpp"
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <vector>

template<class T>
void register_data_edge(py::module &m, const char *class_name) {
    using namespace shamrock::solvergraph;

    py::class_<IDataEdge<T>, IEdge, std::shared_ptr<IDataEdge<T>>>(m, class_name)
        .def(py::init<std::string, std::string>(), py::arg("name"), py::arg("tex_symbol"))
        .def_readwrite("data", &IDataEdge<T>::data)
        .def("__repr__", [=](IDataEdge<T> &self) {
            return sham::format(
                "{}(label={}, tex_symbol={}, data={})",
                class_name,
                self.get_label(),
                self.get_tex_symbol(),
                self.data);
        });
}

template<class T>
void register_field(py::module &m, const char *class_name) {
    using namespace shamrock::solvergraph;

    py::class_<Field<T>, IEdge, std::shared_ptr<Field<T>>>(m, class_name)
        .def(
            "get_buf",
            [](Field<T> &self, u64 id_patch) -> sham::DeviceBuffer<T> & {
                return self.get_buf(id_patch);
            },
            py::return_value_policy::reference)
        .def(
            "__repr__",
            [=](Field<T> &self) {
                return sham::format(
                    "{}(label={}, tex_symbol={}, nvar={})",
                    class_name,
                    self.get_label(),
                    self.get_tex_symbol(),
                    self.get_nvar());
            })
        .def("collect_data", [](Field<T> &self) -> std::vector<T> {
            std::vector<T> base = {};
            self.get_refs().for_each([&](u64 id, std::reference_wrapper<PatchDataField<T>> &pdf) {
                auto copy = pdf.get().get_buf().copy_to_stdvec();
                base.insert(base.end(), copy.begin(), copy.end());
            });

            std::vector<T> collected = {};
            shamalgs::collective::vector_allgatherv(base, collected, MPI_COMM_WORLD);
            return collected;
        });

    std::string map_fields_name = []() -> std::string {
        if (std::is_same_v<T, f64>) {
            return "map_fields_f64";
        } else if (std::is_same_v<T, f64_3>) {
            return "map_fields_f64_3";
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>("Unsupported type");
        }
    }();

    m.def(
        map_fields_name.c_str(),
        [](py::function func,
           py::kwargs kwargs // only Field<T> are allowed
        ) {
            for (auto item : kwargs) {
                if (!py::isinstance<Field<T>>(item.second)) {
                    throw py::type_error(
                        "all keyword arguments to map_fields must be Field objects");
                }
            }

            shambase::DistributedData<u32> sizes = {};

            for (auto item : kwargs) {
                auto name = py::cast<std::string>(item.first);

                auto &field = py::cast<Field<T> &>(item.second);

                if (sizes.is_empty()) {
                    sizes = field.get_obj_cnts();
                } else {
                    field.check_sizes(sizes);
                }
            }

            Field<T> result = Field<T>(1, "ret", "ret");
            result.ensure_sizes(sizes);

            sizes.for_each([&](u64 id, u32 size) {
                py::dict call_kwargs;

                for (auto item : kwargs) {
                    auto name = py::cast<std::string>(item.first);

                    auto &field = py::cast<Field<T> &>(item.second);

                    auto vec_data = field.get(id).get_buf().copy_to_stdvec();
                    auto pyarray  = shamrock::VecToNumpy<T>::convert(vec_data);

                    call_kwargs[name.c_str()] = pyarray;
                }

                py::tuple args(1);
                args[0] = size;

                py::object py_result = func(*args, **call_kwargs);

                auto result_data = py_result.cast<std::vector<T>>();

                result.get(id).get_buf().copy_from_stdvec(result_data);
            });

            return result;
        });
}

ON_PYTHON_INIT {

    using namespace shamrock::solvergraph;

    py::class_<IEdge, std::shared_ptr<IEdge>>(root_module, "IEdge")
        .def("get_label", &IEdge::get_label)
        .def("get_tex_symbol", &IEdge::get_tex_symbol)
        .def("get_uuid", &IEdge::get_uuid)
        .def("free_alloc", &IEdge::free_alloc);

    register_field<f64>(root_module, "Field_f64");
    register_field<f64_3>(root_module, "Field_f64_3");

    register_data_edge<f64>(root_module, "IDataEdge_f64");
    register_data_edge<u32>(root_module, "IDataEdge_u32");
    register_data_edge<u64>(root_module, "IDataEdge_u64");

    py::class_<INode, std::shared_ptr<INode>>(root_module, "INode")
        .def("evaluate", &INode::evaluate)
        .def("get_label", &INode::get_label)
        .def("get_uuid", &INode::get_uuid)
        .def("get_dot_graph", &INode::get_dot_graph)
        .def("get_dot_graph_partial", &INode::get_dot_graph_partial)
        .def("print_node_info", &INode::print_node_info)
        .def("get_ro_edges", [](INode &self) { return self.get_ro_edges(); })
        .def("get_rw_edges", [](INode &self) { return self.get_rw_edges(); })
        .def("__repr__", &INode::print_node_info);

    py::class_<SolverGraph>(root_module, "SolverGraph")
        .def(py::init<>())
        .def("register_node", &SolverGraph::register_node_ptr_base, py::arg("name"), py::arg("node"))
        .def("register_edge", &SolverGraph::register_edge_ptr_base, py::arg("name"), py::arg("edge"))
        .def(
            "get_node",
            [](SolverGraph &self, const std::string &name) { return self.get_node_ptr_base(name); },
            py::arg("name"))
        .def(
            "get_edge",
            [](SolverGraph &self, const std::string &name) { return self.get_edge_ptr_base(name); },
            py::arg("name"))
        .def("has_node", &SolverGraph::has_node, py::arg("name"))
        .def("has_edge", &SolverGraph::has_edge, py::arg("name"))
        .def("get_node_names", &SolverGraph::get_node_names)
        .def("get_edge_names", &SolverGraph::get_edge_names);

    // NodeFreeAlloc is exposed as a minimal, concrete example of a node wired from Python: its
    // single read-write edge is typed as the IEdge base itself, so it works with any edge type
    // without needing extra per-type bindings, and it exercises the generic vector-based
    // INode::set_edges() overload added for Python wiring.
    py::class_<NodeFreeAlloc, INode, std::shared_ptr<NodeFreeAlloc>>(root_module, "NodeFreeAlloc")
        .def(py::init<>())
        .def(
            "set_edges",
            [](NodeFreeAlloc &self,
               std::vector<std::shared_ptr<IEdge>> ro_edges,
               std::vector<std::shared_ptr<IEdge>> rw_edges) {
                self.set_edges(std::move(ro_edges), std::move(rw_edges));
            },
            py::arg("ro_edges"),
            py::arg("rw_edges"));
}
