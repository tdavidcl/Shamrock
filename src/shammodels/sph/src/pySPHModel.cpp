// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more
// information
//
// -------------------------------------------------------//

/**
 * @file pySPHModel.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief SPH Python module init, factory, and shared types.
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/Model.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shammodels/sph/pySPHModelBindings.hpp"
#include <memory>
#include <variant>

using namespace shammodels::sph;
using namespace shammodels::sph::pysph;

template<template<class> class Kernel>
void bind_sph_kernel(
    py::module &msph, const std::string &config_name, const std::string &model_name) {
    add_config<f64_3, Kernel>(msph, config_name);
    add_setup<f64_3, Kernel>(msph, model_name);
    add_model<f64_3, Kernel>(msph, model_name);
    add_render<f64_3, Kernel>(msph, model_name);
    add_analysis<f64_3, Kernel>(msph, model_name);
}

ON_PYTHON_INIT {
    auto &m = root_module;

    py::module msph = m.def_submodule("model_sph", "Shamrock sph solver");

    py::class_<EvolveUntilResults>(m, "EvolveUntilResults")
        .def_readwrite("reach_target_time", &EvolveUntilResults::reach_target_time)
        .def_readwrite("reach_niter_max", &EvolveUntilResults::reach_niter_max)
        .def_readwrite("reach_max_walltime", &EvolveUntilResults::reach_max_walltime)
        .def_readwrite("iter_count", &EvolveUntilResults::iter_count)
        .def("__repr__", [](const EvolveUntilResults &self) {
            return shambase::format(
                "EvolveUntilResults(reach_target_time={}, reach_niter_max={}, "
                "reach_max_walltime={}, iter_count={})",
                self.reach_target_time,
                self.reach_niter_max,
                self.reach_max_walltime,
                self.iter_count);
        });

    bind_sph_kernel<shammath::M4>(msph, "SPHModel_f64_3_M4_SolverConfig", "SPHModel_f64_3_M4");
    bind_sph_kernel<shammath::M6>(msph, "SPHModel_f64_3_M6_SolverConfig", "SPHModel_f64_3_M6");
    bind_sph_kernel<shammath::M8>(msph, "SPHModel_f64_3_M8_SolverConfig", "SPHModel_f64_3_M8");
    bind_sph_kernel<shammath::C2>(msph, "SPHModel_f64_3_C2_SolverConfig", "SPHModel_f64_3_C2");
    bind_sph_kernel<shammath::C4>(msph, "SPHModel_f64_3_C4_SolverConfig", "SPHModel_f64_3_C4");
    bind_sph_kernel<shammath::C6>(msph, "SPHModel_f64_3_C6_SolverConfig", "SPHModel_f64_3_C6");

    using VariantSPHModelBind = std::variant<
        std::unique_ptr<Model<f64_3, shammath::M4>>,
        std::unique_ptr<Model<f64_3, shammath::M6>>,
        std::unique_ptr<Model<f64_3, shammath::M8>>,
        std::unique_ptr<Model<f64_3, shammath::C2>>,
        std::unique_ptr<Model<f64_3, shammath::C4>>,
        std::unique_ptr<Model<f64_3, shammath::C6>>>;

    m.def(
        "get_Model_SPH",
        [](ShamrockCtx &ctx,
           const std::string &vector_type,
           const std::string &kernel) -> VariantSPHModelBind {
            VariantSPHModelBind ret;

            if (vector_type == "f64_3" && kernel == "M4") {
                ret = std::make_unique<Model<f64_3, shammath::M4>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "M6") {
                ret = std::make_unique<Model<f64_3, shammath::M6>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "M8") {
                ret = std::make_unique<Model<f64_3, shammath::M8>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "C2") {
                ret = std::make_unique<Model<f64_3, shammath::C2>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "C4") {
                ret = std::make_unique<Model<f64_3, shammath::C4>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "C6") {
                ret = std::make_unique<Model<f64_3, shammath::C6>>(ctx);
            } else {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    "unknown combination of representation and kernel");
            }

            return ret;
        },
        py::kw_only(),
        py::arg("context"),
        py::arg("vector_type"),
        py::arg("sph_kernel"));

    py::class_<
        shammodels::sph::modules::ISPHSetupNode,
        std::shared_ptr<shammodels::sph::modules::ISPHSetupNode>>(msph, "ISPHSetupNode")
        .def("get_dot", [](std::shared_ptr<shammodels::sph::modules::ISPHSetupNode> &self) {
            return self->get_dot();
        });

    py::class_<shammodels::sph::TimestepLog>(msph, "TimestepLog")
        .def(py::init<>())
        .def_readwrite("rank", &shammodels::sph::TimestepLog::rank)
        .def_readwrite("rate", &shammodels::sph::TimestepLog::rate)
        .def_readwrite("npart", &shammodels::sph::TimestepLog::npart)
        .def_readwrite("tcompute", &shammodels::sph::TimestepLog::tcompute)
        .def("rate_sum", &shammodels::sph::TimestepLog::rate_sum)
        .def("npart_sum", &shammodels::sph::TimestepLog::npart_sum);

    register_sph_analysis_python(msph);
}
