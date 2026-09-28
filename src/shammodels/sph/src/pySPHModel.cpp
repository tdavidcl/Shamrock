// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHModel.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/logs/loglevels.hpp"
#include "shambase/memory.hpp"
#include "pySPHModel_add_instance.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammath/crystalLattice.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/common/modules/ComputeGravWave.hpp"
#include "shammodels/common/shamrock_json_to_py_json.hpp"
#include "shammodels/sph/Model.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shammodels/sph/modules/AnalysisAngularMomentum.hpp"
#include "shammodels/sph/modules/AnalysisBarycenter.hpp"
#include "shammodels/sph/modules/AnalysisDisc.hpp"
#include "shammodels/sph/modules/AnalysisDustMass.hpp"
#include "shammodels/sph/modules/AnalysisEnergyKinetic.hpp"
#include "shammodels/sph/modules/AnalysisEnergyPotential.hpp"
#include "shammodels/sph/modules/AnalysisSodTube.hpp"
#include "shammodels/sph/modules/AnalysisTotalMomentum.hpp"
#include "shammodels/sph/modules/render/CartesianRender.hpp"
#include "shammodels/sph/modules/render/RenderFieldGetter.hpp"
#include "shammodels/sph/sink_edges_helper.hpp"
#include "shamphys/SodTube.hpp"
#include "shampylib/PatchDataToPy.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include <experimental/mdspan>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <memory>
#include <optional>
#include <random>
#include <utility>

// The C2, C4 & C6 instantiations are in pySPHModel_C_kernels.cpp
template void add_instance<f64_3, shammath::M4>(
    py::module &m, std::string name_config, std::string name_model);
template void add_instance<f64_3, shammath::M6>(
    py::module &m, std::string name_config, std::string name_model);
template void add_instance<f64_3, shammath::M8>(
    py::module &m, std::string name_config, std::string name_model);

template<class Tvec, template<class> class SPHKernel>
void add_analysisBarycenter_instance(py::module &m, const std::string &name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;

    using T = Model<Tvec, SPHKernel>;

    py::class_<modules::AnalysisBarycenter<Tvec, SPHKernel>>(m, name_model.c_str())
        .def(py::init([](T &model) {
            return std::make_unique<modules::AnalysisBarycenter<Tvec, SPHKernel>>(model);
        }))
        .def("get_barycenter", [](modules::AnalysisBarycenter<Tvec, SPHKernel> &self) {
            auto result = self.get_barycenter();
            return py::make_tuple(result.barycenter, result.mass_disc);
        });
}

template<class Tvec, template<class> class SPHKernel>
void add_analysisEnergyKinetic_instance(py::module &m, const std::string &name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;
    using T     = Model<Tvec, SPHKernel>;

    py::class_<modules::AnalysisEnergyKinetic<Tvec, SPHKernel>>(m, name_model.c_str())
        .def(py::init([](T &model) {
            return std::make_unique<modules::AnalysisEnergyKinetic<Tvec, SPHKernel>>(model);
        }))
        .def("get_kinetic_energy", [](modules::AnalysisEnergyKinetic<Tvec, SPHKernel> &self) {
            return self.get_kinetic_energy();
        });
}

template<class Tvec, template<class> class SPHKernel>
void add_analysisEnergyPotential_instance(py::module &m, const std::string &name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;
    using T     = Model<Tvec, SPHKernel>;

    py::class_<modules::AnalysisEnergyPotential<Tvec, SPHKernel>>(m, name_model.c_str())
        .def(py::init([](T &model) {
            return std::make_unique<modules::AnalysisEnergyPotential<Tvec, SPHKernel>>(model);
        }))
        .def("get_potential_energy", [](modules::AnalysisEnergyPotential<Tvec, SPHKernel> &self) {
            return self.get_potential_energy();
        });
}

template<class Tvec, template<class> class SPHKernel>
void add_analysisTotalMomentum_instance(py::module &m, const std::string &name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;
    using T     = Model<Tvec, SPHKernel>;

    py::class_<modules::AnalysisTotalMomentum<Tvec, SPHKernel>>(m, name_model.c_str())
        .def(py::init([](T &model) {
            return std::make_unique<modules::AnalysisTotalMomentum<Tvec, SPHKernel>>(model);
        }))
        .def("get_total_momentum", [](modules::AnalysisTotalMomentum<Tvec, SPHKernel> &self) {
            return self.get_total_momentum();
        });
}

template<class Tvec, template<class> class SPHKernel>
void add_analysisAngularMomentum_instance(py::module &m, const std::string &name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;
    using T     = Model<Tvec, SPHKernel>;

    py::class_<modules::AnalysisAngularMomentum<Tvec, SPHKernel>>(m, name_model.c_str())
        .def(py::init([](T &model) {
            return std::make_unique<modules::AnalysisAngularMomentum<Tvec, SPHKernel>>(model);
        }))
        .def("get_angular_momentum", [](modules::AnalysisAngularMomentum<Tvec, SPHKernel> &self) {
            return self.get_angular_momentum();
        });
}

template<class Tvec, template<class> class SPHKernel>
void add_analysisDustMass_instance(py::module &m, const std::string &name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;
    using T     = Model<Tvec, SPHKernel>;

    py::class_<modules::AnalysisDustMass<Tvec, SPHKernel>>(m, name_model.c_str())
        .def(py::init([](T &model) {
            return std::make_unique<modules::AnalysisDustMass<Tvec, SPHKernel>>(model);
        }))
        .def("get_dust_mass", [](modules::AnalysisDustMass<Tvec, SPHKernel> &self) {
            return self.get_dust_mass();
        });
}

using namespace shammodels::sph;

template<class Analysis, typename Tvec, template<class> class SPHKernel>
auto analysis_impl(shammodels::sph::Model<Tvec, SPHKernel> &model) -> Analysis {
    return Analysis(model);
}

template<template<class, template<class> class> class Analysis>
void register_analysis_impl_for_each_kernel(py::module &msph, const char *name_class) {
    using namespace shammodels::sph;

    using SPHModel_f64_3_M4 = shammodels::sph::Model<f64_3, shammath::M4>;
    using SPHModel_f64_3_M6 = shammodels::sph::Model<f64_3, shammath::M6>;
    using SPHModel_f64_3_M8 = shammodels::sph::Model<f64_3, shammath::M8>;

    using SPHModel_f64_3_C2 = shammodels::sph::Model<f64_3, shammath::C2>;
    using SPHModel_f64_3_C4 = shammodels::sph::Model<f64_3, shammath::C4>;
    using SPHModel_f64_3_C6 = shammodels::sph::Model<f64_3, shammath::C6>;

    msph.def(
        name_class,
        [](SPHModel_f64_3_M4 &model) {
            return analysis_impl<Analysis<f64_3, shammath::M4>>(model);
        },
        py::kw_only(),
        py::arg("model"));

    msph.def(
        name_class,
        [](SPHModel_f64_3_M6 &model) {
            return analysis_impl<Analysis<f64_3, shammath::M6>>(model);
        },
        py::kw_only(),
        py::arg("model"));

    msph.def(
        name_class,
        [](SPHModel_f64_3_M8 &model) {
            return analysis_impl<Analysis<f64_3, shammath::M8>>(model);
        },
        py::kw_only(),
        py::arg("model"));

    msph.def(
        name_class,
        [](SPHModel_f64_3_C2 &model) {
            return analysis_impl<Analysis<f64_3, shammath::C2>>(model);
        },
        py::kw_only(),
        py::arg("model"));

    msph.def(
        name_class,
        [](SPHModel_f64_3_C4 &model) {
            return analysis_impl<Analysis<f64_3, shammath::C4>>(model);
        },
        py::kw_only(),
        py::arg("model"));

    msph.def(
        name_class,
        [](SPHModel_f64_3_C6 &model) {
            return analysis_impl<Analysis<f64_3, shammath::C6>>(model);
        },
        py::kw_only(),
        py::arg("model"));
}

ON_PYTHON_INIT {
    auto &m = root_module;

    py::module msph = m.def_submodule("model_sph", "Shamrock sph solver");

    py::class_<shamrock::PatchDataLazyGetter>(m, "PatchDataLazyGetter")
        .def("__getitem__", &shamrock::PatchDataLazyGetter::get_item);

    py::class_<EvolveUntilResults>(m, "EvolveUntilResults")
        .def_readwrite("reach_target_time", &EvolveUntilResults::reach_target_time)
        .def_readwrite("reach_niter_max", &EvolveUntilResults::reach_niter_max)
        .def_readwrite("reach_max_walltime", &EvolveUntilResults::reach_max_walltime)
        .def_readwrite("iter_count", &EvolveUntilResults::iter_count)
        .def("__repr__", [](const EvolveUntilResults &self) {
            return sham::format(
                "EvolveUntilResults(reach_target_time={}, reach_niter_max={}, "
                "reach_max_walltime={}, iter_count={})",
                self.reach_target_time,
                self.reach_niter_max,
                self.reach_max_walltime,
                self.iter_count);
        });

    using namespace shammodels::sph;

    add_instance<f64_3, shammath::M4>(msph, "SPHModel_f64_3_M4_SolverConfig", "SPHModel_f64_3_M4");
    add_instance<f64_3, shammath::M6>(msph, "SPHModel_f64_3_M6_SolverConfig", "SPHModel_f64_3_M6");
    add_instance<f64_3, shammath::M8>(msph, "SPHModel_f64_3_M8_SolverConfig", "SPHModel_f64_3_M8");

    add_instance<f64_3, shammath::C2>(msph, "SPHModel_f64_3_C2_SolverConfig", "SPHModel_f64_3_C2");
    add_instance<f64_3, shammath::C4>(msph, "SPHModel_f64_3_C4_SolverConfig", "SPHModel_f64_3_C4");
    add_instance<f64_3, shammath::C6>(msph, "SPHModel_f64_3_C6_SolverConfig", "SPHModel_f64_3_C6");

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

    add_analysisBarycenter_instance<f64_3, shammath::M4>(msph, "AnalysisBarycenter_f64_3_M4");
    add_analysisBarycenter_instance<f64_3, shammath::M6>(msph, "AnalysisBarycenter_f64_3_M6");
    add_analysisBarycenter_instance<f64_3, shammath::M8>(msph, "AnalysisBarycenter_f64_3_M8");

    add_analysisBarycenter_instance<f64_3, shammath::C2>(msph, "AnalysisBarycenter_f64_3_C2");
    add_analysisBarycenter_instance<f64_3, shammath::C4>(msph, "AnalysisBarycenter_f64_3_C4");
    add_analysisBarycenter_instance<f64_3, shammath::C6>(msph, "AnalysisBarycenter_f64_3_C6");

    add_analysisEnergyKinetic_instance<f64_3, shammath::M4>(msph, "AnalysisEnergyKinetic_f64_3_M4");
    add_analysisEnergyKinetic_instance<f64_3, shammath::M6>(msph, "AnalysisEnergyKinetic_f64_3_M6");
    add_analysisEnergyKinetic_instance<f64_3, shammath::M8>(msph, "AnalysisEnergyKinetic_f64_3_M8");

    add_analysisEnergyKinetic_instance<f64_3, shammath::C2>(msph, "AnalysisEnergyKinetic_f64_3_C2");
    add_analysisEnergyKinetic_instance<f64_3, shammath::C4>(msph, "AnalysisEnergyKinetic_f64_3_C4");
    add_analysisEnergyKinetic_instance<f64_3, shammath::C6>(msph, "AnalysisEnergyKinetic_f64_3_C6");

    add_analysisEnergyPotential_instance<f64_3, shammath::M4>(
        msph, "AnalysisEnergyPotential_f64_3_M4");
    add_analysisEnergyPotential_instance<f64_3, shammath::M6>(
        msph, "AnalysisEnergyPotential_f64_3_M6");
    add_analysisEnergyPotential_instance<f64_3, shammath::M8>(
        msph, "AnalysisEnergyPotential_f64_3_M8");

    add_analysisEnergyPotential_instance<f64_3, shammath::C2>(
        msph, "AnalysisEnergyPotential_f64_3_C2");
    add_analysisEnergyPotential_instance<f64_3, shammath::C4>(
        msph, "AnalysisEnergyPotential_f64_3_C4");
    add_analysisEnergyPotential_instance<f64_3, shammath::C6>(
        msph, "AnalysisEnergyPotential_f64_3_C6");

    add_analysisTotalMomentum_instance<f64_3, shammath::M4>(msph, "AnalysisTotalMomentum_f64_3_M4");
    add_analysisTotalMomentum_instance<f64_3, shammath::M6>(msph, "AnalysisTotalMomentum_f64_3_M6");
    add_analysisTotalMomentum_instance<f64_3, shammath::M8>(msph, "AnalysisTotalMomentum_f64_3_M8");

    add_analysisTotalMomentum_instance<f64_3, shammath::C2>(msph, "AnalysisTotalMomentum_f64_3_C2");
    add_analysisTotalMomentum_instance<f64_3, shammath::C4>(msph, "AnalysisTotalMomentum_f64_3_C4");
    add_analysisTotalMomentum_instance<f64_3, shammath::C6>(msph, "AnalysisTotalMomentum_f64_3_C6");

    add_analysisAngularMomentum_instance<f64_3, shammath::M4>(
        msph, "AnalysisAngularMomentum_f64_3_M4");
    add_analysisAngularMomentum_instance<f64_3, shammath::M6>(
        msph, "AnalysisAngularMomentum_f64_3_M6");
    add_analysisAngularMomentum_instance<f64_3, shammath::M8>(
        msph, "AnalysisAngularMomentum_f64_3_M8");

    add_analysisAngularMomentum_instance<f64_3, shammath::C2>(
        msph, "AnalysisAngularMomentum_f64_3_C2");
    add_analysisAngularMomentum_instance<f64_3, shammath::C4>(
        msph, "AnalysisAngularMomentum_f64_3_C4");
    add_analysisAngularMomentum_instance<f64_3, shammath::C6>(
        msph, "AnalysisAngularMomentum_f64_3_C6");

    register_analysis_impl_for_each_kernel<modules::AnalysisBarycenter>(msph, "analysisBarycenter");
    register_analysis_impl_for_each_kernel<modules::AnalysisEnergyKinetic>(
        msph, "analysisEnergyKinetic");
    register_analysis_impl_for_each_kernel<modules::AnalysisEnergyPotential>(
        msph, "analysisEnergyPotential");
    register_analysis_impl_for_each_kernel<modules::AnalysisTotalMomentum>(
        msph, "analysisTotalMomentum");
    register_analysis_impl_for_each_kernel<modules::AnalysisAngularMomentum>(
        msph, "analysisAngularMomentum");

    add_analysisDustMass_instance<f64_3, shammath::M4>(msph, "AnalysisDustMass_f64_3_M4");
    add_analysisDustMass_instance<f64_3, shammath::M6>(msph, "AnalysisDustMass_f64_3_M6");
    add_analysisDustMass_instance<f64_3, shammath::M8>(msph, "AnalysisDustMass_f64_3_M8");

    add_analysisDustMass_instance<f64_3, shammath::C2>(msph, "AnalysisDustMass_f64_3_C2");
    add_analysisDustMass_instance<f64_3, shammath::C4>(msph, "AnalysisDustMass_f64_3_C4");
    add_analysisDustMass_instance<f64_3, shammath::C6>(msph, "AnalysisDustMass_f64_3_C6");

    register_analysis_impl_for_each_kernel<modules::AnalysisDustMass>(msph, "analysisDustMass");
}
