// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHModel_analysis.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief SPH analysis Python bindings.
 */

#include "shambase/memory.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/Model.hpp"
#include "shammodels/sph/modules/AnalysisAngularMomentum.hpp"
#include "shammodels/sph/modules/AnalysisBarycenter.hpp"
#include "shammodels/sph/modules/AnalysisDisc.hpp"
#include "shammodels/sph/modules/AnalysisDustMass.hpp"
#include "shammodels/sph/modules/AnalysisEnergyKinetic.hpp"
#include "shammodels/sph/modules/AnalysisEnergyPotential.hpp"
#include "shammodels/sph/modules/AnalysisSodTube.hpp"
#include "shammodels/sph/modules/AnalysisTotalMomentum.hpp"
#include "shammodels/sph/pySPHModelBindings.hpp"
#include "shamphys/SodTube.hpp"
#include <pybind11/cast.h>
#include <memory>

namespace shammodels::sph::pysph {

    template<class Tvec, template<class> class SPHKernel>
    void add_analysis(py::module &m, const std::string &name_model) {
        using namespace shammodels::sph;

        using Tscal            = shambase::VecComponent<Tvec>;
        using T                = Model<Tvec, SPHKernel>;
        using TAnalysisSodTube = shammodels::sph::modules::AnalysisSodTube<Tvec, SPHKernel>;
        using TAnalysisDisc    = shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>;

        std::string sod_tube_analysis_name = name_model + "_AnalysisSodTube";
        py::class_<TAnalysisSodTube>(m, sod_tube_analysis_name.c_str())
            .def("compute_L2_dist", [](TAnalysisSodTube &self) -> std::tuple<Tscal, Tvec, Tscal> {
                auto ret = self.compute_L2_dist();
                return {ret.rho, ret.v, ret.P};
            });

        std::string disc_analysis_name = name_model + "_AnalysisDisc";
        py::class_<TAnalysisDisc>(m, disc_analysis_name.c_str())
            .def(
                "collect_data",
                [](TAnalysisDisc &self, Tscal Rmin, Tscal Rmax, u32 Nbin, ShamrockCtx &ctx) {
                    auto anal = self.compute_analysis(Rmin, Rmax, Nbin, ctx);
                    py::dict dic_out;

                    auto radius  = anal.radius.copy_to_stdvec();
                    auto counter = anal.counter.copy_to_stdvec();
                    auto Sigma   = anal.Sigma.copy_to_stdvec();
                    auto lx      = anal.lx.copy_to_stdvec();
                    auto ly      = anal.ly.copy_to_stdvec();
                    auto lz      = anal.lz.copy_to_stdvec();
                    auto tilt    = anal.tilt.copy_to_stdvec();
                    auto twist   = anal.twist.copy_to_stdvec();
                    auto psi     = anal.psi.copy_to_stdvec();
                    auto Hsq     = anal.Hsq.copy_to_stdvec();

                    dic_out["radius"]  = radius;
                    dic_out["counter"] = counter;
                    dic_out["Sigma"]   = Sigma;
                    dic_out["lx"]      = lx;
                    dic_out["ly"]      = ly;
                    dic_out["lz"]      = lz;
                    dic_out["tilt"]    = tilt;
                    dic_out["twist"]   = twist;
                    dic_out["psi"]     = psi;
                    dic_out["Hsq"]     = Hsq;

                    return dic_out;
                });

        auto cls = registered_class<T>();

        cls.def(
               "make_analysis_sodtube",
               [](T &self,
                  shamphys::SodTube sod,
                  Tvec direction,
                  Tscal time_val,
                  Tscal x_ref,
                  Tscal x_min,
                  Tscal x_max) {
                   return std::make_unique<TAnalysisSodTube>(
                       self.ctx,
                       self.solver.solver_config,
                       self.solver.storage,
                       sod,
                       direction,
                       time_val,
                       x_ref,
                       x_min,
                       x_max);
               },
               py::arg("sod"),
               py::arg("direction"),
               py::arg("time_val"),
               py::arg("x_ref"),
               py::arg("x_min"),
               py::arg("x_max"))
            .def("make_analysis_disc", [](T &self) {
                return std::make_unique<TAnalysisDisc>(
                    self.ctx, self.solver.solver_config, self.solver.storage);
            });
    }

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
            .def(
                "get_potential_energy",
                [](modules::AnalysisEnergyPotential<Tvec, SPHKernel> &self) {
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
            .def(
                "get_angular_momentum",
                [](modules::AnalysisAngularMomentum<Tvec, SPHKernel> &self) {
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

    void register_sph_analysis_python(py::module &msph) {
        using namespace shammodels::sph;

        add_analysisBarycenter_instance<f64_3, shammath::M4>(msph, "AnalysisBarycenter_f64_3_M4");
        add_analysisBarycenter_instance<f64_3, shammath::M6>(msph, "AnalysisBarycenter_f64_3_M6");
        add_analysisBarycenter_instance<f64_3, shammath::M8>(msph, "AnalysisBarycenter_f64_3_M8");

        add_analysisBarycenter_instance<f64_3, shammath::C2>(msph, "AnalysisBarycenter_f64_3_C2");
        add_analysisBarycenter_instance<f64_3, shammath::C4>(msph, "AnalysisBarycenter_f64_3_C4");
        add_analysisBarycenter_instance<f64_3, shammath::C6>(msph, "AnalysisBarycenter_f64_3_C6");

        add_analysisEnergyKinetic_instance<f64_3, shammath::M4>(
            msph, "AnalysisEnergyKinetic_f64_3_M4");
        add_analysisEnergyKinetic_instance<f64_3, shammath::M6>(
            msph, "AnalysisEnergyKinetic_f64_3_M6");
        add_analysisEnergyKinetic_instance<f64_3, shammath::M8>(
            msph, "AnalysisEnergyKinetic_f64_3_M8");

        add_analysisEnergyKinetic_instance<f64_3, shammath::C2>(
            msph, "AnalysisEnergyKinetic_f64_3_C2");
        add_analysisEnergyKinetic_instance<f64_3, shammath::C4>(
            msph, "AnalysisEnergyKinetic_f64_3_C4");
        add_analysisEnergyKinetic_instance<f64_3, shammath::C6>(
            msph, "AnalysisEnergyKinetic_f64_3_C6");

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

        add_analysisTotalMomentum_instance<f64_3, shammath::M4>(
            msph, "AnalysisTotalMomentum_f64_3_M4");
        add_analysisTotalMomentum_instance<f64_3, shammath::M6>(
            msph, "AnalysisTotalMomentum_f64_3_M6");
        add_analysisTotalMomentum_instance<f64_3, shammath::M8>(
            msph, "AnalysisTotalMomentum_f64_3_M8");

        add_analysisTotalMomentum_instance<f64_3, shammath::C2>(
            msph, "AnalysisTotalMomentum_f64_3_C2");
        add_analysisTotalMomentum_instance<f64_3, shammath::C4>(
            msph, "AnalysisTotalMomentum_f64_3_C4");
        add_analysisTotalMomentum_instance<f64_3, shammath::C6>(
            msph, "AnalysisTotalMomentum_f64_3_C6");

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

        register_analysis_impl_for_each_kernel<modules::AnalysisBarycenter>(
            msph, "analysisBarycenter");
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

} // namespace shammodels::sph::pysph

SHAMROCK_SPH_PYBIND_INSTANTIATE(shammodels::sph::pysph::add_analysis)
