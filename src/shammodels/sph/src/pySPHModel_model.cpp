// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHModel_model.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief SPH model class Python bindings.
 */

#include "shambase/exception.hpp"
#include "shambase/logs/loglevels.hpp"
#include "shambase/memory.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammath/crystalLattice.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/Model.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shammodels/sph/pySPHModelBindings.hpp"
#include "shammodels/sph/sink_edges_helper.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include <pybind11/cast.h>
#include <memory>
#include <optional>
#include <random>
#include <utility>

namespace shammodels::sph::pysph {

    template<class Tvec, template<class> class SPHKernel>
    void add_model(py::module &m, const std::string &name_model) {
        using namespace shammodels::sph;

        using Tscal = shambase::VecComponent<Tvec>;
        using T     = Model<Tvec, SPHKernel>;

        shamlog_debug_ln("[Py]", "registering class :", name_model, typeid(T).name());

        py::class_<T> cls(m, name_model.c_str());
        cls.def(py::init([](ShamrockCtx &ctx) {
               return std::make_unique<T>(ctx);
           }))
            .def("init", &T::init)
            .def("init_scheduler", &T::init_scheduler)

            .def(
                "evolve_once_override_time",
                &T::evolve_once_time_expl,
                py::arg("t_curr"),
                py::arg("dt_input"))
            .def("evolve_once", &T::evolve_once)
            .def(
                "evolve_until",
                [](T &self, f64 target_time, i32 niter_max, f64 max_walltime) {
                    return self.evolve_until(target_time, niter_max, max_walltime);
                },
                py::arg("target_time"),
                py::kw_only(),
                py::arg("niter_max")    = -1,
                py::arg("max_walltime") = -1)
            .def("timestep", &T::timestep)
            .def("set_cfl_cour", &T::set_cfl_cour, py::arg("cfl_cour"))
            .def("set_cfl_force", &T::set_cfl_force, py::arg("cfl_force"))
            .def("set_eta_sink", &T::set_eta_sink, py::arg("eta_sink"))
            .def("set_particle_mass", &T::set_particle_mass, py::arg("gpart_mass"))
            .def("get_particle_mass", &T::get_particle_mass)
            .def("rho_h", &T::rho_h)
            .def("get_hfact", &T::get_hfact)
            .def(
                "get_solver_tex",
                [](T &self) {
                    return shambase::get_check_ref(self.solver.storage.solver_sequence).get_tex();
                })
            .def(
                "get_solver_dot_graph",
                [](T &self) {
                    return shambase::get_check_ref(self.solver.storage.solver_sequence)
                        .get_dot_graph();
                })
            .def(
                "get_box_dim_fcc_3d",
                [](T &self, f64 dr, u32 xcnt, u32 ycnt, u32 zcnt) {
                    return self.get_box_dim_fcc_3d(dr, xcnt, ycnt, zcnt);
                })
            .def(
                "get_ideal_fcc_box",
                [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                    ON_RANK_0(
                        shamcomm::logs::warn_ln(
                            "SPH",
                            "The python function get_ideal_fcc_box is deprecated in the SPH model "
                            "and "
                            "will be removed at some point, replace it by "
                            "shamrock.math.get_ideal_hcp_box"));
                    return shammath::LatticeHCP<f64_3>::get_ideal_hcp_box(dr, {box_min, box_max});
                })
            .def(
                "get_ideal_hcp_box",
                [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                    ON_RANK_0(
                        shamcomm::logs::warn_ln(
                            "SPH",
                            "The python function get_ideal_hcp_box is deprecated in the SPH model "
                            "and "
                            "will be removed at some point, replace it by "
                            "shamrock.math.get_ideal_hcp_box"));
                    return shammath::LatticeHCP<f64_3>::get_ideal_hcp_box(dr, {box_min, box_max});
                })
            .def(
                "resize_simulation_box",
                [](T &self, f64_3 box_min, f64_3 box_max) {
                    return self.resize_simulation_box({box_min, box_max});
                })
            .def(
                "push_particle",
                [](T &self,
                   std::vector<f64_3> pos,
                   std::vector<f64> hpart,
                   std::vector<f64> upart) {
                    return self.push_particle(pos, hpart, upart);
                })
            .def(
                "push_particle_mhd",
                [](T &self,
                   std::vector<f64_3> pos,
                   std::vector<f64> hpart,
                   std::vector<f64> upart,
                   std::vector<f64_3> B_on_rho,
                   std::vector<f64> psi_on_ch) {
                    return self.push_particle_mhd(pos, hpart, upart, B_on_rho, psi_on_ch);
                })
            .def(
                "add_cube_fcc_3d",
                [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                    return self.add_cube_fcc_3d(dr, {box_min, box_max});
                })
            .def(
                "add_cube_hcp_3d",
                [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                    return self.add_cube_hcp_3d(dr, {box_min, box_max});
                })
            .def(
                "add_cube_hcp_3d_v2",
                [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                    return self.add_cube_hcp_3d_v2(dr, {box_min, box_max});
                })
            .def(
                "add_disc_3d_keplerian",
                [](T &self,
                   Tvec center,
                   u32 Npart,
                   Tscal p,
                   Tscal rho_0,
                   Tscal m,
                   Tscal r_in,
                   Tscal r_out,
                   Tscal q,
                   Tscal cmass) {
                    return self.add_cube_disc_3d(center, Npart, p, rho_0, m, r_in, r_out, q, cmass);
                })
            .def(
                "add_disc_3d",
                [](T &self,
                   Tvec center,
                   Tscal central_mass,
                   u32 Npart,
                   Tscal r_in,
                   Tscal r_out,
                   Tscal disc_mass,
                   Tscal p,
                   Tscal H_r_in,
                   Tscal q) {
                    return self.add_disc_3d(
                        center, central_mass, Npart, r_in, r_out, disc_mass, p, H_r_in, q);
                })
            .def(
                "add_big_disc_3d",
                [](T &self,
                   Tvec center,
                   Tscal central_mass,
                   u32 Npart,
                   Tscal r_in,
                   Tscal r_out,
                   Tscal disc_mass,
                   Tscal p,
                   Tscal H_r_in,
                   Tscal q,
                   u16 seed) {
                    self.add_big_disc_3d(
                        center,
                        central_mass,
                        Npart,
                        r_in,
                        r_out,
                        disc_mass,
                        p,
                        H_r_in,
                        q,
                        std::mt19937{seed});
                    return disc_mass / Npart;
                })
            .def("get_total_part_count", &T::get_total_part_count)
            .def("total_mass_to_part_mass", &T::total_mass_to_part_mass)
            .def(
                "set_value_in_a_box",
                [](T &self,
                   const std::string &field_name,
                   const std::string &field_type,
                   const pybind11::object &value,
                   f64_3 box_min,
                   f64_3 box_max,
                   u32 ivar) {
                    if (field_type == "f64") {
                        f64 val = value.cast<f64>();
                        self.set_value_in_a_box(field_name, val, {box_min, box_max}, ivar);
                    } else if (field_type == "f64_3") {
                        f64_3 val = value.cast<f64_3>();
                        self.set_value_in_a_box(field_name, val, {box_min, box_max}, ivar);
                    } else {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "unknown field type");
                    }
                },
                py::arg("field_name"),
                py::arg("field_type"),
                py::arg("value"),
                py::arg("box_min"),
                py::arg("box_max"),
                py::kw_only(),
                py::arg("ivar") = 0)
            .def(
                "set_value_in_sphere",
                [](T &self,
                   const std::string &field_name,
                   const std::string &field_type,
                   const pybind11::object &value,
                   f64_3 center,
                   f64 radius) {
                    if (field_type == "f64") {
                        f64 val = value.cast<f64>();
                        self.set_value_in_sphere(field_name, val, center, radius);
                    } else if (field_type == "f64_3") {
                        f64_3 val = value.cast<f64_3>();
                        self.set_value_in_sphere(field_name, val, center, radius);
                    } else {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "unknown field type");
                    }
                })
            .def(
                "set_field_value_lambda_f64",
                [](T &self,
                   std::string field_name,
                   const std::function<f64(Tvec)> pos_to_val,
                   const u32 offset) {
                    return self.template set_field_value_lambda<f64>(
                        std::move(field_name), pos_to_val, offset);
                },
                py::arg("field_name"),
                py::arg("pos_to_val"),
                py::arg("offset") = 0)
            .def(
                "set_field_value_lambda_f64_3",
                [](T &self,
                   std::string field_name,
                   const std::function<f64_3(Tvec)> pos_to_val,
                   const u32 offset) {
                    return self.template set_field_value_lambda<f64_3>(
                        std::move(field_name), pos_to_val, offset);
                },
                py::arg("field_name"),
                py::arg("pos_to_val"),
                py::arg("offset") = 0)
            .def("overwrite_field_value_f64", &T::template overwrite_field_value<f64>)
            .def("overwrite_field_value_f64_3", &T::template overwrite_field_value<f64_3>)
            .def("remap_positions", &T::remap_positions)
            //.def("set_field_value_lambda_f64_3",[](T&self,std::string field_name, const
            // std::function<f64_3 (Tscal, Tscal , Tscal)> pos_to_val){
            //    self.template set_field_value_lambda<f64_3>(field_name, [=](Tvec v){
            //        return pos_to_val(v.x(), v.y(),v.z());
            //    });
            //})
            .def(
                "add_kernel_value",
                [](T &self,
                   const std::string &field_name,
                   const std::string &field_type,
                   const pybind11::object &value,
                   f64_3 center,
                   f64 h_ker) {
                    if (field_type == "f64") {
                        f64 val = value.cast<f64>();
                        self.add_kernel_value(field_name, val, center, h_ker);
                    } else if (field_type == "f64_3") {
                        f64_3 val = value.cast<f64_3>();
                        self.add_kernel_value(field_name, val, center, h_ker);
                    } else {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "unknown field type");
                    }
                })
            .def(
                "get_sum",
                [](T &self, const std::string &field_name, const std::string &field_type) {
                    if (field_type == "f64") {
                        return py::cast(self.template get_sum<f64>(field_name));
                    } else if (field_type == "f64_3") {
                        return py::cast(self.template get_sum<f64_3>(field_name));
                    } else {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "unknown field type");
                    }
                })
            .def(
                "get_closest_part_to",
                [](T &self, f64_3 pos) -> f64_3 {
                    return self.get_closest_part_to(pos);
                })
            .def(
                "gen_default_config",
                [](T &self) {
                    return typename T::Solver::Config{};
                })
            .def(
                "get_current_config",
                [](T &self) {
                    return self.solver.solver_config;
                })
            .def("set_solver_config", &T::set_solver_config)
            .def("add_sink", &T::add_sink)
            .def(
                "get_sinks",
                [](T &self) {
                    py::list list_out;

                    auto edges = get_sink_edges<Tvec>(
                        shambase::get_check_ref(self.ctx.sched).synchronized_data);
                    for (auto &sink : to_sink_particles(edges)) {
                        py::dict sink_dic;
                        sink_dic["pos"]              = sink.pos;
                        sink_dic["velocity"]         = sink.velocity;
                        sink_dic["sph_acceleration"] = sink.sph_acceleration;
                        sink_dic["ext_acceleration"] = sink.ext_acceleration;
                        sink_dic["mass"]             = sink.mass;
                        sink_dic["angular_momentum"] = sink.angular_momentum;
                        sink_dic["accretion_radius"] = sink.accretion_radius;
                        list_out.append(sink_dic);
                    }

                    return list_out;
                })
            .def("get_units", [](T &self) {
                return self.solver.solver_config.unit_sys;
            });

        cls.def(
               "gen_config_from_phantom_dump",
               [](T &self, PhantomDump &dump, bool bypass_error) {
                   return self.gen_config_from_phantom_dump(dump, bypass_error);
               },
               py::arg("dump"),
               py::arg("bypass_error") = false,
               R"==(
    This function generate a shamrock sph solver config from a phantom dump

    Parameters
    ----------
    PhantomDump dump
    bypass_error = false (default) bypass any error in the config
)==")
            .def(
                "init_from_phantom_dump",
                [](T &self, PhantomDump &dump, Tscal hpart_fact_load) {
                    self.init_from_phantom_dump(dump, hpart_fact_load);
                },
                py::arg("dump"),
                py::arg("hpart_fact_load") = 1.0)
            .def(
                "make_phantom_dump",
                [](T &self) {
                    return self.make_phantom_dump();
                })
            .def("do_vtk_dump", &T::do_vtk_dump)
            .def("set_debug_dump", &T::set_debug_dump)
            .def("solver_logs_last_rate", &T::solver_logs_last_rate)
            .def("solver_logs_last_obj_count", &T::solver_logs_last_obj_count)
            .def(
                "solver_logs_last_system_metrics",
                [&](T &self) {
                    auto system_metrics = self.solver.solve_logs.get_last_system_metrics();
                    py::dict ret;
                    ret["duration"] = system_metrics.wall_time;
                    if (system_metrics.rank_energy_consummed.has_value()) {
                        ret["rank_energy_consummed"] = system_metrics.rank_energy_consummed.value();
                    }
                    if (system_metrics.gpu_energy_consummed.has_value()) {
                        ret["gpu_energy_consummed"] = system_metrics.gpu_energy_consummed.value();
                    }
                    if (system_metrics.cpu_energy_consummed.has_value()) {
                        ret["cpu_energy_consummed"] = system_metrics.cpu_energy_consummed.value();
                    }
                    if (system_metrics.dram_energy_consummed.has_value()) {
                        ret["dram_energy_consummed"] = system_metrics.dram_energy_consummed.value();
                    }
                    return ret;
                })
            .def("solver_logs_cumulated_step_time", &T::solver_logs_cumulated_step_time)
            .def("solver_logs_reset_cumulated_step_time", &T::solver_logs_reset_cumulated_step_time)
            .def("solver_logs_step_count", &T::solver_logs_step_count)
            .def("solver_logs_reset_step_count", &T::solver_logs_reset_step_count)
            .def(
                "get_time",
                [](T &self) {
                    return self.get_time();
                })
            .def(
                "get_dt",
                [](T &self) {
                    return self.get_dt_sph();
                })
            .def(
                "set_time",
                [](T &self, Tscal t) {
                    return self.set_time(t);
                })
            .def(
                "set_next_dt",
                [](T &self, Tscal dt) {
                    return self.set_next_dt(dt);
                })
            .def(
                "set_dt",
                [](T &self, f64 dt) {
                    self.set_next_dt(dt);
                })
            .def(
                "set_cfl_multipler",
                [](T &self, Tscal lambda) {
                    return self.set_cfl_multipler(lambda);
                },
                py::arg("lambda"))
            .def(
                "set_cfl_mult_stiffness",
                [](T &self, Tscal cstiff) {
                    return self.solver.solver_config.set_cfl_mult_stiffness(cstiff);
                },
                py::arg("cstiff"))
            .def(
                "change_htolerance",
                [](T &self, Tscal in) {
                    ON_RANK_0(shamlog_warn_ln(
                                  "SPH",
                                  ".change_htolerance(val) is deprecated,\n"
                                  "    -> calling this is replaced internally by "
                                  ".change_htolerances(coarse=val, fine=min(val, 1.1))\n"
                                  "    see: "
                                  "https://shamrock-code.github.io/Shamrock/mkdocs/models/sph/"
                                  "smoothing_length_tolerance"););
                    self.change_htolerances(in, std::min(in, (Tscal) 1.1));
                })
            .def(
                "change_htolerances",
                [](T &self, Tscal coarse, Tscal fine) {
                    self.change_htolerances(coarse, fine);
                },
                py::kw_only(),
                py::arg("coarse"),
                py::arg("fine"));

        cls.def("load_from_dump", &T::load_from_dump)
            .def("dump", &T::dump)
            .def("get_setup", &T::get_setup)
            .def(
                "get_patch_transform",
                [](T &self) {
                    PatchScheduler &sched = shambase::get_check_ref(self.ctx.sched);
                    return sched.get_patch_transform<Tvec>();
                })
            .def("apply_momentum_offset", &T::apply_momentum_offset)
            .def("apply_position_offset", &T::apply_position_offset)
            .def(
                "add_timestep_callback",
                [](T &self,
                   std::optional<std::function<void(void)>> step_begin_callback,
                   std::optional<std::function<void(void)>> step_end_callback) {
                    self.solver.timestep_callbacks.push_back(
                        {std::move(step_begin_callback), std::move(step_end_callback)});
                },
                py::kw_only(),
                py::arg("step_begin") = std::nullopt,
                py::arg("step_end")   = std::nullopt);
    }

} // namespace shammodels::sph::pysph

SHAMROCK_SPH_PYBIND_INSTANTIATE(shammodels::sph::pysph::add_model)
