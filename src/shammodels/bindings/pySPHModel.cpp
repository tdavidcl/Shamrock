// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include <memory>

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamrock/sph/kernels.hpp"
#include "shammodels/SPHModel.hpp"




Register_pymod(pysphmodel) {

    using namespace shammodels;

    using T = SPHModel<f64_3, shamrock::sph::kernels::M4>;

    py::class_<T>(m, "SPHModel_f64_3_M4")
        .def(py::init([](ShamrockCtx &ctx) { return std::make_unique<T>(ctx); }))
        .def("init_scheduler", &T::init_scheduler)
        .def("evolve", &T::evolve_once)
        .def("set_cfl_cour", &T::set_cfl_cour)
        .def("set_cfl_force", &T::set_cfl_force)
        .def("set_particle_mass", &T::set_particle_mass)
        .def("get_box_dim_fcc_3d", [](T &self, f64 dr, u32 xcnt, u32 ycnt, u32 zcnt) {
            return self.get_box_dim_fcc_3d(dr, xcnt, ycnt, zcnt);
        })
        .def("get_ideal_fcc_box", [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
            return self.get_ideal_fcc_box(dr, {box_min,box_max});
        })
        .def("resize_simulation_box", [](T &self, f64_3 box_min, f64_3 box_max) {
            return self.resize_simulation_box({box_min,box_max});
        })
        .def("add_cube_fcc_3d", [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
            return self.add_cube_fcc_3d(dr, {box_min,box_max});
        })
        .def("get_total_part_count", &T::get_total_part_count)
        .def("total_mass_to_part_mass", &T::total_mass_to_part_mass)
        .def("set_value_in_a_box", [](T & self, std::string field_name, std::string field_type, pybind11::object value, f64_3 box_min, f64_3 box_max){
            if(field_type == "f64"){
                f64 val = value.cast<f64>();
                self.set_value_in_a_box(field_name, val, {box_min,box_max});
            }else if(field_type == "f64_3"){
                f64_3 val = value.cast<f64_3>();
                self.set_value_in_a_box(field_name, val, {box_min,box_max});
            }else{
                throw shambase::throw_with_loc<std::invalid_argument>(
                    "unknown field type");
            }
        })
        .def("set_value_in_sphere", [](T & self, std::string field_name, std::string field_type, pybind11::object value, f64_3 center, f64 radius){
            if(field_type == "f64"){
                f64 val = value.cast<f64>();
                self.set_value_in_sphere(field_name, val, center,radius);
            }else if(field_type == "f64_3"){
                f64_3 val = value.cast<f64_3>();
                self.set_value_in_sphere(field_name, val, center,radius);
            }else{
                throw shambase::throw_with_loc<std::invalid_argument>(
                    "unknown field type");
            }
        })
        .def("get_sum", [](T & self, std::string field_name, std::string field_type){
            if(field_type == "f64"){
                return py::cast(self.get_sum<f64>(field_name));
            }else if(field_type == "f64_3"){
                return py::cast(self.get_sum<f64_3>(field_name));
            }else{
                throw shambase::throw_with_loc<std::invalid_argument>(
                    "unknown field type");
            }
        })
        ;





    using VariantSPHModelBind =
        std::variant<std::unique_ptr<SPHModel<f64_3, shamrock::sph::kernels::M4>>>;

    m.def(
        "get_SPHModel",
        [](ShamrockCtx &ctx, std::string vector_type, std::string kernel) -> VariantSPHModelBind {
            VariantSPHModelBind ret;

            if (vector_type == "f64_3" && kernel == "M4") {
                ret = std::make_unique<SPHModel<f64_3, shamrock::sph::kernels::M4>>(ctx);
            } else {
                throw shambase::throw_with_loc<std::invalid_argument>(
                    "unknown combination of representation and kernel");
            }

            return ret;
        },
        py::kw_only(),
        py::arg("context"),
        py::arg("vector_type"),
        py::arg("sph_kernel"));
}