// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHModel_render.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief SPH render Python bindings.
 */

#include "shambase/exception.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/Model.hpp"
#include "shammodels/sph/modules/render/CartesianRender.hpp"
#include "shammodels/sph/modules/render/RenderFieldGetter.hpp"
#include "shammodels/sph/pySPHModelBindings.hpp"
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace shammodels::sph::pysph {

    template<class Tvec, template<class> class SPHKernel>
    void add_render(py::module &m, const std::string &name_model) {
        using namespace shammodels::sph;

        using Tscal           = shambase::VecComponent<Tvec>;
        using T               = Model<Tvec, SPHKernel>;
        using custom_getter_t = std::function<pybind11::array_t<f64>(size_t, pybind11::dict &)>;

        auto cls = registered_class<T>();

        cls.def(
               "render_slice",
               [](T &self,
                  const std::string &name,
                  const std::string &field_type,
                  const std::vector<Tvec> &positions,
                  const std::optional<custom_getter_t> &custom_getter)
                   -> std::variant<std::vector<f64>, std::vector<f64_3>> {
                   if (custom_getter.has_value()) {
                       if (!(name == "custom" && field_type == "f64")) {
                           throw shambase::make_except_with_loc<std::invalid_argument>(
                               "custom_getter only available for name=custom and field_type=f64");
                       }
                   }

                   if (field_type == "f64") {
                       modules::CartesianRender<Tvec, f64, SPHKernel> render(
                           self.ctx, self.solver.solver_config, self.solver.storage);
                       return render.compute_slice(name, positions, custom_getter).copy_to_stdvec();
                   }

                   if (field_type == "f64_3") {
                       modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                           self.ctx, self.solver.solver_config, self.solver.storage);
                       return render.compute_slice(name, positions, std::nullopt).copy_to_stdvec();
                   }

                   throw shambase::make_except_with_loc<std::runtime_error>("unknown field type");
               },
               py::arg("name"),
               py::arg("field_type"),
               py::arg("positions"),
               py::arg("custom_getter") = std::nullopt)
            .def(
                "render_column_integ",
                [](T &self,
                   const std::string &name,
                   const std::string &field_type,
                   const std::vector<shammath::Ray<Tvec>> &rays,
                   const std::optional<custom_getter_t> &custom_getter)
                    -> std::variant<std::vector<f64>, std::vector<f64_3>> {
                    if (custom_getter.has_value()) {
                        if (!(name == "custom" && field_type == "f64")) {
                            throw shambase::make_except_with_loc<std::invalid_argument>(
                                "custom_getter only available for name=custom and field_type=f64");
                        }
                    }

                    if (field_type == "f64") {
                        modules::CartesianRender<Tvec, f64, SPHKernel> render(
                            self.ctx, self.solver.solver_config, self.solver.storage);
                        return render.compute_column_integ(name, rays, custom_getter)
                            .copy_to_stdvec();
                    }

                    if (field_type == "f64_3") {
                        modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                            self.ctx, self.solver.solver_config, self.solver.storage);
                        return render.compute_column_integ(name, rays, std::nullopt)
                            .copy_to_stdvec();
                    }

                    throw shambase::make_except_with_loc<std::runtime_error>("unknown field type");
                },
                py::arg("name"),
                py::arg("field_type"),
                py::arg("rays"),
                py::arg("custom_getter") = std::nullopt)
            .def(
                "compute_field",
                [](T &self,
                   const std::string &name,
                   const std::string &field_type,
                   const std::optional<custom_getter_t> &custom_getter)
                    -> std::variant<
                        shamrock::solvergraph::Field<f64>,
                        shamrock::solvergraph::Field<f64_3>> {
                    if (custom_getter.has_value()) {
                        if (!(name == "custom" && field_type == "f64")) {
                            throw shambase::make_except_with_loc<std::invalid_argument>(
                                "custom_getter only available for name=custom and field_type=f64");
                        }
                    }

                    if (field_type == "f64") {
                        modules::RenderFieldGetter<Tvec, f64, SPHKernel> render_field_getter(
                            self.ctx, self.solver.solver_config, self.solver.storage);
                        return render_field_getter.build_field(name, custom_getter);
                    }

                    if (field_type == "f64_3") {
                        modules::RenderFieldGetter<Tvec, f64_3, SPHKernel> render_field_getter(
                            self.ctx, self.solver.solver_config, self.solver.storage);
                        return render_field_getter.build_field(name, custom_getter);
                    }

                    throw shambase::make_except_with_loc<std::runtime_error>("unknown field type");
                },
                py::arg("name"),
                py::arg("field_type"),
                py::arg("custom_getter") = std::nullopt)
            .def(
                "render_azymuthal_integ",
                [](T &self,
                   const std::string &name,
                   const std::string &field_type,
                   const std::vector<shammath::RingRay<Tvec>> &ring_rays,
                   const std::optional<custom_getter_t> &custom_getter)
                    -> std::variant<std::vector<f64>, std::vector<f64_3>> {
                    if (custom_getter.has_value()) {
                        if (!(name == "custom" && field_type == "f64")) {
                            throw shambase::make_except_with_loc<std::invalid_argument>(
                                "custom_getter only available for name=custom and field_type=f64");
                        }
                    }

                    if (field_type == "f64") {
                        modules::CartesianRender<Tvec, f64, SPHKernel> render(
                            self.ctx, self.solver.solver_config, self.solver.storage);
                        return render.compute_azymuthal_integ(name, ring_rays, custom_getter)
                            .copy_to_stdvec();
                    }

                    if (field_type == "f64_3") {
                        modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                            self.ctx, self.solver.solver_config, self.solver.storage);
                        return render.compute_azymuthal_integ(name, ring_rays, std::nullopt)
                            .copy_to_stdvec();
                    }

                    throw shambase::make_except_with_loc<std::runtime_error>("unknown field type");
                },
                py::arg("name"),
                py::arg("field_type"),
                py::arg("ring_rays"),
                py::arg("custom_getter") = std::nullopt)
            .def(
                "render_cartesian_slice",
                [](T &self,
                   const std::string &name,
                   const std::string &field_type,
                   Tvec center,
                   Tvec delta_x,
                   Tvec delta_y,
                   u32 nx,
                   u32 ny,
                   const std::optional<custom_getter_t> &custom_getter)
                    -> std::variant<py::array_t<Tscal>> {
                    if (custom_getter.has_value()) {
                        if (!(name == "custom" && field_type == "f64")) {
                            throw shambase::make_except_with_loc<std::invalid_argument>(
                                "custom_getter only available for name=custom and field_type=f64");
                        }
                    }

                    if (field_type == "f64") {
                        py::array_t<Tscal> ret({ny, nx});

                        modules::CartesianRender<Tvec, f64, SPHKernel> render(
                            self.ctx, self.solver.solver_config, self.solver.storage);

                        std::vector<f64> slice
                            = render
                                  .compute_slice(
                                      name, center, delta_x, delta_y, nx, ny, custom_getter)
                                  .copy_to_stdvec();

                        for (u32 iy = 0; iy < ny; iy++) {
                            for (u32 ix = 0; ix < nx; ix++) {
                                ret.mutable_at(iy, ix) = slice[ix + nx * iy];
                            }
                        }

                        return ret;
                    }

                    if (field_type == "f64_3") {
                        py::array_t<Tscal> ret({ny, nx, 3_u32});

                        modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                            self.ctx, self.solver.solver_config, self.solver.storage);

                        std::vector<f64_3> slice
                            = render
                                  .compute_slice(
                                      name, center, delta_x, delta_y, nx, ny, std::nullopt)
                                  .copy_to_stdvec();

                        for (u32 iy = 0; iy < ny; iy++) {
                            for (u32 ix = 0; ix < nx; ix++) {
                                ret.mutable_at(iy, ix, 0) = slice[ix + nx * iy][0];
                                ret.mutable_at(iy, ix, 1) = slice[ix + nx * iy][1];
                                ret.mutable_at(iy, ix, 2) = slice[ix + nx * iy][2];
                            }
                        }

                        return ret;
                    }

                    shambase::throw_with_loc<std::runtime_error>("unknown field type");
                    return py::array_t<Tscal>({nx, ny});
                },
                py::arg("name"),
                py::arg("field_type"),
                py::arg("center"),
                py::arg("delta_x"),
                py::arg("delta_y"),
                py::arg("nx"),
                py::arg("ny"),
                py::arg("custom_getter") = std::nullopt)
            .def(
                "render_cartesian_column_integ",
                [](T &self,
                   const std::string &name,
                   const std::string &field_type,
                   Tvec center,
                   Tvec delta_x,
                   Tvec delta_y,
                   u32 nx,
                   u32 ny,
                   const std::optional<custom_getter_t> &custom_getter)
                    -> std::variant<py::array_t<Tscal>> {
                    if (custom_getter.has_value()) {
                        if (!(name == "custom" && field_type == "f64")) {
                            throw shambase::make_except_with_loc<std::invalid_argument>(
                                "custom_getter only available for name=custom and field_type=f64");
                        }
                    }

                    if (field_type == "f64") {
                        py::array_t<Tscal> ret({ny, nx});

                        modules::CartesianRender<Tvec, f64, SPHKernel> render(
                            self.ctx, self.solver.solver_config, self.solver.storage);

                        std::vector<f64> slice
                            = render
                                  .compute_column_integ(
                                      name, center, delta_x, delta_y, nx, ny, custom_getter)
                                  .copy_to_stdvec();

                        for (u32 iy = 0; iy < ny; iy++) {
                            for (u32 ix = 0; ix < nx; ix++) {
                                ret.mutable_at(iy, ix) = slice[ix + nx * iy];
                            }
                        }

                        return ret;
                    }

                    if (field_type == "f64_3") {
                        py::array_t<Tscal> ret({ny, nx, 3_u32});

                        modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                            self.ctx, self.solver.solver_config, self.solver.storage);

                        std::vector<f64_3> slice
                            = render
                                  .compute_column_integ(
                                      name, center, delta_x, delta_y, nx, ny, std::nullopt)
                                  .copy_to_stdvec();

                        for (u32 iy = 0; iy < ny; iy++) {
                            for (u32 ix = 0; ix < nx; ix++) {
                                ret.mutable_at(iy, ix, 0) = slice[ix + nx * iy][0];
                                ret.mutable_at(iy, ix, 1) = slice[ix + nx * iy][1];
                                ret.mutable_at(iy, ix, 2) = slice[ix + nx * iy][2];
                            }
                        }

                        return ret;
                    }

                    shambase::throw_with_loc<std::runtime_error>("unknown field type");
                    return py::array_t<Tscal>({nx, ny});
                },
                py::arg("name"),
                py::arg("field_type"),
                py::arg("center"),
                py::arg("delta_x"),
                py::arg("delta_y"),
                py::arg("nx"),
                py::arg("ny"),
                py::arg("custom_getter") = std::nullopt);
    }

} // namespace shammodels::sph::pysph

SHAMROCK_SPH_PYBIND_INSTANTIATE(shammodels::sph::pysph::add_render)
