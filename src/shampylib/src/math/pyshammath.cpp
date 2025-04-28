// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyshammath.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammath/derivatives.hpp"
#include "shampylib/math/pyAABB.hpp"
#include "shampylib/math/pyRay.hpp"
#include "shampylib/math/pySPHKernels.hpp"
#include "shampylib/math/pySfc.hpp"
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <functional>

Register_pymod(pysham_mathinit) {

    py::module math_module = m.def_submodule("math", "Shamrock math lib");

    shampylib::init_shamrock_math_AABB<f64_3>(math_module, "AABB_f64_3");
    shampylib::init_shamrock_math_Ray<f64_3>(math_module, "Ray_f64_3");
    shampylib::init_shamrock_math_sfc(math_module);
    shampylib::init_shamrock_math_sphkernels(math_module);

    math_module.def("derivative_upwind", [](f64 x, f64 dx, std::function<f64(f64)> fct) {
        return shammath::derivative_upwind<f64>(x, dx, [&](f64 x) {
            return fct(x);
        });
    });
    math_module.def("derivative_centered", [](f64 x, f64 dx, std::function<f64(f64)> fct) {
        return shammath::derivative_centered<f64>(x, dx, [&](f64 x) {
            return fct(x);
        });
    });
    math_module.def("derivative_3point_forward", [](f64 x, f64 dx, std::function<f64(f64)> fct) {
        return shammath::derivative_3point_forward<f64>(x, dx, [&](f64 x) {
            return fct(x);
        });
    });
    math_module.def("derivative_3point_backward", [](f64 x, f64 dx, std::function<f64(f64)> fct) {
        return shammath::derivative_3point_backward<f64>(x, dx, [&](f64 x) {
            return fct(x);
        });
    });
    math_module.def("derivative_5point_midpoint", [](f64 x, f64 dx, std::function<f64(f64)> fct) {
        return shammath::derivative_5point_midpoint<f64>(x, dx, [&](f64 x) {
            return fct(x);
        });
    });
    math_module.def(
        "estim_deriv_step",
        [](u32 order) {
            return shammath::estim_deriv_step<f64>(order);
        },
        R"pbdoc(
    Estim the correct step to use for a given order when using derivatives
    )pbdoc");
}
