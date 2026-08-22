// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file pySPHModelBindings.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Declarations for the split SPH Python binding translation units.
 */

#include "shambindings/pybindaliases.hpp"
#include "shammath/sphkernels.hpp"
#include <string>

namespace shammodels::sph::pysph {

    template<class Tvec, template<class> class SPHKernel>
    void add_config(py::module &m, const std::string &name_config);

    template<class Tvec, template<class> class SPHKernel>
    void add_setup(py::module &m, const std::string &name_model);

    template<class Tvec, template<class> class SPHKernel>
    void add_model(py::module &m, const std::string &name_model);

    template<class Tvec, template<class> class SPHKernel>
    void add_render(py::module &m, const std::string &name_model);

    template<class Tvec, template<class> class SPHKernel>
    void add_analysis(py::module &m, const std::string &name_model);

    void register_sph_analysis_python(py::module &msph);

    template<class T>
    inline py::class_<T> registered_class() {
        return py::reinterpret_borrow<py::class_<T>>(py::type::of<T>());
    }

} // namespace shammodels::sph::pysph

#define SHAMROCK_SPH_PYBIND_INSTANTIATE(fn)                                                        \
    template void fn<f64_3, shammath::M4>(py::module &, const std::string &);                      \
    template void fn<f64_3, shammath::M6>(py::module &, const std::string &);                      \
    template void fn<f64_3, shammath::M8>(py::module &, const std::string &);                      \
    template void fn<f64_3, shammath::C2>(py::module &, const std::string &);                      \
    template void fn<f64_3, shammath::C4>(py::module &, const std::string &);                      \
    template void fn<f64_3, shammath::C6>(py::module &, const std::string &);
