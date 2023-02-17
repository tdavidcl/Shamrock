// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shambindings/pybindaliases.hpp"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <memory>

#include "shammodels/amr/AMROverheadtest.hpp"

Register_pymod(pyamrtestmode){


    py::class_<AMRTestModel>(m, "AMRTestModel")
        .def(py::init([](AMRTestModel::Grid & grd) {
                return std::make_unique<AMRTestModel>(grd);
            }))
        .def("step",
        [](AMRTestModel& obj){
                obj.step();
        });





}