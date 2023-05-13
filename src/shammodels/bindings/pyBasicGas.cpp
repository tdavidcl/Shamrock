// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shambindings/pybindaliases.hpp"
#include "shammodels/BasicGas.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

Register_pymod(pybasicgassph){

    using namespace shammodels::sph;

    py::class_<BasicGas>(m, "BasicGasSPH")
        .def(
            py::init([](ShamrockCtx & ctx) {
                return std::make_unique<BasicGas>(ctx);
            })
        ) 
        .def("setup_fields",&BasicGas::setup_fields)
        .def("evolve",[](BasicGas & self, f64 dt, bool do_dump, std::string dump_name, bool debug_dump){
            self.evolve(dt, BasicGas::DumpOption{
                do_dump,
                dump_name,
                debug_dump
            });
        })
    ;
}