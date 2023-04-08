// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambindings/pybindaliases.hpp"
#include "shamrock/physics/units/UnitSystem.hpp"


Register_pymod(pyunits_init) {

    using UnitSystem = shamrock::UnitSystem<f64>;


    py::class_<UnitSystem>(m, "UnitSystem")
        .def(py::init([](f64 unit_time,
                         f64 unit_lenght,
                         f64 unit_mass,
                         f64 unit_current,
                         f64 unit_temperature,
                         f64 unit_qte,
                         f64 unit_lumint) {
            return std::make_unique<UnitSystem>(unit_time,
                                                unit_lenght,
                                                unit_mass,
                                                unit_current,
                                                unit_temperature,
                                                unit_qte,
                                                unit_lumint);
        }))
        .def("get",
            [](UnitSystem & self, std::string prefix, std::string name, i32 power){
            
            
        });
}