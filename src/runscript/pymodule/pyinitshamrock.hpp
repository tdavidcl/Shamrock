// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "models/sph/models/basic_sph_gas.hpp"
#include "pylib.hpp"

#include "pymethods.hpp"
#include "pyshamrockcontext.hpp"

static PyModuleDef Pyshamrock_Module = {
    PyModuleDef_HEAD_INIT, "shamrock", NULL, -1, Pyshamrock_Methods,
    NULL, NULL, NULL, NULL
};

static PyObject* PyInit_shamrock(void) {


    PyObject *m;
    

    m = PyModule_Create(&Pyshamrock_Module);
    if (m == NULL)
        return NULL;

    for(auto fct : init_python_binding_lst){
        fct(m);
    }

    return m;

}