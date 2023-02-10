// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//



#include "numeric_tests.hpp"
#include "shamalgs/numeric/details/exclusiveScanGPUGems39.hpp"


TestStart(Unittest, "shamalgs/numeric/details/exclusive_sum_gpugems39", test_exclusive_sum_gpugems39_1, 1){
    
    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::details::exclusive_sum_gpugems39_1);
    test.check();
}
