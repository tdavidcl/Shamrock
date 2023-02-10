// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shamalgs/memory/memory.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"


#include "shamalgs/numeric/numeric.hpp"
#include <numeric>

#include "numeric_tests.hpp"

TestStart(Unittest, "shamalgs/numeric/exclusive_sum", exclsumtest, 1){
    
    TestExclScan<u32> test ((TestExclScan<u32>::vFunctionCall)shamalgs::numeric::exclusive_sum);
    test.check();
}



TestStart(Unittest, "shamalgs/numeric/stream_compact", streamcompactalg, 1){
    
    TestStreamCompact test ((TestStreamCompact::vFunctionCall)shamalgs::numeric::stream_compact);
    test.check();

}