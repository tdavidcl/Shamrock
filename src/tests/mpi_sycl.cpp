// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <handler.hpp>
#include <usm.hpp>


/*

hipsycl

T* usm = sycl::malloc_device<T>(10000, queue);
auto e = queue.submit([=](sycl::handler & cgh){
    //copy to usm
});

queue_in_order.single_task(e,[=](){
    __hipsycl_if_target_host(
        //mpi
    )
}).wait();

sycl::free(usm,queue);

dpcpp ?

sycl::queue & queue = shamsys::instance::get_compute_queue();

using T = int;

T* usm = sycl::malloc_device<T>(10000, queue);
auto e = queue.submit([=](sycl::handler & cgh){
    //copy to usm
});

queue.submit([&](sycl::handler & cgh){
    cgh.depends_on(e);
    cgh.host_task([](){
        //mpi
    });
}).wait();

sycl::free(usm,queue);

*/

template<class T>
class MPISYCLAwareBuffer{
    T* usm_ptr;

    sycl::queue & queue_copy;
    sycl::queue & queue_mpi;



    MPISYCLAwareBuffer(){

    }

    inline void mpi_op(std::function<void(T*)> fct){
        
    }

};



TestStart(Unittest, "sycl_mpi", testsycl_mpi, 1) {

    

    

}