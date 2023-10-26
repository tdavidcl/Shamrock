cmake_minimum_required(VERSION 3.9)

# set the project name
project(coala)

# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Debug)
# endif()

# if (APPLE)
#     set(CMAKE_CXX_COMPILER /usr/local/bin/g++-10)
# endif()
# message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")


#LIST(APPEND CMAKE_SYSTEM_PROGRAM_PATH  "${DPCPP_INSTALL_LOC}/bin")
#LIST(APPEND CMAKE_SYSTEM_LIBRARY_PATH  "${DPCPP_INSTALL_LOC}/bin/lib")
#set(CMAKE_CXX_COMPILER "${DPCPP_INSTALL_LOC}/bin/clang++")

#LIST(APPEND CMAKE_SYSTEM_PROGRAM_PATH  "../../bin")
LIST(APPEND CMAKE_SYSTEM_PROGRAM_PATH  "../../sycl_cpl/hipSYCL/bin")
LIST(APPEND CMAKE_SYSTEM_LIBRARY_PATH  "../../sycl_cpl/hipSYCL/lib")


#choose sycl backend

#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_COMPILER "../../sycl_cpl/hipSYCL/bin/syclcc")
set(CMAKE_CXX_FLAGS "-std=c++17 -pedantic-errors -fcolor-diagnostics -fsycl")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-unknown-nvidiacl ")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl ")



set(CMAKE_CXX_FLAGS_DEBUG "-g  -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra ")

message("CMAKE_CXX_FLAGS is ${CMAKE_CXX_FLAGS}")


add_executable(
        coala
        setup.cpp
        math_functions.cpp
        kernel_approx/flux/flux_components.cpp
        kernel_approx/flux/flux_function.cpp
        kernel_approx/intflux/termA/termA_intflux_components.cpp
        kernel_approx/intflux/termA/termA_intflux_function.cpp
        kernel_approx/intflux/termB/termB_intflux_components.cpp
        kernel_approx/intflux/termB/termB_intflux_function.cpp
        init_grid.cpp
        polynomials_legendre.cpp
        reconstruction_gh.cpp
        limiter.cpp
        GQ_legendre_nodes_weights.cpp
        L2proj_GQ.cpp
        generate_tabflux_tabintflux.cpp
        generate_flux_intflux.cpp
        solver_DG.cpp 
        generate_gij.cpp
        compute_coag.cpp
        iterate_coag.cpp
        coala.cpp
)

        
        





