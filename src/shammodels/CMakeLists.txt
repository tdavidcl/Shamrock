
#add_subdirectory(algs)
#add_subdirectory(physics)
#add_subdirectory(setup)
add_subdirectory(generic)
add_subdirectory(sph)
add_subdirectory(nbody)
add_subdirectory(amr)

set(SPH_ENABLED_KERNEL "M4;M6;M8" CACHE STRING "Enabled SPH kernels")
set(SPH_ENABLED_VEC_TYPE "f64" CACHE STRING "Enabled position types" )
set(SPH_ENABLED_DIM "3" CACHE STRING "Enabled dimensions" )


set(XMACRO_SRC "// XMACRO generated with cmake\n")

set(XMACRO_SRC "${XMACRO_SRC}#define XMAC_SPH_KERNEL ")
foreach(elem2 ${SPH_ENABLED_KERNEL})
    set(XMACRO_SRC "${XMACRO_SRC} X(${elem2})")
endforeach()

set(XMACRO_SRC "${XMACRO_SRC}\n")

set(XMACRO_SRC "${XMACRO_SRC}#define XMAC_SPH_VECTYPE_KERNEL ")
foreach(elem1 ${SPH_ENABLED_VEC_TYPE})
foreach(dim ${SPH_ENABLED_DIM})
foreach(elem2 ${SPH_ENABLED_KERNEL})
    set(XMACRO_SRC "${XMACRO_SRC} X(${elem1}_${dim},${elem2})")
endforeach()
endforeach()
endforeach()

set(XMACRO_SRC "${XMACRO_SRC}\n")

set(XMACRO_SRC "${XMACRO_SRC}#define XMAC_SPH_BASE_VECTYPE_KERNEL ")
foreach(elem1 ${SPH_ENABLED_VEC_TYPE})
foreach(dim ${SPH_ENABLED_DIM})
foreach(elem2 ${SPH_ENABLED_KERNEL})
    set(XMACRO_SRC "${XMACRO_SRC} X(${elem1},${elem1}_${dim},${elem2})")
endforeach()
endforeach()
endforeach()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/include/_sph_xmacros.hpp "${XMACRO_SRC}")
message("-- XMACRO SPH file : ${CMAKE_CURRENT_BINARY_DIR}/include/_sph_xmacros.hpp")

target_include_directories(shamrock PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)
target_include_directories(shamrock_lib PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)
target_include_directories(shamrock_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)
