cmake_minimum_required(VERSION 3.9)

project(Shammath CXX C)

set(Sources
    src/CoordRangeTransform.cpp
    src/CoordRange.cpp
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shammath target (Shared lib)")
add_library(shammath SHARED ${Sources})
else()
message("-- adding shammath target (object lib)")
add_library(shammath OBJECT ${Sources})
endif()

add_library(shammath::shammath ALIAS shammath)

target_link_libraries(shammath PUBLIC shambackends)

target_include_directories(shammath PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
