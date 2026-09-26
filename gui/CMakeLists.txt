# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
# ~~~

# Standalone Shamrock control GUI (Dear ImGui docking + GLFW + OpenGL 3).
# Not linked against any Shamrock library yet: it runs on demo data only.
# Enabled from the main project with -DSHAMROCK_BUILD_GUI=ON, or built on its own with
#   cmake -S gui -B build-gui && cmake --build build-gui

cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(shamrock_gui CXX C)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release)
    endif()
endif()

message("-- adding shamrock_gui target")

include(FetchContent)
# Docking build of the Dear ImGui version used by imgui-bundle 1.92.900.
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.9-docking
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    imgui_color_text_edit
    GIT_REPOSITORY https://github.com/goossens/ImGuiColorTextEdit.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui imgui_color_text_edit stb)

# GLFW: use the system package when present, otherwise build it.
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(glfw)
endif()
find_package(OpenGL REQUIRED)

# FreeType rasterises glyphs the same way as the imgui-bundle wheel.
# Without it Dear ImGui falls back to its built-in stb_truetype.
option(SHAMROCK_GUI_FREETYPE "Use FreeType for font rasterisation in shamrock_gui" ON)
if(SHAMROCK_GUI_FREETYPE)
    find_package(Freetype)
endif()

add_library(
    shamrock_gui_imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    ${imgui_color_text_edit_SOURCE_DIR}/TextEditor.cpp
)
target_include_directories(
    shamrock_gui_imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends
                              ${imgui_color_text_edit_SOURCE_DIR}
)
target_link_libraries(shamrock_gui_imgui PUBLIC glfw OpenGL::GL)
if(SHAMROCK_GUI_FREETYPE AND FREETYPE_FOUND)
    target_sources(shamrock_gui_imgui PRIVATE ${imgui_SOURCE_DIR}/misc/freetype/imgui_freetype.cpp)
    target_compile_definitions(shamrock_gui_imgui PUBLIC IMGUI_ENABLE_FREETYPE)
    target_link_libraries(shamrock_gui_imgui PUBLIC Freetype::Freetype)
    message(STATUS "shamrock_gui fonts: FreeType")
else()
    message(STATUS "shamrock_gui fonts: stb_truetype")
endif()

add_executable(shamrock_gui src/main.cpp)
target_include_directories(shamrock_gui PRIVATE ${stb_SOURCE_DIR})
target_link_libraries(shamrock_gui PRIVATE shamrock_gui_imgui)
if(MSVC)
    target_compile_options(shamrock_gui PRIVATE /utf-8)
    target_compile_options(shamrock_gui_imgui PRIVATE /utf-8)
endif()

# Fonts and logo next to the executable (the app also looks in ./assets).
add_custom_command(
    TARGET shamrock_gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets
            $<TARGET_FILE_DIR:shamrock_gui>/assets
)
