// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file start_python.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include <string>

namespace shambindings {

#if defined(DOXYGEN) || defined(SHAMROCK_EXECUTABLE_BUILD)

    /// set the value of sys.path before init
    void setpypath(std::string path);

    /// set the value of sys.path before init from the supplied binary
    void setpypath_from_binary(std::string binary_path);

    /**
     * @brief Start shamrock embded ipython interpreter
     *
     * This function is available only if the flag SHAMROCK_EXECUTABLE_BUILD is set
     *
     * @warning This function shall not be called if more than one processes are running
     * @param do_print print log at python startup
     */
    void start_ipython(bool do_print);

    /**
     * @brief run python runscript
     *
     * This function is available only if the flag SHAMROCK_EXECUTABLE_BUILD is set
     *
     * @param do_print print log at python startup
     * @param file_path path to the runscript
     */
    void run_py_file(std::string file_path, bool do_print);

    /// Modify Python sys.path to point to one detected during cmake invocation
    void modify_py_sys_path();
#endif

} // namespace shambindings
