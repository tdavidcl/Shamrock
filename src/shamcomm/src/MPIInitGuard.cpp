// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MPIInitGuard.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief RAII wrapper around MPI_Init and MPI_Finalize
 */

#include "shamcomm/MPIInitGuard.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiErrorCheck.hpp"

namespace shamcomm {

    MPIInitGuard::MPIInitGuard(int *argc, char ***argv) {
        int initialized = 0;
        MPICHECK(MPI_Initialized(&initialized));
        if (initialized) {
            owns_mpi = false;
            return;
        }

        MPICHECK(MPI_Init(argc, argv));
        owns_mpi = true;
    }

    MPIInitGuard::~MPIInitGuard() { close(); }

    void MPIInitGuard::close() {
        if (!owns_mpi) {
            return;
        }

        int finalized = 0;
        MPICHECK(MPI_Finalized(&finalized));
        if (!finalized) {
            MPICHECK(MPI_Finalize());
        }

        owns_mpi = false;
    }

    bool MPIInitGuard::is_active() const { return owns_mpi; }

} // namespace shamcomm
