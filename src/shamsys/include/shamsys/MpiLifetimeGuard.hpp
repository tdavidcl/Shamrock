// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file MpiLifetimeGuard.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief RAII guard of the MPI library lifetime
 */

#include <exception>

namespace shamsys::instance {

    /**
     * @brief RAII guard of the MPI library lifetime
     *
     * The constructor calls `MPI_Init`, unless MPI was already initialized by someone else (e.g.
     * mpi4py), in which case the guard does not own MPI and leaves its lifetime alone.
     *
     * If the guard owns MPI, on destruction (if MPI was not finalized in the meantime):
     *  - if an exception is in flight (the guard is destroyed during stack unwinding), the other
     *    ranks cannot be expected to reach `MPI_Finalize`, so `MPI_Abort` is called to tear down
     *    the whole job instead of leaving it hanging.
     *  - otherwise `MPI_Finalize` is called.
     *
     * An exception escaping `main` does not necessarily unwind the stack (it is implementation
     * defined), in which case `std::terminate` is called without running any destructor. To cover
     * that case an owning guard also installs, for its own lifetime, a terminate handler that
     * prints the exception and calls `MPI_Abort`.
     *
     * Usage:
     * @code{.cpp}
     * std::unique_ptr<MpiLifetimeGuard> mpi_guard;
     * mpi_guard = std::make_unique<MpiLifetimeGuard>(&argc, &argv); // MPI_Init
     * ... do stuff ...
     * mpi_guard.reset(); // MPI_Finalize (or MPI_Abort if an exception is in flight)
     * @endcode
     */
    class MpiLifetimeGuard {
        public:
        /**
         * @brief Initialize MPI if it is not already initialized
         *
         * @param argc pointer to the number of arguments, forwarded to `MPI_Init`
         * @param argv pointer to the argument vector, forwarded to `MPI_Init`
         */
        MpiLifetimeGuard(int *argc, char ***argv);
        ~MpiLifetimeGuard() noexcept;

        MpiLifetimeGuard(const MpiLifetimeGuard &)            = delete;
        MpiLifetimeGuard &operator=(const MpiLifetimeGuard &) = delete;
        MpiLifetimeGuard(MpiLifetimeGuard &&)                 = delete;
        MpiLifetimeGuard &operator=(MpiLifetimeGuard &&)      = delete;

        private:
        /// true if the guard called `MPI_Init`
        bool owns_mpi = false;

        /// Number of uncaught exceptions when the guard was created
        int uncaught_exceptions_at_ctor;

        /// Terminate handler that was active before the guard installed its own
        std::terminate_handler previous_terminate_handler = nullptr;
    };

} // namespace shamsys::instance
