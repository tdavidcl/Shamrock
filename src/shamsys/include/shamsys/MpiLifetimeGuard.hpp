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
 * @brief RAII guard tying the MPI library lifetime to a scope
 */

#include <exception>

namespace shamsys::instance {

    /**
     * @brief RAII guard of the MPI library lifetime
     *
     * On destruction, if MPI is initialized and not yet finalized:
     *  - if an exception is in flight (the guard is destroyed during stack unwinding), the other
     *    ranks cannot be expected to reach `MPI_Finalize`, so `MPI_Abort` is called to tear down
     *    the whole job instead of leaving it hanging.
     *  - otherwise MPI is finalized through `shamsys::instance::close_mpi()`.
     *
     * If MPI was never started, or was already finalized explicitly (e.g. by
     * `shamsys::instance::close()`), the destructor does nothing.
     *
     * An exception escaping `main` does not necessarily unwind the stack (it is implementation
     * defined), in which case `std::terminate` is called without running this destructor. To cover
     * that case the guard also installs, for its own lifetime, a terminate handler that prints the
     * exception and calls `MPI_Abort` while MPI is alive.
     *
     * Usage:
     * @code{.cpp}
     * int main(int argc, char *argv[]) {
     *     shamsys::instance::MpiLifetimeGuard mpi_guard;
     *     shamsys::instance::init(argc, argv);
     *     ... do stuff ...
     * } // MPI_Finalize (or MPI_Abort if an exception is in flight)
     * @endcode
     */
    class MpiLifetimeGuard {
        public:
        MpiLifetimeGuard();
        ~MpiLifetimeGuard() noexcept;

        MpiLifetimeGuard(const MpiLifetimeGuard &)            = delete;
        MpiLifetimeGuard &operator=(const MpiLifetimeGuard &) = delete;
        MpiLifetimeGuard(MpiLifetimeGuard &&)                 = delete;
        MpiLifetimeGuard &operator=(MpiLifetimeGuard &&)      = delete;

        private:
        /// Number of uncaught exceptions when the guard was created
        int uncaught_exceptions_at_ctor;

        /// Terminate handler that was active before the guard installed its own
        std::terminate_handler previous_terminate_handler;
    };

} // namespace shamsys::instance
