// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MpiLifetimeGuard.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief RAII guard tying the MPI library lifetime to a scope
 */

#include "shamsys/MpiLifetimeGuard.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamsys/NodeInstance.hpp"
#include <cstdio>
#include <cstdlib>
#include <exception>

namespace {

    /// Terminate handler to chain to when MPI is not alive
    std::terminate_handler chained_terminate_handler = nullptr;

    bool is_mpi_alive() { return shamcomm::is_mpi_initialized() && !shamcomm::is_mpi_finalized(); }

    [[noreturn]] void mpi_abort(const char *reason) {
        std::fprintf(
            stderr,
            "[MpiLifetimeGuard] world rank %d : %s, calling MPI_Abort\n",
            shamcomm::world_rank(),
            reason);
        std::fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        // MPI_Abort is not supposed to return
        std::abort();
    }

    [[noreturn]] void mpi_terminate_handler() {
        if (is_mpi_alive()) {
            if (std::exception_ptr eptr = std::current_exception()) {
                try {
                    std::rethrow_exception(eptr);
                } catch (const std::exception &e) {
                    std::fprintf(stderr, "Uncaught exception : %s\n", e.what());
                } catch (...) {
                    std::fprintf(stderr, "Uncaught exception of unknown type\n");
                }
            }
            mpi_abort("std::terminate called while MPI is alive");
        }

        if (chained_terminate_handler != nullptr) {
            chained_terminate_handler();
        }
        std::abort();
    }

} // namespace

namespace shamsys::instance {

    MpiLifetimeGuard::MpiLifetimeGuard()
        : uncaught_exceptions_at_ctor(std::uncaught_exceptions()),
          previous_terminate_handler(std::set_terminate(mpi_terminate_handler)) {
        chained_terminate_handler = previous_terminate_handler;
    }

    MpiLifetimeGuard::~MpiLifetimeGuard() noexcept {
        std::set_terminate(previous_terminate_handler);

        if (!is_mpi_alive()) {
            return;
        }

        if (std::uncaught_exceptions() > uncaught_exceptions_at_ctor) {
            mpi_abort("exception in flight while destroying the MPI lifetime guard");
        }

        try {
            close_mpi();
        } catch (const std::exception &e) {
            std::fprintf(stderr, "Exception thrown while finalizing MPI : %s\n", e.what());
            mpi_abort("MPI finalization failed");
        } catch (...) {
            mpi_abort("MPI finalization failed");
        }
    }

} // namespace shamsys::instance
