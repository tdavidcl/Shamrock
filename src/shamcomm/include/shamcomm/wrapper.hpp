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
 * @file wrapper.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambase/time.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include <array>

namespace shamcomm::mpi {

    namespace timers {
        enum TimersType { Isend = 0, Irecv = 1, Wait = 2, Allreduce = 3, Allgather = 4 };

        inline f64 total_time            = 0;
        inline std::array<f64, 6> timers = {};

        void register_time_entry(TimersType type, f64 time) {
            total_time += time;
            timers[type] += time;
        }
    } // namespace timers

    inline void check_tag_value(i32 tag) {
        if (tag > mpi_max_tag_value()) {
            shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                "mpi_max_tag_value ({}) exceeded with tag {}", mpi_max_tag_value(), tag));
        }
    }

    inline void Isend(
        const void *buf,
        int count,
        MPI_Datatype datatype,
        int dest,
        int tag,
        MPI_Comm comm,
        MPI_Request *request) {
        StackEntry stack_loc{};

        check_tag_value(tag);

        f64 tstart = shambase::details::get_wtime();
        MPICHECK(MPI_Isend(buf, count, datatype, dest, tag, comm, request));
        timers::register_time_entry(timers::Isend, shambase::details::get_wtime() - tstart);
    }

    inline void Irecv(
        void *buf,
        int count,
        MPI_Datatype datatype,
        int source,
        int tag,
        MPI_Comm comm,
        MPI_Request *request) {

        check_tag_value(tag);

        f64 tstart = shambase::details::get_wtime();
        MPICHECK(MPI_Irecv(buf, count, datatype, source, tag, comm, request));
        timers::register_time_entry(timers::Irecv, shambase::details::get_wtime() - tstart);
    }

    inline void Allreduce(
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm) {

        f64 tstart = shambase::details::get_wtime();
        MPICHECK(MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm));
        timers::register_time_entry(timers::Allreduce, shambase::details::get_wtime() - tstart);
    }

    inline void Allgather(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm) {

        f64 tstart = shambase::details::get_wtime();
        int ret = MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        timers::register_time_entry(timers::Allgather, shambase::details::get_wtime() - tstart);
    }

} // namespace shamcomm::mpi
