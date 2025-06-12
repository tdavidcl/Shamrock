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

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include "shamcomm/mpi.hpp"
#include <string>

namespace shamcomm::mpi {

    void register_time(std::string timername, f64 time);

    f64 get_timer(std::string timername);

    void Allreduce(
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm);

    void Allgather(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm);

    void Allgatherv(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        const int recvcounts[],
        const int displs[],
        MPI_Datatype recvtype,
        MPI_Comm comm);

    void Isend(
        const void *buf,
        int count,
        MPI_Datatype datatype,
        int dest,
        int tag,
        MPI_Comm comm,
        MPI_Request *request);

    void Irecv(
        void *buf,
        int count,
        MPI_Datatype datatype,
        int source,
        int tag,
        MPI_Comm comm,
        MPI_Request *request);

    void Exscan(
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm);

    void Wait(MPI_Request *request, MPI_Status *status);

    void Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses);

    void Barrier(MPI_Comm comm);

    void Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);

    void Recv(
        void *buf,
        int count,
        MPI_Datatype datatype,
        int source,
        int tag,
        MPI_Comm comm,
        MPI_Status *status);

    void Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);

    void Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);

    void File_set_view(
        MPI_File fh,
        MPI_Offset disp,
        MPI_Datatype etype,
        MPI_Datatype filetype,
        const char *datarep,
        MPI_Info info);

    void Type_size(MPI_Datatype type, int *size);

    void File_write_all(
        MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);

    void
    File_write(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);

    void File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);

    void File_write_at(
        MPI_File fh,
        MPI_Offset offset,
        const void *buf,
        int count,
        MPI_Datatype datatype,
        MPI_Status *status);

    void File_read_at(
        MPI_File fh,
        MPI_Offset offset,
        void *buf,
        int count,
        MPI_Datatype datatype,
        MPI_Status *status);

    void File_close(MPI_File *fh);
    void File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh);

    void Test(MPI_Request *request, int *flag, MPI_Status *status);
} // namespace shamcomm::mpi
