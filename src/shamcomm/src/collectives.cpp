// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file collectives.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/mpi.hpp"
#include "shamcomm/mpiErrorCheck.hpp"
#include "shamcomm/worldInfo.hpp"
#include <unordered_map>

void shamcomm::gather_str(const std::string &send_vec, std::string &recv_vec) {
    StackEntry stack_loc{};

    u32 local_count = send_vec.size();

    // querry global size and resize the receiving vector
    u32 global_len;
    MPICHECK(MPI_Allreduce(&local_count, &global_len, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    recv_vec.resize(global_len);

    // int *table_data_count = new int[shamcomm::world_size()];
    std::vector<int> table_data_count{shamcomm::world_size()};

    MPICHECK(
        MPI_Allgather(&local_count, 1, MPI_INT, &table_data_count[0], 1, MPI_INT, MPI_COMM_WORLD));

    // printf("table_data_count =
    // [%d,%d,%d,%d]\n",table_data_count[0],table_data_count[1],table_data_count[2],table_data_count[3]);

    // int *node_displacments_data_table = new int[shamcomm::world_size()];
    std::vector<int> node_displacments_data_table{shamcomm::world_size()};

    node_displacments_data_table[0] = 0;

    for (u32 i = 1; i < shamcomm::world_size(); i++) {
        node_displacments_data_table[i]
            = node_displacments_data_table[i - 1] + table_data_count[i - 1];
    }

    // printf("node_displacments_data_table =
    // [%d,%d,%d,%d]\n",node_displacments_data_table[0],node_displacments_data_table[1],node_displacments_data_table[2],node_displacments_data_table[3]);

    MPICHECK(MPI_Allgatherv(
        send_vec.data(),
        send_vec.size(),
        MPI_CHAR,
        recv_vec.data(),
        table_data_count.data(),
        node_displacments_data_table.data(),
        MPI_CHAR,
        MPI_COMM_WORLD));

    // delete[] table_data_count;
    // delete[] node_displacments_data_table;
}

void shamcomm::gather_basic_str(
    const std::basic_string<byte> &send_vec, std::basic_string<byte> &recv_vec) {

    std::basic_string<byte> out_res_string;

    if (shamcomm::world_size() == 1) {
        out_res_string = send_vec;
    } else {
        std::basic_string<byte> loc_string = send_vec;

        // int *counts   = new int[shamcomm::world_size()];
        std::vector<int> counts{shamcomm::world_size()};
        int nelements = (int) loc_string.size();
        // Each process tells the root how many elements it holds
        MPICHECK(MPI_Gather(&nelements, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD));

        // Displacements in the receive buffer for MPI_GATHERV
        // int *disps = new int[shamcomm::world_size()];
        std::vector<int> disps{shamcomm::world_size()};

        // Displacement for the first chunk of data - 0
        for (int i = 0; i < shamcomm::world_size(); i++)
            disps[i] = (i > 0) ? (disps[i - 1] + counts[i - 1]) : 0;

        // Place to hold the gathered data
        // Allocate at root only
        byte *gather_data = NULL;
        if (shamcomm::world_rank() == 0)
            // disps[size-1]+counts[size-1] == total number of elements
            gather_data
                = new byte[disps[shamcomm::world_size() - 1] + counts[shamcomm::world_size() - 1]];

        // Collect everything into the root
        MPICHECK(MPI_Gatherv(
            loc_string.c_str(),
            nelements,
            MPI_CHAR,
            gather_data,
            counts.data(),
            disps.data(),
            MPI_CHAR,
            0,
            MPI_COMM_WORLD));

        if (shamcomm::world_rank() == 0) {
            out_res_string = std::basic_string<byte>(
                gather_data,
                disps[shamcomm::world_size() - 1] + counts[shamcomm::world_size() - 1]);
        }

        // delete[] counts;
        // delete[] disps;
    }

    recv_vec = out_res_string;
}

std::unordered_map<std::string, int>
shamcomm::string_histogram(const std::vector<std::string> &inputs, std::string delimiter) {
    std::string accum_loc = "";
    for (auto &s : inputs) {
        accum_loc += s + delimiter;
    }

    std::string recv = "";
    gather_str(accum_loc, recv);

    if (world_rank() == 0) {

        std::vector<std::string> splitted = shambase::split_str(recv, delimiter);

        std::unordered_map<std::string, int> histogram;

        for (size_t i = 0; i < splitted.size(); i++) {
            histogram[splitted[i]] += 1;
        }

        return histogram;
    }

    return {};
}
