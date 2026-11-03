// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ExchangeGhostLayer.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of ghost layer data exchange for distributed hydrodynamics simulations
 *
 * This file implements the ExchangeGhostLayer class methods, providing the concrete
 * functionality for exchanging ghost layer data between distributed computational
 * domains using sparse communication and serialization mechanisms.
 */

#include "shamrock/solvergraph/ExchangeGhostLayer.hpp"
#include "shamalgs/collective/distributedDataComm.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamrock/solvergraph/RankGetter.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"

void shamrock::solvergraph::ExchangeGhostLayer::_impl_evaluate_internal() {
    __shamrock_stack_entry();
    auto edges = get_edges();

    // outputs
    auto &ghost_layer                                   = edges.ghost_layer;
    const shamrock::solvergraph::RankGetter &rank_owner = edges.rank_owner;

    // exchange in place, the layers held by the edge are reused to receive the data (see
    // serialize_sparse_comm_inplace) instead of being reallocated at every exchange
    shamalgs::collective::serialize_sparse_comm_inplace<shamrock::patch::PatchDataLayer>(
        shamsys::instance::get_compute_scheduler_ptr(),
        ghost_layer.patchdatas,
        [&](u64 id) {
            return rank_owner.get_rank_owner(id);
        },
        [](shamrock::patch::PatchDataLayer &pdat) {
            shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
            ser.allocate(pdat.serialize_buf_byte_size());
            pdat.serialize_buf(ser);
            return ser.finalize();
        },
        [&](sham::DeviceBuffer<u8> &&buf, shamrock::patch::PatchDataLayer &pdat) {
            // exchange the buffer held by the distrib data and give it to the serializer
            shamalgs::SerializeHelper ser(
                shamsys::instance::get_compute_scheduler_ptr(),
                std::forward<sham::DeviceBuffer<u8>>(buf));

            // a recycled layer may come with another layout
            if (pdat.get_layout_ptr().get() != ghost_layer_layout.get()) {
                pdat = shamrock::patch::PatchDataLayer(ghost_layer_layout);
            }
            pdat.deserialize_buf_inplace(ser);
        },
        [&]() {
            return shamrock::patch::PatchDataLayer(ghost_layer_layout);
        },
        cache);
}

std::string shamrock::solvergraph::ExchangeGhostLayer::_impl_get_tex() const {
    std::string tex = R"tex(
        Exchange ghost layer data between distributed processes

        \begin{align}
        {ghost_layer}_{i \rightarrow \underline{j}} = \text{Sparse comm}({ghost_layer}_{\underline{i} \rightarrow j}) \\
        \text{where } {rank_owner}_{\underline{j}} = \text{MPI world rank} \\
        \text{and } {rank_owner}_{\underline{i}} = \text{MPI world rank} \\
        \text{and } i \in [0, N_{\rm patch}] \\
        \text{and } j \in [0, N_{\rm patch}] \\
        \end{align},
        underlined indices denotes one that currently owned by the local process.
    )tex";

    replace_edges_tex_symbols(tex);

    return tex;
}
