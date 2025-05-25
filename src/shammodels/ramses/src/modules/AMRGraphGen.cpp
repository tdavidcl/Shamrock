// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AMRGraphGen.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shammodels/ramses/modules/AMRGraphGen.hpp"
#include "shammodels/ramses/modules/BlockNeighToCellNeigh.hpp"
#include "shammodels/ramses/modules/FindBlockNeigh.hpp"
#include "shammodels/ramses/modules/details/compute_neigh_graph.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include <utility>

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
// AMR block graph generation
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Tvec, class TgridVec>
shambase::DistributedData<shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>>
shammodels::basegodunov::modules::AMRGraphGen<Tvec, TgridVec>::
    find_AMR_block_graph_links_common_face() {

    using MergedPDat = shamrock::MergedPatchData;
    using RTree      = typename Storage::RTree;

    StackEntry stack_loc{};

    std::shared_ptr<shammodels::basegodunov::solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>
        block_graph_edge = std::make_shared<
            shammodels::basegodunov::solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(
            "block_graph_edge", "block graph edge");

    FindBlockNeigh<Tvec, TgridVec, u_morton> node;
    node.set_edges(
        storage.block_counts_with_ghost,
        storage.refs_block_min,
        storage.refs_block_max,
        storage.trees,
        block_graph_edge);
    node.evaluate();

    return std::move(block_graph_edge->graph);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
// Lowering from block graph to cell graph
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::AMRGraphGen<Tvec, TgridVec>::
    lower_AMR_block_graph_to_cell_common_face_graph(
        shambase::DistributedData<OrientedAMRGraph> &oriented_blocks_graph_links) {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;
    using RTree      = typename Storage::RTree;

    std::shared_ptr<shammodels::basegodunov::solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>
        block_graph_edge = std::make_shared<
            shammodels::basegodunov::solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(
            "block_graph_edge", "block graph edge");

    block_graph_edge->graph = std::move(oriented_blocks_graph_links);

    std::shared_ptr<shammodels::basegodunov::solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>
        cell_graph_edge = std::make_shared<
            shammodels::basegodunov::solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(
            "cell_graph_edge", "cell graph edge");

    BlockNeighToCellNeigh<Tvec, TgridVec, u_morton> node(Config::NsideBlockPow);
    node.set_edges(
        storage.block_counts_with_ghost,
        storage.refs_block_min,
        storage.refs_block_max,
        block_graph_edge,
        cell_graph_edge);
    node.evaluate();

    storage.cell_link_graph.set(std::move(cell_graph_edge->graph));
}

template class shammodels::basegodunov::modules::AMRGraphGen<f64_3, i64_3>;
