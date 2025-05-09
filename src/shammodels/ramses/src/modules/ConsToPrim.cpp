// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ConsToPrim.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/riemann.hpp"
#include "shammath/riemann_dust.hpp"
#include "shammodels/ramses/modules/ConsToPrim.hpp"
#include "shammodels/ramses/modules/ConsToPrimDust.hpp"
#include "shammodels/ramses/modules/ConsToPrimGas.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include <memory>
#include <utility>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::cons_to_prim_gas() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());

    // will be filled by NodeConsToPrimGas
    auto spans_vel = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
        AMRBlock::block_size, "vel", "\\mathbf{v}");
    auto spans_P
        = std::make_shared<shamrock::solvergraph::Field<Tscal>>(AMRBlock::block_size, "P", "P");

    NodeConsToPrimGas<Tvec> node{AMRBlock::block_size, solver_config.eos_gamma};

    node.set_edges(
        storage.block_counts_with_ghost,
        storage.spans_rho,
        storage.spans_rhov,
        storage.spans_rhoe,
        spans_vel,
        spans_P);
    node.evaluate();

    // logger::raw_ln(" --- dot:\n" + node.get_dot_graph());
    // logger::raw_ln(" --- tex:\n" + node.get_tex());

    auto print_comp_field_state = [](std::string name, auto &cfield) {
        logger::raw_ln("Comp field state :", name);
        cfield.field_data.for_each([&](u64 id, auto &f) {
            logger::raw_ln(id, f.get_obj_cnt());
        });
    };

    shamrock::ComputeField<Tvec> v_ghost  = std::move(spans_vel->extract());
    shamrock::ComputeField<Tscal> P_ghost = std::move(spans_P->extract());

    // print_comp_field_state("v_ghost", v_ghost);
    // print_comp_field_state("P_ghost", P_ghost);

    storage.vel.set(std::move(v_ghost));
    storage.press.set(std::move(P_ghost));
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::cons_to_prim_dust() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());

    u32 ndust                                      = solver_config.dust_config.ndust;
    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();

    // will be filled by NodeConsToPrimDust
    auto spans_vel_dust = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
        AMRBlock::block_size * ndust, "vel_dust", "\\mathbf{v}_{\\rm dust}");

    auto print_block_sizes = [](std::string name, auto &cfield) {
        logger::raw_ln("Comp field state :", name);
        cfield.for_each([&](u64 id, auto &f) {
            logger::raw_ln(id, f);
        });
    };

    // print_block_sizes("block_counts_with_ghost", storage.block_counts_with_ghost->indexes);

    NodeConsToPrimDust<Tvec> node{AMRBlock::block_size, ndust};

    node.set_edges(
        storage.block_counts_with_ghost,
        storage.spans_rho_dust,
        storage.spans_rhov_dust,
        spans_vel_dust);
    node.evaluate();

    // logger::raw_ln(" --- dot:\n" + node.get_dot_graph());
    // logger::raw_ln(" --- tex:\n" + node.get_tex());

    auto print_comp_field_state = [](std::string name, auto &cfield) {
        logger::raw_ln("Comp field state :", name);
        cfield.field_data.for_each([&](u64 id, auto &f) {
            logger::raw_ln(id, f.get_obj_cnt());
        });
    };

    shamrock::ComputeField<Tvec> v_dust_ghost = std::move(spans_vel_dust->extract());

    // print_comp_field_state("v_ghost", v_dust_ghost);

    storage.vel_dust.set(std::move(v_dust_ghost));
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::reset_gas() {
    storage.vel.reset();
    storage.press.reset();
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::reset_dust() {
    storage.vel_dust.reset();
}

template class shammodels::basegodunov::modules::ConsToPrim<f64_3, i64_3>;
