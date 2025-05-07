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

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/riemann.hpp"
#include "shammath/riemann_dust.hpp"
#include "shammodels/ramses/modules/ConsToPrim.hpp"
#include "shammodels/ramses/modules/ConsToPrimGas.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/ComputeFieldEdge.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include <memory>
#include <utility>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::cons_to_prim_dust_spans(
    shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rho_dust,
    shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_rhov_dust,

    shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_vel_dust,
    shambase::DistributedData<u32> &sizes) {

    u32 ndust = solver_config.dust_config.ndust;

    shambase::DistributedData<u32> cell_counts = sizes.map<u32>([&](u64 id, u32 block_count) {
        u32 cell_count = block_count * AMRBlock::block_size * ndust;
        return cell_count;
    });

    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{spans_rho_dust, spans_rhov_dust},
        sham::DDMultiRef{spans_vel_dust},
        cell_counts,
        [gamma = solver_config.eos_gamma](
            u32 i,
            const Tscal *__restrict rho_dust,
            const Tvec *__restrict rhov_dust,
            Tvec *__restrict vel_dust) {
            auto d_conststate = shammath::DustConsState<Tvec>{rho_dust[i], rhov_dust[i]};

            auto d_prim_state = shammath::d_cons_to_prim(d_conststate);

            vel_dust[i] = d_prim_state.vel;
        });
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::cons_to_prim_gas() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());

    shamrock::ComputeField<Tvec> v_ghost
        = utility.make_compute_field<Tvec>("vel", AMRBlock::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });

    shamrock::ComputeField<Tscal> P_ghost
        = utility.make_compute_field<Tscal>("P", AMRBlock::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_ghost                                 = ghost_layout.get_field_idx<Tscal>("rho");
    u32 irhov_ghost                                = ghost_layout.get_field_idx<Tvec>("rhovel");
    u32 irhoe_ghost                                = ghost_layout.get_field_idx<Tscal>("rhoetot");

    auto block_counts
        = std::make_shared<shamrock::solvergraph::Indexes<u32>>("block_count", "N_{\\rm block}");
    block_counts->indexes
        = storage.merged_patchdata_ghost.get().template map<u32>([&](u64 id, MergedPDat &mpdat) {
              return mpdat.total_elements;
          });

    auto spans_rho   = std::make_shared<shamrock::solvergraph::FieldSpan<Tscal>>("rho", "\\rho");
    spans_rho->spans = storage.merged_patchdata_ghost.get()
                           .template map<shamrock::PatchDataFieldSpanPointer<Tscal>>(
                               [&](u64 id, MergedPDat &mpdat) {
                                   return mpdat.pdat.get_field_pointer_span<Tscal>(irho_ghost);
                               });

    auto spans_rhov
        = std::make_shared<shamrock::solvergraph::FieldSpan<Tvec>>("rhovel", "(\\rho \\mathbf{v})");
    spans_rhov->spans = storage.merged_patchdata_ghost.get()
                            .template map<shamrock::PatchDataFieldSpanPointer<Tvec>>(
                                [&](u64 id, MergedPDat &mpdat) {
                                    return mpdat.pdat.get_field_pointer_span<Tvec>(irhov_ghost);
                                });

    auto spans_rhoe
        = std::make_shared<shamrock::solvergraph::FieldSpan<Tscal>>("rhoetot", "(\\rho E)");
    spans_rhoe->spans = storage.merged_patchdata_ghost.get()
                            .template map<shamrock::PatchDataFieldSpanPointer<Tscal>>(
                                [&](u64 id, MergedPDat &mpdat) {
                                    return mpdat.pdat.get_field_pointer_span<Tscal>(irhoe_ghost);
                                });

    auto spans_vel = std::make_shared<shamrock::solvergraph::FieldSpan<Tvec>>("vel", "\\mathbf{v}");
    spans_vel->spans = storage.merged_patchdata_ghost.get()
                           .template map<shamrock::PatchDataFieldSpanPointer<Tvec>>(
                               [&](u64 id, MergedPDat &mpdat) {
                                   return v_ghost.get_field(id).get_pointer_span();
                               });

    auto spans_P   = std::make_shared<shamrock::solvergraph::FieldSpan<Tscal>>("P", "P");
    spans_P->spans = storage.merged_patchdata_ghost.get()
                         .template map<shamrock::PatchDataFieldSpanPointer<Tscal>>(
                             [&](u64 id, MergedPDat &mpdat) {
                                 return P_ghost.get_field(id).get_pointer_span();
                             });

    NodeConsToPrimGas<Tvec> node{AMRBlock::block_size, solver_config.eos_gamma};

    node.set_edges(block_counts, spans_rho, spans_rhov, spans_rhoe, spans_vel, spans_P);
    node.evaluate();

    logger::raw_ln(" --- dot:\n" + node.get_dot_graph());
    logger::raw_ln(" --- tex:\n" + node.get_tex());

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

    shamrock::ComputeField<Tvec> v_dust_ghost
        = utility.make_compute_field<Tvec>("vel_dust", ndust * AMRBlock::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });
    u32 irho_dust_ghost  = ghost_layout.get_field_idx<Tscal>("rho_dust");
    u32 irhov_dust_ghost = ghost_layout.get_field_idx<Tvec>("rhovel_dust");

    auto spans_rho_dust
        = storage.merged_patchdata_ghost.get()
              .template map<shamrock::PatchDataFieldSpanPointer<Tscal>>(
                  [&](u64 id, MergedPDat &mpdat) {
                      return mpdat.pdat.get_field_pointer_span<Tscal>(irho_dust_ghost);
                  });

    auto spans_rhov_dust
        = storage.merged_patchdata_ghost.get()
              .template map<shamrock::PatchDataFieldSpanPointer<Tvec>>(
                  [&](u64 id, MergedPDat &mpdat) {
                      return mpdat.pdat.get_field_pointer_span<Tvec>(irhov_dust_ghost);
                  });

    auto spans_vel_dust = storage.merged_patchdata_ghost.get()
                              .template map<shamrock::PatchDataFieldSpanPointer<Tvec>>(
                                  [&](u64 id, MergedPDat &mpdat) {
                                      return v_dust_ghost.get_field(id).get_pointer_span();
                                  });

    shambase::DistributedData<u32> block_sizes
        = storage.merged_patchdata_ghost.get().template map<u32>([&](u64 id, MergedPDat &mpdat) {
              return mpdat.total_elements;
          });

    cons_to_prim_dust_spans(spans_rho_dust, spans_rhov_dust, spans_vel_dust, block_sizes);

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
