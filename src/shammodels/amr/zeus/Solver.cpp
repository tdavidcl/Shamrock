// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/zeus/Solver.hpp"
#include "shammodels/amr/zeus/modules/AMRTree.hpp"
#include "shammodels/amr/zeus/modules/ComputePressure.hpp"
#include "shammodels/amr/zeus/modules/DiffOperator.hpp"
#include "shammodels/amr/zeus/modules/FaceFlagger.hpp"
#include "shammodels/amr/zeus/modules/GhostZones.hpp"
#include "shammodels/amr/zeus/modules/SourceStep.hpp"
#include "shammodels/amr/zeus/modules/ValueLoader.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
using Solver = shammodels::zeus::Solver<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
auto Solver<Tvec, TgridVec>::evolve_once(Tscal t_current, Tscal dt_input) -> Tscal{

    StackEntry stack_loc{};
    shambase::Timer tstep;
    tstep.start();

    SerialPatchTree<TgridVec> _sptree = SerialPatchTree<TgridVec>::build(scheduler());
    _sptree.attach_buf();
    storage.serial_patch_tree.set(std::move(_sptree));

    //ghost zone exchange
    modules::GhostZones gz(context,solver_config,storage);
    gz.build_ghost_cache();

    gz.exchange_ghost();


    //compute bound received
    //round to next pow of 2
    //build radix trees
    modules::AMRTree amrtree(context,solver_config,storage);
    amrtree.build_trees();

    //build neigh table
    amrtree.build_neigh_cache();


    modules::ComputePressure comp_eos(context, solver_config, storage);
    comp_eos.compute_p();
    
    modules::FaceFlagger compute_face_flag(context,solver_config,storage);
    compute_face_flag.flag_faces();
    compute_face_flag.split_face_list();

    
    //modules::DiffOperator diff_op(context,solver_config,storage);
    //diff_op.compute_gradu();





    //save velocity field
    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");
    u32 ieint_interf                                = ghost_layout.get_field_idx<Tscal>("eint");
    u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

    shamrock::SchedulerUtility utility(scheduler());
    storage.vel_n.set(
        utility.save_field_custom<Tvec>("vel_n", [&](u64 id_patch)-> PatchDataField<Tvec> & {
            using MergedPDat = shamrock::MergedPatchData;
            MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id_patch);
            return mpdat.pdat.get_field<Tvec>(ivel_interf);
        })
    );

    //prepare velocity gradients
    modules::ValueLoader<Tvec, TgridVec, Tvec> val_load_vec(context, solver_config, storage);
    storage.vel_n_xp.set( val_load_vec.load_value_with_gz("vel", {1, 0, 0}, "vel_n_xp"));
    storage.vel_n_yp.set( val_load_vec.load_value_with_gz("vel", {0, 1, 0}, "vel_n_yp"));
    storage.vel_n_zp.set( val_load_vec.load_value_with_gz("vel", {0, 0, 1}, "vel_n_zp"));

    modules::ValueLoader<Tvec, TgridVec, Tscal> val_load_scal(context, solver_config, storage);
    storage.rho_n_xm.set( val_load_scal.load_value_with_gz("rho", {-1, 0, 0}, "rho_n_xm"));
    storage.rho_n_ym.set( val_load_scal.load_value_with_gz("rho", {0, -1, 0}, "rho_n_ym"));
    storage.rho_n_zm.set( val_load_scal.load_value_with_gz("rho", {0, 0, -1}, "rho_n_zm"));

    shamrock::ComputeField<Tscal> &pressure_field = storage.pressure.get();
    storage.pres_n_xm.set( val_load_scal.load_value_with_gz(pressure_field, {-1, 0, 0}, "pres_n_xm"));
    storage.pres_n_ym.set( val_load_scal.load_value_with_gz(pressure_field, {0, -1, 0}, "pres_n_ym"));
    storage.pres_n_zm.set( val_load_scal.load_value_with_gz(pressure_field, {0, 0, -1}, "pres_n_zm"));

    modules::SourceStep src_step(context,solver_config,storage);
    src_step.compute_forces();
    src_step.apply_force(dt_input);

    shamrock::ComputeField<Tvec> &q_AV = storage.q_AV.get();
    storage.q_AV_n_xm.set( val_load_vec.load_value_with_gz(q_AV, {-1, 0, 0}, "q_AV_n_xm"));
    storage.q_AV_n_ym.set( val_load_vec.load_value_with_gz(q_AV, {0, -1, 0}, "q_AV_n_ym"));
    storage.q_AV_n_zm.set( val_load_vec.load_value_with_gz(q_AV, {0, 0, -1}, "q_AV_n_zm"));
    


using namespace shamrock::patch;
    using namespace shamrock;

    using Block = typename Config::AMRBlock;

    

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {

        using MergedPDat = shamrock::MergedPatchData;
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<Tscal> &rho_merged = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tscal> &eint_merged = mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf);
        sycl::buffer<Tvec> &vel_merged = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);

        PatchData &patch_dest = scheduler().patch_data.get_pdat(p.id_patch);
        sycl::buffer<Tscal> &rho_dest = patch_dest.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tscal> &eint_dest = patch_dest.get_field_buf_ref<Tscal>(ieint_interf);
        sycl::buffer<Tvec> &vel_dest = patch_dest.get_field_buf_ref<Tvec>(ivel_interf);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor acc_rho_src{rho_merged, cgh, sycl::read_only};
            sycl::accessor acc_eint_src{eint_merged, cgh, sycl::read_only};
            sycl::accessor acc_vel_src{vel_merged, cgh, sycl::read_only};

            sycl::accessor acc_rho_dest{rho_dest, cgh, sycl::write_only};
            sycl::accessor acc_eint_dest{eint_dest, cgh, sycl::write_only};
            sycl::accessor acc_vel_dest{vel_dest, cgh, sycl::write_only};

            shambase::parralel_for(cgh, mpdat.original_elements*Block::block_size, "copy_back", [=](u32 id){
                acc_rho_dest[id] = acc_rho_src[id];
                acc_eint_dest[id] = acc_eint_src[id];
                acc_vel_dest[id] = acc_vel_src[id];
            });
        });
        
    });





    storage.face_lists.reset();
    storage.pressure.reset();
    storage.trees.reset();
    storage.merge_patch_bounds.reset();
    storage.merged_patchdata_ghost.reset();
    storage.ghost_layout.reset();
    storage.ghost_zone_infos.reset();
    storage.serial_patch_tree.reset();


    tstep.end();

    u64 rank_count = scheduler().get_rank_count()*AMRBlock::block_size;
    f64 rate = f64(rank_count) / tstep.elasped_sec();

    std::string log_rank_rate = shambase::format(
        "\n| {:<4} |    {:.4e}    | {:11} |   {:.3e}   |  {:3.0f} % | {:3.0f} % | {:3.0f} % |", 
        shamsys::instance::world_rank,rate,  rank_count,  tstep.elasped_sec(),
        100*(storage.timings_details.interface / tstep.elasped_sec()),
        100*(storage.timings_details.neighbors / tstep.elasped_sec()),
        100*(storage.timings_details.io / tstep.elasped_sec())
        );

    std::string gathered = "";
    shamalgs::collective::gather_str(log_rank_rate, gathered);

    if(shamsys::instance::world_rank == 0){
        std::string print = "processing rate infos : \n";
        print+=("---------------------------------------------------------------------------------\n");
        print+=("| rank |  rate  (N.s^-1)  |      N      | t compute (s) | interf | neigh |   io  |\n");
        print+=("---------------------------------------------------------------------------------");
        print+=(gathered) + "\n";
        print+=("---------------------------------------------------------------------------------");
        logger::info_ln("amr::Zeus",print);
    }

    storage.timings_details.reset();

    return 0;
}


template class shammodels::zeus::Solver<f64_3, i64_3>;