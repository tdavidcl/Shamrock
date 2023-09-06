// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/zeus/modules/SourceStep.hpp"
#include "shammodels/amr/zeus/modules/FaceFlagger.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::zeus::modules::SourceStep<Tvec, TgridVec>;


/*
template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::load_grid_val_xm(){

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;
    using Flagger = FaceFlagger<Tvec, TgridVec>;
    using Block = typename Config::AMRBlock;
    using T = Tscal;

    shamrock::SchedulerUtility utility(scheduler());
    ComputeField<T> tmp = utility.make_compute_field<T>("load_xm", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    });

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ifield                                = ghost_layout.get_field_idx<Tscal>("rho");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {

        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<T> &buf_src = mpdat.pdat.get_field_buf_ref<T>(ifield);
        sycl::buffer<T> &buf_dest = tmp.get_buf_check(p.id_patch); 

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

            sycl::accessor val_out{buf_dest, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor src{buf_src, cgh, sycl::read_only};

            shambase::parralel_for(cgh, pdat.get_obj_cnt()*Block::block_size, "compute xm val (1)", [=](u64 id_a) {

                const u32 base_idx = id_a;
                const u32 lid = id_a % Block::block_size;

                static_assert(dim == 3, "implemented only in dim 3");
                std::array<u32, 3> lid_coord = Block::get_coord(lid);

                if(lid_coord[0] > 0){
                    lid_coord[0] -= 1;
                    val_out[base_idx] =
                        src[base_idx -lid+Block::get_index(lid_coord)];
                }
                       
            });

        });

    });



    

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {

        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<T> &buf_src = mpdat.pdat.get_field_buf_ref<T>(ifield);
        sycl::buffer<T> &buf_dest = tmp.get_buf_check(p.id_patch); 

        shammodels::zeus::NeighFaceList<Tvec> & face_lists = storage.face_lists.get().get(p.id_patch);
        OrientedNeighFaceList<Tvec> & face_xm = face_lists.xm();

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

            sycl::accessor val_out{buf_dest, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor src{buf_src, cgh, sycl::read_only};

            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};
            tree::ObjectCacheIterator faces_xm(face_xm.neigh_info, cgh);

            shambase::parralel_for(cgh, pdat.get_obj_cnt()*Block::block_size, "compute xm val (2)", [=](u64 id_a) {

                const u32 base_idx = id_a;
                const u32 block_id = id_a / Block::block_size;
                const u32 lid = id_a % Block::block_size;

                std::array<u32, 3> lid_coord = Block::get_coord(lid);
                
                if(lid_coord[0] == 0){
                    auto tmp = cell_max[block_id] - cell_min[block_id];
                    i32 Va = tmp.x()*tmp.y()*tmp.z();

                    static_assert(dim == 3, "implemented only in dim 3");
                    faces_xm.for_each_object(block_id, [&](u32 block_id_b) {

                        auto tmp = cell_max[block_id_b] - cell_min[block_id_b];
                        i32 nV = tmp.x()*tmp.y()*tmp.z();

                        if(nV == Va){ //same level
                            val_out[base_idx] =
                            src[block_id_b*Block::block_size +Block::get_index({Block::Nside-1,lid_coord[1],lid_coord[2]})];
                        }

                    });

                }
                       
            });

        });

    });

    





    

}
*/



template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::compute_forces() {





    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Flagger = FaceFlagger<Tvec, TgridVec>;

    using Block = typename Config::AMRBlock;



    //load_grid_val_xm();



    shamrock::SchedulerUtility utility(scheduler());
    storage.forces.set(utility.make_compute_field<Tvec>("forces", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        shammodels::zeus::NeighFaceList<Tvec> & face_lists = storage.face_lists.get().get(p.id_patch);
        
        OrientedNeighFaceList<Tvec> & face_xm = face_lists.xm();
        OrientedNeighFaceList<Tvec> & face_ym = face_lists.ym();
        OrientedNeighFaceList<Tvec> & face_zm = face_lists.zm();

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact;

        sycl::buffer<Tscal> & buf_p = storage.pressure.get().get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_rho   = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tvec> &forces_buf = storage.forces.get().get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            
            tree::ObjectCacheIterator faces_xm(face_xm.neigh_info, cgh);
            tree::ObjectCacheIterator faces_ym(face_ym.neigh_info, cgh);
            tree::ObjectCacheIterator faces_zm(face_zm.neigh_info, cgh);

            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor grad_p{forces_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor rho{buf_rho, cgh, sycl::read_only};
            sycl::accessor p{buf_p, cgh, sycl::read_only};

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "compute grad p", [=](u64 id_a) {
                Tvec cell2_a = (cell_min[id_a] + cell_max[id_a]).template convert<Tscal>() *
                               coord_conv_fact * 0.5f;

                Tscal rho_i_j_k = rho[id_a];
                Tscal p_i_j_k = p[id_a];

                Tscal rho_im1_j_k ;
                Tscal p_im1_j_k ;

                Tvec sum_grad_p = {};






                auto tmp = cell_max[id_a] - cell_min[id_a];
                i32 Va = tmp.x()*tmp.y()*tmp.z(); 

                auto get_xm_val = [=](auto acc, u32 block_id, std::array<u32,3> cell_id){

                    // within block
                    if(cell_id[0] > 0){
                        return acc[block_id*Block::block_size +Block::get_index({cell_id[0]-1,cell_id[1],cell_id[2]})];
                    }

                    faces_xm.for_each_object(id_a, [&](u32 id_b) {

                        auto tmp = cell_max[id_b] - cell_min[id_b];
                        i32 nV = tmp.x()*tmp.y()*tmp.z();

                        if(nV == Va){ //same level
                            return acc[id_b*Block::block_size +Block::get_index({Block::Nside-1,cell_id[1],cell_id[2]})];
                        }else if(nV == Va*8){ // l-1 case (interpolate)

                            bool test1 = cell_max[id_a].z() == cell_max[id_b].z();
                            bool test2 = cell_max[id_a].y() == cell_max[id_b].y();

                            if((!test1) && (!test2)){
                                // low z + low y

                            }else if(test1 && (!test2)){
                                // high z + low y

                            }else if((!test1) && test2){
                                // low z + high y

                            }else if(test1 && test2){
                                // high z + high y

                            }

                        }else if(nV == Va/8){ // l+1 case (average)

                        }else{
                            Tvec{}/0;
                        }

                    });



                };







                // looks like it's on the double preicision roofline there is
                // nothing to optimize here turn around
                //or it was the case before i touched to it '^^
                faces_xm.for_each_object(id_a, [&](u32 id_b) {
                    Tvec cell2_b = (cell_min[id_b] + cell_max[id_b]).template convert<Tscal>() *
                                   coord_conv_fact * 0.5f;
                    Tscal rho_im1_j_k = rho[id_b];
                    Tscal p_im1_j_k = p[id_b];

                    Tscal dxmubi = sycl::dot(cell2_b - cell2_a, Tvec{-1, 0, 0});
                    Tscal div = dxmubi*(rho_i_j_k - rho_im1_j_k)/2;
                    sum_grad_p.x() -= (p_i_j_k - p_im1_j_k)/div;
                });

                faces_ym.for_each_object(id_a, [&](u32 id_b) {
                    Tvec cell2_b = (cell_min[id_b] + cell_max[id_b]).template convert<Tscal>() *
                                   coord_conv_fact * 0.5f;
                    Tscal rho_im1_j_k = rho[id_b];
                    Tscal p_im1_j_k = p[id_b];

                    Tscal dxmubi = sycl::dot(cell2_b - cell2_a, Tvec{0, -1, 0});
                    Tscal div = dxmubi*(rho_i_j_k - rho_im1_j_k)/2;
                    sum_grad_p.y() -= (p_i_j_k - p_im1_j_k)/div;
                });

                faces_zm.for_each_object(id_a, [&](u32 id_b) {
                    Tvec cell2_b = (cell_min[id_b] + cell_max[id_b]).template convert<Tscal>() *
                                   coord_conv_fact * 0.5f;
                    Tscal rho_im1_j_k = rho[id_b];
                    Tscal p_im1_j_k = p[id_b];

                    Tscal dxmubi = sycl::dot(cell2_b - cell2_a, Tvec{0, 0, -1});
                    Tscal div = dxmubi*(rho_i_j_k - rho_im1_j_k)/2;
                    sum_grad_p.z() -= (p_i_j_k - p_im1_j_k)/div;
                });

                grad_p[id_a] = sum_grad_p;
            });
        });
    });

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<Tvec> &forces_buf = storage.forces.get().get_buf_check(p.id_patch);

        sycl::buffer<Tscal> &buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact;

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor forces{forces_buf, cgh, sycl::read_write};
            sycl::accessor rho{buf_rho, cgh, sycl::read_only};

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "add ext force", [=](u64 id_a) {

                Tvec block_min = cell_min[id_a].template convert<Tscal>();
                Tvec block_max = cell_max[id_a].template convert<Tscal>();
                Tvec delta_cell = (block_max - block_min)/Block::side_size;
                Tvec delta_cell_h = delta_cell * Tscal(0.5);

                Block::for_each_cell_in_block(delta_cell, [=](u32 lid, Tvec delta){

                    auto get_ext_force = [](Tvec r) {
                        Tscal d = sycl::length(r);
                        return r / (d * d * d);
                    };
                    
                    forces[id_a*Block::block_size + lid] +=  get_ext_force(block_min + delta + delta_cell_h);
                });

            });
        });

        logger::raw_ln(storage.forces.get().get_field(p.id_patch).compute_max());
    });

    
}

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::apply_force(Tscal dt) {

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Flagger = FaceFlagger<Tvec, TgridVec>;

    using Block = typename Config::AMRBlock;

    PatchDataLayout &pdl = scheduler().pdl;
    const u32 ivel       = pdl.get_field_idx<Tvec>("vel");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        sycl::buffer<Tvec> &forces_buf = storage.forces.get().get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &vel_buf    = storage.forces.get().get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor forces{forces_buf, cgh, sycl::read_only};
            sycl::accessor vel{vel_buf, cgh, sycl::read_write};

            shambase::parralel_for(cgh, pdat.get_obj_cnt()*Block::block_size, "add ext force", [=](u64 id_a) {
                vel[id_a] += dt * forces[id_a];
            });
        });

        logger::raw_ln(storage.forces.get().get_field(p.id_patch).compute_max());
    });

    

    storage.forces.reset();
}



template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::compute_AV() {

}

template class shammodels::zeus::modules::SourceStep<f64_3, i64_3>;