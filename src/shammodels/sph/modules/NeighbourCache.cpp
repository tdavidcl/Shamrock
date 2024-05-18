// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NeighbourCache.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "ExternalForces.hpp"

#include "shambase/aliases_int.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/modules/NeighbourCache.hpp"
#include "shamrock/legacy/utils/geometry_utils.hpp"
#include "shamrock/tree/TreeTraversal.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamunits/Constants.hpp"


template<class Tvec, class Tmorton, template<class> class SPHKernel>
void shammodels::sph::modules::NeighbourCache<Tvec,Tmorton, SPHKernel>::start_neighbors_cache() {

    // interface_control
    using GhostHandle        = sph::BasicSPHGhostHandler<Tvec>;
    using GhostHandleCache   = typename GhostHandle::CacheMap;
    using PreStepMergedField = typename GhostHandle::PreStepMergedField;
    using RTree = RadixTree<Tmorton, Tvec>;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    storage.neighbors_cache.set(shamrock::tree::ObjectCacheHandler(solver_config.max_neigh_cache_size, [&](u64 patch_id) {
        logger::debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        PreStepMergedField &mfield = storage.merged_xyzh.get().get(patch_id);

        sycl::buffer<Tvec> &buf_xyz    = shambase::get_check_ref(mfield.field_pos.get_buf());
        sycl::buffer<Tscal> &buf_hpart = shambase::get_check_ref(mfield.field_hpart.get_buf());
        sycl::buffer<Tscal> &tree_field_rint = shambase::get_check_ref(
            storage.rtree_rint_field.get().get(patch_id).radix_tree_field_buf);

        sycl::range range_npart{mfield.original_elements};

        RTree &tree = storage.merged_pos_trees.get().get(patch_id);

        u32 obj_cnt       = mfield.original_elements;
        Tscal h_tolerance = solver_config.htol_up_tol;

        NamedStackEntry stack_loc1{"init cache"};

        using namespace shamrock;

        sycl::buffer<u32> neigh_count(obj_cnt);

        shamsys::instance::get_compute_queue().wait_and_throw();

        logger::debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);

        shamsys::instance::get_compute_queue().submit([&, h_tolerance](sycl::handler &cgh) {
            tree::ObjectIterator particle_looper(tree, cgh);

            // tree::LeafCacheObjectIterator particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};

            sycl::accessor rint_tree{tree_field_rint, cgh, sycl::read_only};

            sycl::accessor neigh_cnt{neigh_count, cgh, sycl::write_only, sycl::no_init};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, obj_cnt, "compute neigh cache 1", [=](u64 gid) {
                u32 id_a = (u32)gid;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                u32 cnt = 0;

                particle_looper.rtree_for(
                    [&](u32 node_id, Tvec bmin, Tvec bmax) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        using namespace walker::interaction_crit;

                        return sph_radix_cell_crit(
                            xyz_a, inter_box_a_min, inter_box_a_max, bmin, bmax, int_r_max_cell);
                    },
                    [&](u32 id_b) {
                        // particle_looper.for_each_object(id_a,[&](u32 id_b){
                        //  compute only omega_a
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact =
                            rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        cnt += (no_interact) ? 0 : 1;
                    });

                neigh_cnt[id_a] = cnt;
            });
        });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc2{"fill cache"};

        shamsys::instance::get_compute_queue().submit([&, h_tolerance](sycl::handler &cgh) {
            tree::ObjectIterator particle_looper(tree, cgh);

            // tree::LeafCacheObjectIterator particle_looper(tree,*xyz_cell_id,leaf_cache,cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};

            sycl::accessor rint_tree{tree_field_rint, cgh, sycl::read_only};

            sycl::accessor scanned_neigh_cnt{pcache.scanned_cnt, cgh, sycl::read_only};
            sycl::accessor neigh{pcache.index_neigh_map, cgh, sycl::write_only, sycl::no_init};

            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, obj_cnt, "compute neigh cache 2", [=](u64 gid) {
                u32 id_a = (u32)gid;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                u32 cnt = scanned_neigh_cnt[id_a];

                particle_looper.rtree_for(
                    [&](u32 node_id, Tvec bmin, Tvec bmax) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        using namespace walker::interaction_crit;

                        return sph_radix_cell_crit(
                            xyz_a, inter_box_a_min, inter_box_a_max, bmin, bmax, int_r_max_cell);
                    },
                    [&](u32 id_b) {
                        // particle_looper.for_each_object(id_a,[&](u32 id_b){
                        //  compute only omega_a
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact =
                            rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        if (!no_interact) {
                            neigh[cnt] = id_b;
                        }
                        cnt += (no_interact) ? 0 : 1;
                    });
            });
        });

        return pcache;
    }));

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        storage.neighbors_cache.get().preload(cur_p.id_patch);
    });

    time_neigh.end();
    storage.timings_details.neighbors += time_neigh.elasped_sec();

}

template<class Tvec, class Tmorton, template<class> class SPHKernel>
void shammodels::sph::modules::NeighbourCache<Tvec,Tmorton, SPHKernel>::start_neighbors_cache_2stages() {


    // interface_control
    using GhostHandle        = sph::BasicSPHGhostHandler<Tvec>;
    using GhostHandleCache   = typename GhostHandle::CacheMap;
    using PreStepMergedField = typename GhostHandle::PreStepMergedField;
    using RTree = RadixTree<Tmorton, Tvec>;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    storage.neighbors_cache.set(shamrock::tree::ObjectCacheHandler(solver_config.max_neigh_cache_size, [&](u64 patch_id) {
        logger::debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        PreStepMergedField &mfield = storage.merged_xyzh.get().get(patch_id);

        sycl::buffer<Tvec> &buf_xyz    = shambase::get_check_ref(mfield.field_pos.get_buf());
        sycl::buffer<Tscal> &buf_hpart = shambase::get_check_ref(mfield.field_hpart.get_buf());
        sycl::buffer<Tscal> &tree_field_rint = shambase::get_check_ref(
            storage.rtree_rint_field.get().get(patch_id).radix_tree_field_buf);


        RTree &tree = storage.merged_pos_trees.get().get(patch_id);


        u32 leaf_cnt = tree.tree_reduced_morton_codes.tree_leaf_count;
        u32 intnode_cnt = tree.tree_struct.internal_cell_count;
        u32 obj_cnt       = mfield.original_elements;


        sycl::range range_nleaf{leaf_cnt};
        sycl::range range_nobj{obj_cnt};
        using namespace shamrock;


        Tscal h_tolerance = solver_config.htol_up_tol;

        NamedStackEntry stack_loc1{"init cache"};

        //start by counting number of leaf neighbours

        sycl::buffer<u32> neigh_count_leaf(leaf_cnt);

        shamsys::instance::get_compute_queue().wait_and_throw();

        logger::debug_sycl_ln("Cache", "generate cache for Nleaf=", leaf_cnt);



        shamsys::instance::get_compute_queue().submit([&, h_tolerance](sycl::handler &cgh) {
            tree::LeafIterator leaf_looper(tree, cgh);

            sycl::accessor rint_tree{tree_field_rint, cgh, sycl::read_only};
            sycl::accessor neigh_cnt{neigh_count_leaf, cgh, sycl::write_only, sycl::no_init};

            u32 offset_leaf = intnode_cnt;

            shambase::parralel_for(cgh, leaf_cnt, "compute neigh cache 1", [=](u64 gid) {
                u32 id_a = (u32)gid;

                Tscal leaf_a_rint = rint_tree[offset_leaf + gid]* Kernel::Rkern;
                Tvec leaf_a_bmin = leaf_looper.pos_min_cell[offset_leaf + gid];
                Tvec leaf_a_bmax = leaf_looper.pos_max_cell[offset_leaf + gid];
                Tvec leaf_a_bmin_ext = leaf_a_bmin - leaf_a_rint;
                Tvec leaf_a_bmax_ext = leaf_a_bmax + leaf_a_rint;

                u32 cnt = 0;

                leaf_looper.rtree_for(
                    [&](u32 node_id, Tvec bmin, Tvec bmax) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        Tvec ext_bmin = bmin - int_r_max_cell;
                        Tvec ext_bmax = bmax + int_r_max_cell;

                        return BBAA::cella_neigh_b(
                            leaf_a_bmin, leaf_a_bmax, 
                            ext_bmin, ext_bmax) ||
                        BBAA::cella_neigh_b(
                            leaf_a_bmin_ext, leaf_a_bmax_ext,                   
                            bmin, bmax);

                    },
                    [&](u32 leaf_b) {
                        cnt ++;
                    });

                neigh_cnt[id_a] = cnt;
            });
        });

        //{
        //    u32 offset_leaf = intnode_cnt;
        //    sycl::host_accessor neigh_cnt{neigh_count_leaf};
        //    sycl::host_accessor pos_min_cell  {shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt)};
        //    sycl::host_accessor pos_max_cell  {shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt)};
        //    
        //    for (u32 i = 0; i < 1000; i++) {
        //        if(neigh_cnt[i] > 30){
        //            logger::raw_ln(i, neigh_cnt[i], pos_max_cell[i+offset_leaf] - pos_min_cell[i+offset_leaf]);
        //        }
        //    }
        //}

        tree::ObjectCache pleaf_cache = tree::prepare_object_cache(std::move(neigh_count_leaf), leaf_cnt);



        //fill ids of leaf neighbours

        NamedStackEntry stack_loc2{"fill cache"};

        shamsys::instance::get_compute_queue().submit([&, h_tolerance](sycl::handler &cgh) {
            tree::LeafIterator leaf_looper(tree, cgh);

            sycl::accessor rint_tree{tree_field_rint, cgh, sycl::read_only};

            u32 offset_leaf = intnode_cnt;

            sycl::accessor scanned_neigh_cnt{pleaf_cache.scanned_cnt, cgh, sycl::read_only};
            sycl::accessor neigh{pleaf_cache.index_neigh_map, cgh, sycl::write_only, sycl::no_init};


            shambase::parralel_for(cgh, leaf_cnt, "compute neigh cache 2", [=](u64 gid) {
                u32 id_a = (u32)gid;

                Tscal leaf_a_rint = rint_tree[offset_leaf + gid]* Kernel::Rkern;
                Tvec leaf_a_bmin = leaf_looper.pos_min_cell[offset_leaf + gid];
                Tvec leaf_a_bmax = leaf_looper.pos_max_cell[offset_leaf + gid];
                Tvec leaf_a_bmin_ext = leaf_a_bmin - leaf_a_rint;
                Tvec leaf_a_bmax_ext = leaf_a_bmax + leaf_a_rint;

                u32 cnt = scanned_neigh_cnt[id_a];

                leaf_looper.rtree_for(
                    [&](u32 node_id, Tvec bmin, Tvec bmax) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        Tvec ext_bmin = bmin - int_r_max_cell;
                        Tvec ext_bmax = bmax + int_r_max_cell;

                        return BBAA::cella_neigh_b(
                            leaf_a_bmin, leaf_a_bmax, 
                            ext_bmin, ext_bmax) ||
                        BBAA::cella_neigh_b(
                            leaf_a_bmin_ext, leaf_a_bmax_ext,                   
                            bmin, bmax);

                    },
                    [&](u32 leaf_b) {
                        neigh[cnt] = leaf_b;
                        cnt ++;
                    });
            });
        });





        // search in which leaf each parts are
        sycl::buffer<u32> leaf_part_id(obj_cnt);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            tree::LeafIterator leaf_looper(tree, cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};

            sycl::accessor found_id{leaf_part_id, cgh, sycl::write_only, sycl::no_init};
            u32 offset_leaf = intnode_cnt;
            //sycl::stream out {4096,4096,cgh};
            shambase::parralel_for(cgh, obj_cnt, "search particles parent leaf", [=](u64 gid) {
                u32 id_a = (u32)gid;

                Tvec r_a = xyz[id_a];

                u32 found_id_ = i32_max;// to ensure a crash because of out of bound access if not found

                leaf_looper.rtree_for(
                    [&](u32 node_id, Tvec bmin, Tvec bmax) -> bool {
                        bool ret = BBAA::is_coord_in_range_incl_max(
                            r_a, bmin,bmax);
                        
                        //error : i= 44245 r= (0.3495433344162232,-0.005627362002766546,-0.21312104638358176) leaf_id= 2147483647 
                        //if(id_a == 44245) {out << node_id << " " << bmin << " " << bmax << " " << ret << "\n";};
                        return ret;

                    },
                    [&](u32 leaf_b) {
                        found_id_ = leaf_b - offset_leaf;
                    });

                found_id[id_a] = found_id_;
            });
        });



        //{
        //    sycl::host_accessor xyz{buf_xyz};
        //    sycl::host_accessor acc {leaf_part_id};
        //
        //    for(u32 i = 0; i < obj_cnt; i++){
        //        u32 leaf_id = acc[i];
        //        if(leaf_id >= leaf_cnt){
        //            logger::raw_ln("error : i=",i,"r=",xyz[i],"leaf_id=",leaf_id);
        //        }
        //    }
        //}






















        sycl::buffer<u32> neigh_count(obj_cnt);

        shamsys::instance::get_compute_queue().wait_and_throw();

        logger::debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);

        shamsys::instance::get_compute_queue().submit([&, h_tolerance](sycl::handler &cgh) {
            
            tree::ObjectCacheIterator neigh_leaf_looper(pleaf_cache, cgh);
            tree::ObjectIterator particle_looper(tree, cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};

            sycl::accessor neigh_cnt{neigh_count, cgh, sycl::write_only, sycl::no_init};

            sycl::accessor leaf_owner{leaf_part_id, cgh, sycl::read_only};

            u32 offset_leaf = intnode_cnt;
            // sycl::stream out {4096,1024,cgh};

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, obj_cnt, "compute neigh cache 1", [=](u64 gid) {
                u32 id_a = (u32)gid;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                u32 cnt = 0;

                u32 leaf_own_a = leaf_owner[id_a];

                neigh_leaf_looper.for_each_object(
                    leaf_own_a
                    ,
                    [&](u32 leaf_b) {
                        particle_looper.iter_object_in_cell(leaf_b,[&](u32 id_b){
                            
                            Tvec dr      = xyz_a - xyz[id_b];
                            Tscal rab2   = sycl::dot(dr, dr);
                            Tscal rint_b = hpart[id_b] * h_tolerance;

                            bool no_interact =
                                rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                            cnt += (no_interact) ? 0 : 1;
                        });
                    });

                neigh_cnt[id_a] = cnt;
            });
        });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc3{"fill cache"};

        shamsys::instance::get_compute_queue().submit([&, h_tolerance](sycl::handler &cgh) {
            tree::ObjectCacheIterator neigh_leaf_looper(pleaf_cache, cgh);
            tree::ObjectIterator particle_looper(tree, cgh);

            sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
            sycl::accessor hpart{buf_hpart, cgh, sycl::read_only};

            sycl::accessor scanned_neigh_cnt{pcache.scanned_cnt, cgh, sycl::read_only};
            sycl::accessor neigh{pcache.index_neigh_map, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor leaf_owner{leaf_part_id, cgh, sycl::read_only};

            u32 offset_leaf = intnode_cnt;

            constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

            shambase::parralel_for(cgh, obj_cnt, "compute neigh cache 2", [=](u64 gid) {
                u32 id_a = (u32)gid;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                u32 cnt = scanned_neigh_cnt[id_a];

                neigh_leaf_looper.for_each_object(
                    leaf_owner[id_a]
                    ,
                    [&](u32 leaf_b) {
                        particle_looper.iter_object_in_cell(leaf_b,[&](u32 id_b){
                            
                            Tvec dr      = xyz_a - xyz[id_b];
                            Tscal rab2   = sycl::dot(dr, dr);
                            Tscal rint_b = hpart[id_b] * h_tolerance;

                            bool no_interact =
                                rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                            if (!no_interact) {
                                neigh[cnt] = id_b;
                            }
                            cnt += (no_interact) ? 0 : 1;
                        });
                    });
            });
        });






















        return pcache;
    }));

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        storage.neighbors_cache.get().preload(cur_p.id_patch);
    });

    time_neigh.end();
    storage.timings_details.neighbors += time_neigh.elasped_sec();

}


using namespace shammath;
template class shammodels::sph::modules::NeighbourCache<f64_3,u32, M4>;
template class shammodels::sph::modules::NeighbourCache<f64_3,u32, M6>;
template class shammodels::sph::modules::NeighbourCache<f64_3,u32, M8>;