// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


//%Impl status : Good

#pragma once


#include "aliases.hpp"
#include "core/io/logs.hpp"
#include "core/patch/base/patchdata.hpp"
#include "core/patch/base/patchdata_layout.hpp"
#include "core/patch/interfaces/interface_handler.hpp"
#include "core/patch/interfaces/interface_selector.hpp"
//#include "core/patch/patchdata_buffer.hpp"
#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "core/patch/utility/merged_patch.hpp"
#include "core/patch/utility/serialpatchtree.hpp"
#include "shamrock/sys/log.hpp"
#include "core/tree/radix_tree.hpp"
#include "models/sph/algs/smoothing_lenght_impl.hpp"
#include "models/sph/base/kernels.hpp"
#include "models/sph/base/sphpart.hpp"

namespace models::sph {
namespace algs {

template<class flt, class morton_prec, class Kernel>
class SmoothingLenghtCompute{

    using vec = sycl::vec<flt, 3>;

    using Rtree = Radix_Tree<morton_prec, vec>;
    using Rta = walker::Radix_tree_accessor<morton_prec, vec>;

    flt htol_up_tol;
    flt htol_up_iter;

    u32 ihpart;
    u32 ixyz;

    public:

    SmoothingLenghtCompute (
        PatchDataLayout &pdl,
        f32 htol_up_tol,
        f32 htol_up_iter){

        ixyz = pdl.get_field_idx<vec>("xyz");
        ihpart = pdl.get_field_idx<flt>("hpart");

        this->htol_up_tol  = htol_up_tol;
        this->htol_up_iter = htol_up_iter;
    }

    inline void iterate_smoothing_lenght(
        sycl::queue & queue,
        u32 or_element_cnt,
        
        flt gpart_mass,

        Rtree & radix_t,

        PatchData & pdat_merge,
        sycl::buffer<flt> & hnew,
        sycl::buffer<flt> & omega,
        sycl::buffer<flt> & eps_h){

        auto timer = timings::start_timer("iterate_smoothing_lenght",timings::function);

        impl::sycl_init_h_iter_bufs(queue, or_element_cnt,ihpart, pdat_merge, hnew, omega, eps_h);

        for (u32 it_num = 0 ; it_num < 30; it_num++) {

            impl::IntSmoothingLenghtCompute<morton_prec, Kernel>::template sycl_h_iter_step<flt>(queue, 
                or_element_cnt, 
                ihpart, 
                ixyz,
                gpart_mass,
                htol_up_tol,
                htol_up_iter,
                radix_t,
                pdat_merge, 
                hnew, 
                omega, 
                eps_h);

            //{
            //    sycl::host_accessor acc {eps_h};
            //
            //    logger::raw_ln("------eps_h-----");
            //    for (u32 i = 0; i < eps_h.size(); i ++) {
            //        logger::raw(acc[i],",");
            //    }
            //    logger::raw_ln("----------------");
            //}
        }

        impl::IntSmoothingLenghtCompute<morton_prec, Kernel>::template sycl_h_iter_omega<flt>(queue, 
                or_element_cnt, 
                ihpart, 
                ixyz,
                gpart_mass,
                htol_up_tol,
                htol_up_iter,
                radix_t,
                pdat_merge, 
                hnew, 
                omega, 
                eps_h);

        timer.stop();

    }

    #if false
    [[deprecated]]
    inline void iterate_smoothing_lenght(
        sycl::queue & queue,
        u32 or_element_cnt,
        
        flt gpart_mass,

        Rtree & radix_t,

        PatchDataBuffer & pdat_buf_merge,
        sycl::buffer<flt> & hnew,
        sycl::buffer<flt> & omega,
        sycl::buffer<flt> & eps_h){

        auto timer = timings::start_timer("iterate_smoothing_lenght",timings::function);

        impl::sycl_init_h_iter_bufs(queue, or_element_cnt,ihpart, pdat_buf_merge, hnew, omega, eps_h);

        for (u32 it_num = 0 ; it_num < 30; it_num++) {

            impl::IntSmoothingLenghtCompute<morton_prec, Kernel>::template sycl_h_iter_step<flt>(queue, 
                or_element_cnt, 
                ihpart, 
                ixyz,
                gpart_mass,
                htol_up_tol,
                htol_up_iter,
                radix_t,
                pdat_buf_merge, 
                hnew, 
                omega, 
                eps_h);

            //{
            //    sycl::host_accessor acc {eps_h};
            //
            //    logger::raw_ln("------eps_h-----");
            //    for (u32 i = 0; i < eps_h.size(); i ++) {
            //        logger::raw(acc[i],",");
            //    }
            //    logger::raw_ln("----------------");
            //}
        }

        impl::IntSmoothingLenghtCompute<morton_prec, Kernel>::template _sycl_h_iter_omega<flt>(queue, 
                or_element_cnt, 
                ihpart, 
                ixyz,
                gpart_mass,
                htol_up_tol,
                htol_up_iter,
                radix_t,
                pdat_buf_merge, 
                hnew, 
                omega, 
                eps_h);

        timer.stop();

    }

    #endif


};



template<class flt, class u_morton, class Kernel>
inline void compute_smoothing_lenght(PatchScheduler &sched,bool periodic_mode,flt htol_up_tol,
        flt htol_up_iter ,flt sph_gpart_mass){

        using vec = sycl::vec<flt, 3>;
        using Rtree = Radix_Tree<u_morton, vec>;
        using Rta = walker::Radix_tree_accessor<u_morton, vec>;

        

        const flt loc_htol_up_tol  = htol_up_tol;
        const flt loc_htol_up_iter = htol_up_iter;

        const u32 ixyz      = sched.pdl.get_field_idx<vec>("xyz");
        const u32 ihpart    = sched.pdl.get_field_idx<flt>("hpart");


        //Init serial patch tree
        SerialPatchTree<vec> sptree(sched.patch_tree, sched.get_box_tranform<vec>());
        sptree.attach_buf();


        //compute hmax
        PatchField<flt> h_field;
        sched.compute_patch_field(
            h_field, get_mpi_type<flt>(), [loc_htol_up_tol](sycl::queue &queue, Patch &p, PatchData &pdat) {
                return patchdata::sph::get_h_max<flt>(pdat.pdl, queue, pdat) * loc_htol_up_tol * Kernel::Rkern;
            });



        //make interfaces
        LegacyInterfacehandler<vec, flt> interface_hndl;
        interface_hndl.template compute_interface_list<InterfaceSelector_SPH<vec, flt>>(sched, sptree, h_field,
                                                                                                 periodic_mode);
        interface_hndl.comm_interfaces(sched, periodic_mode);


        // merging strategy
        std::unordered_map<u64, MergedPatchData<flt>> merge_pdat = 
            MergedPatchData<flt>::merge_patches(sched,interface_hndl);


        //make trees
        std::unordered_map<u64, std::unique_ptr<Radix_Tree<u_morton, vec>>> radix_trees;


        constexpr u32 reduc_level = 5;

        sched.for_each_patch([&](u64 id_patch, Patch  /*cur_p*/) {
            logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","making Radix Tree");

            if (merge_pdat.at(id_patch).or_element_cnt == 0)
                logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","is empty skipping tree build");

            //PatchDataBuffer &mpdat_buf = *merge_pdat_buf.at(id_patch).data;
            PatchData & mpdat = merge_pdat.at(id_patch).data;

            auto & buf_xyz = mpdat.get_field<vec>(ixyz).get_buf();

            std::tuple<vec, vec> &box = merge_pdat.at(id_patch).box;

            // radix tree computation
            radix_trees[id_patch] = std::make_unique<Radix_Tree<u_morton, vec>>(sycl_handler::get_compute_queue(), box,
                                                                                    buf_xyz,mpdat.get_obj_cnt(),reduc_level);
        });

        sched.for_each_patch([&](u64 id_patch, Patch  /*cur_p*/) {
            logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","compute radix tree cell volumes");
            if (merge_pdat.at(id_patch).or_element_cnt == 0)
                logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","is empty skipping tree volumes step");

            radix_trees[id_patch]->compute_cellvolume(sycl_handler::get_compute_queue());
        });

        sched.for_each_patch([&](u64 id_patch, Patch  /*cur_p*/) {
            logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","compute Radix Tree interaction boxes");
            if (merge_pdat.at(id_patch).or_element_cnt == 0)
                logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","is empty skipping interaction box compute");

            PatchData & mpdat = merge_pdat.at(id_patch).data;

            auto & buf_h = mpdat.get_field<flt>(ihpart).get_buf();

            radix_trees[id_patch]->compute_int_boxes(sycl_handler::get_compute_queue(), buf_h, htol_up_tol);
        });



        //create compute field for new h and omega
        std::cout << "making omega field" << std::endl;
        PatchComputeField<flt> hnew_field;
        PatchComputeField<flt> omega_field;

        hnew_field.generate(sched);
        omega_field.generate(sched);

        //iterate smoothing lenght
        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","Init h iteration");
            if (merge_pdat.at(id_patch).or_element_cnt == 0)
                logger::debug_ln("SPHLeapfrog","patch : n°",id_patch,"->","is empty skipping h iteration");


            PatchData &pdat_merge = merge_pdat.at(id_patch).data;

            auto & hnew  = hnew_field.get_buf(id_patch);
            auto & omega = omega_field.get_buf(id_patch);
            sycl::buffer<flt> eps_h  = sycl::buffer<flt>(merge_pdat.at(id_patch).or_element_cnt);

            sycl::range range_npart{merge_pdat.at(id_patch).or_element_cnt};

            logger::debug_ln("SPHLeapfrog","merging -> original size :" , merge_pdat.at(id_patch).or_element_cnt
                      , "| merged :" , pdat_merge.get_obj_cnt());

            SmoothingLenghtCompute<flt, u32, Kernel> h_iterator(sched.pdl, htol_up_tol, htol_up_iter);

            h_iterator.iterate_smoothing_lenght(sycl_handler::get_compute_queue(), merge_pdat.at(id_patch).or_element_cnt,
                                                sph_gpart_mass, *radix_trees[id_patch], pdat_merge, *hnew, *omega, eps_h);

            // write back h test
            //*
            sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {
                auto h_new = hnew->template get_access<sycl::access::mode::read>(cgh);

                auto acc_hpart = pdat_merge.get_field<flt>(ihpart).get_buf()->template get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for(range_npart,
                                                     [=](sycl::item<1> item) { acc_hpart[item] = h_new[item]; });
            });
            //*/
        });

        write_back_merge_patches(sched, merge_pdat);

        //hnew_field.to_map();
        //omega_field.to_map();

    }

} // namespace algs
} // namespace sph



