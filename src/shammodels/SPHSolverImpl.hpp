// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/scheduler/scheduler_mpi.hpp"
#include "shamrock/sph/kernels.hpp"
#include "shamrock/tree/RadixTree.hpp"
#include "shamrock/tree/TreeTraversal.hpp"

namespace shammodels {

    class SPHSolverImpl {
        public:
        using flt                = f64;
        using vec                = f64_3;
        static constexpr u32 dim = 3;
        using u_morton           = u32;
        using Kernel             = shamrock::sph::kernels::M4<flt>;

        static constexpr flt Rkern = Kernel::Rkern;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
        SPHSolverImpl(ShamrockCtx &ctx) : context(ctx){};

        static shamrock::tree::ObjectCache build_neigh_cache(u32 start_offset,
                                                             u32 obj_cnt,
                                                             sycl::buffer<vec> &buf_xyz,
                                                             sycl::buffer<flt> &buf_hpart,
                                                             RadixTree<u_morton, vec, dim> &tree,
                                                             sycl::buffer<flt> &tree_field_hmax);

        static shamrock::tree::ObjectCache
        build_hiter_neigh_cache(u32 start_offset,
                                u32 obj_cnt,
                                sycl::buffer<vec> &buf_xyz,
                                sycl::buffer<flt> &buf_hpart,
                                RadixTree<u_morton, vec, dim> &tree,
                                flt h_tolerance);

        using MergedPositions = shambase::DistributedData<shamrock::MergedPatchDataField<vec>>;
        using RTree           = RadixTree<u_morton, vec, dim>;

        static shambase::DistributedData<RTree>
        make_merge_patch_trees(MergedPositions &merged_xyz, u32 reduction_level) {
            shambase::DistributedData<RTree> trees =
                merged_xyz.map<RTree>([&](u64 id, shamrock::MergedPatchDataField<vec> &merged) {
                    vec bmin = merged.bounds->lower;
                    vec bmax = merged.bounds->upper;

                    RTree tree(shamsys::instance::get_compute_queue(),
                               {bmin, bmax},
                               merged.field.get_buf(),
                               merged.field.get_obj_cnt(),
                               reduction_level);

                    return tree;
                });

            trees.for_each([&](u64 id, RTree &tree) {
                tree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
                tree.convert_bounding_box(shamsys::instance::get_compute_queue());
            });

            return trees;
        }
    };

} // namespace shammodels