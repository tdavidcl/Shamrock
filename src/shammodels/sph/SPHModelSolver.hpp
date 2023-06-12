// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "SPHModelSolverConfig.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/sph/SPHUtilities.hpp"
#include "shamrock/tree/TreeTaversalCache.hpp"
#include <memory>
#include <variant>
namespace shammodels {

    /**
     * @brief The shamrock SPH model
     *
     * @tparam Tvec
     * @tparam SPHKernel
     */
    template<class Tvec, template<class> class SPHKernel>
    class SPHModelSolver {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;
        using u_morton           = u32;

        static constexpr Tscal Rkern = Kernel::Rkern;

        using Config = SPHModelSolverConfig<Tvec, SPHKernel>;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        Config solver_config;

        static constexpr Tscal htol_up_tol  = 1.1;
        static constexpr Tscal htol_up_iter = 1.1;

        Tscal eos_gamma;
        Tscal gpart_mass;
        Tscal cfl_cour;
        Tscal cfl_force;

        inline void init_required_fields() {
            context.pdata_layout_add_field<Tvec>("xyz", 1);
            context.pdata_layout_add_field<Tvec>("vxyz", 1);
            context.pdata_layout_add_field<Tvec>("axyz", 1);
            context.pdata_layout_add_field<Tscal>("hpart", 1);

            if (solver_config.has_uint_field()) {
                context.pdata_layout_add_field<Tscal>("uint", 1);
                context.pdata_layout_add_field<Tscal>("duint", 1);
            }

            if (solver_config.has_alphaAV_field()) {
                context.pdata_layout_add_field<Tscal>("alpha_AV", 1);
            }

            if (solver_config.has_divv_field()) {
                context.pdata_layout_add_field<Tscal>("divv", 1);
            }

            if (solver_config.has_curlv_field()) {
                context.pdata_layout_add_field<Tscal>("curlv", 1);
            }
        }

        // serial patch tree control
        std::unique_ptr<SerialPatchTree<Tvec>> sptree;
        void gen_serial_patch_tree();
        inline void reset_serial_patch_tree() { sptree.reset(); }

        // interface_control
        using GhostHandle        = sph::BasicSPHGhostHandler<Tvec>;
        using GhostHandleCache   = typename GhostHandle::CacheMap;
        using PreStepMergedField = typename GhostHandle::PreStepMergedField;

        std::unique_ptr<GhostHandle> ghost_handler;
        inline void gen_ghost_handler() {
            if (ghost_handler) {
                throw shambase::throw_with_loc<std::runtime_error>(
                    "please reset the ghost_handler before");
            }
            ghost_handler = std::make_unique<GhostHandle>(scheduler());
        }
        inline void reset_ghost_handler() { ghost_handler.reset(); }

        GhostHandleCache ghost_handle_cache;
        void build_ghost_cache();
        void clear_ghost_cache();

        struct TempFields {
            shambase::DistributedData<PreStepMergedField> merged_xyzh;
            shamrock::ComputeField<Tscal> omega;

            void clear() {
                merged_xyzh.reset();
                omega.reset();
            }

        } temp_fields;

        void merge_position_ghost();

        // trees
        using RTree = RadixTree<u_morton, Tvec>;
        shambase::DistributedData<RTree> merged_pos_trees;
        void build_merged_pos_trees();
        void clear_merged_pos_trees();

        shambase::DistributedData<RadixTreeField<Tscal>> rtree_rint_field;
        void compute_presteps_rint();
        void reset_presteps_rint();

        std::unique_ptr<shamrock::tree::ObjectCacheHandler> neighbors_cache;
        void start_neighbors_cache();
        void reset_neighbors_cache();

        void sph_prestep();

        void apply_position_boundary();

        void do_predictor_leapfrog(Tscal dt);

        shamrock::patch::PatchDataLayout ghost_layout;
        void init_ghost_layout();

        SPHModelSolver(ShamrockCtx &context) : context(context) {}

        Tscal evolve_once(Tscal dt_input,
                          bool do_dump,
                          std::string vtk_dump_name,
                          bool vtk_dump_patch_id);
    };

} // namespace shammodels