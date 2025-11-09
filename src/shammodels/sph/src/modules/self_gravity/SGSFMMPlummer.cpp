// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SGSFMMPlummer.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/modules/self_gravity/SGSFMMPlummer.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/AABB.hpp"
#include "shammath/symtensor_collections.hpp"
#include "shammodels/sph/modules/self_gravity/compute_tree_mass_moments.hpp"
#include "shamphys/fmm/GreenFuncGravCartesian.hpp"
#include "shamphys/fmm/contract_grav_moment.hpp"
#include "shamphys/fmm/grav_moment_offset.hpp"
#include "shamphys/fmm/offset_multipole.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/CLBVHDualTreeTraversal.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, u32 mm_order>
    void SGSFMMPlummer<Tvec, mm_order>::_impl_evaluate_internal() {
        __shamrock_stack_entry();

        using Umorton       = u32;
        using RTree         = shamtree::CompressedLeafBVH<Umorton, Tvec, 3>;
        u32 reduction_level = 3;

        auto edges = get_edges();

        if (edges.sizes.indexes.get_ids().size() != 1) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Self gravity direct mode only supports one patch so far, current number "
                "of patches is : "
                + std::to_string(edges.sizes.indexes.get_ids().size()));
        }

        Tscal G          = edges.constant_G.data;
        Tscal gpart_mass = edges.gpart_mass.data;

        Tscal gravitational_softening = epsilon;

        auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        edges.sizes.indexes.for_each([&](u64 id, const u64 &n) {
            PatchDataField<Tvec> &xyz      = edges.field_xyz.get_field(id);
            PatchDataField<Tvec> &axyz_ext = edges.field_axyz_ext.get_field(id);

            Tvec bmax = xyz.compute_max();
            Tvec bmin = xyz.compute_min();
            shammath::AABB<Tvec> aabb(bmin, bmax);

            // build the tree
            auto bvh = RTree::make_empty(dev_sched);
            bvh.rebuild_from_positions(xyz.get_buf(), xyz.get_obj_cnt(), aabb, reduction_level);

            // compute moments in leaves
            auto mass_moments_tree = compute_tree_mass_moments<Tvec, Umorton, mm_order - 1>(
                bvh, xyz.get_buf(), gpart_mass);

            // DTT
            auto dtt_result = shamtree::clbvh_dual_tree_traversal(dev_sched, bvh, theta_crit, true);

            // M2L step
            using GravMoments = shammath::SymTensorCollection<Tscal, 1, mm_order>;
            static constexpr u32 grav_moment_terms = GravMoments::num_component;

            using MassMoments = shammath::SymTensorCollection<Tscal, 0, mm_order - 1>;
            static constexpr u32 mass_moment_terms = MassMoments::num_component;

            auto grav_moments_tree = shamtree::prepare_karras_radix_tree_field_multi_var<Tscal>(
                bvh.structure,
                shamtree::new_empty_karras_radix_tree_field_multi_var<Tscal>(grav_moment_terms));

            // we do not need to reset grav moments tree as it will be overwritten in the M2L step

            logger::raw_ln(
                "SPH", "M2L interact count: ", dtt_result.node_interactions_m2m.get_size());
            logger::raw_ln(
                "SPH", "P2P interact count: ", dtt_result.node_interactions_p2p.get_size());

            // M2L kernel
            sham::kernel_call(
                q,
                sham::MultiRef{
                    dtt_result.node_interactions_m2m,
                    dtt_result.ordered_result->offset_m2m,
                    mass_moments_tree.buf_field,
                    bvh.aabbs.buf_aabb_min,
                    bvh.aabbs.buf_aabb_max},
                sham::MultiRef{grav_moments_tree.buf_field},
                bvh.structure.get_total_cell_count(),
                [](u32 cell_id,
                   const u32_2 *m2l_interactions,
                   const u32 *offset_m2l,
                   const Tscal *mass_moments,
                   const Tvec *aabb_min,
                   const Tvec *aabb_max,
                   Tscal *grav_moments) {
                    auto load_mass_moment = [&](u32 cell_id) -> MassMoments {
                        const Tscal *mass_moment_ptr = mass_moments + cell_id * mass_moment_terms;
                        return MassMoments::load(mass_moment_ptr, 0);
                    };

                    auto load_aabb = [&](u32 cell_id) -> shammath::AABB<Tvec> {
                        return shammath::AABB<Tvec>{aabb_min[cell_id], aabb_max[cell_id]};
                    };

                    GravMoments dM_k = GravMoments::zeros();

                    shammath::AABB<Tvec> aabb_A = load_aabb(cell_id);
                    Tvec s_A                    = aabb_A.get_center();

                    for (u32 i = offset_m2l[cell_id]; i < offset_m2l[cell_id + 1]; i++) {
                        u32_2 interaction = m2l_interactions[i];
                        u32 cell_id_a     = interaction.x();
                        u32 cell_id_b     = interaction.y();
                        SHAM_ASSERT(cell_id_a == cell_id);

                        MassMoments Q_n_B = load_mass_moment(cell_id_b);

                        Tvec s_B = load_aabb(cell_id_b).get_center();

                        Tvec r_fmm = s_B - s_A;

                        auto D_n
                            = shamphys::GreenFuncGravCartesian<Tscal, 1, mm_order>::get_der_tensors(
                                r_fmm);

                        dM_k += shamphys::get_dM_mat(D_n, Q_n_B);
                    }

                    Tscal *cell_moments_ptr = grav_moments + cell_id * grav_moment_terms;
                    dM_k.store(cell_moments_ptr, 0);
                });

            // L2L step
            auto is_moment_complete = shamtree::prepare_karras_radix_tree_field<u8>(
                bvh.structure, shamtree::new_empty_karras_radix_tree_field<u8>());

            // this one will not be fully overwritten so we need to initialize it to zeros
            is_moment_complete.buf_field.fill(0_u8);

            // set the root to 1 to start the process
            is_moment_complete.buf_field.set_val_at_idx(0, 1);

            auto traverser = bvh.structure.get_structure_traverser();

            for (u32 i = 0; i < bvh.structure.tree_depth; i++) {
                sham::kernel_call(
                    q,
                    sham::MultiRef{traverser, bvh.aabbs.buf_aabb_min, bvh.aabbs.buf_aabb_max},
                    sham::MultiRef{is_moment_complete.buf_field, grav_moments_tree.buf_field},
                    bvh.structure.get_internal_cell_count(),
                    [](u32 cell_id,
                       auto tree_traverser,
                       const Tvec *aabb_min,
                       const Tvec *aabb_max,
                       u8 *is_moment_complete,
                       Tscal *grav_moments) {
                        auto load_grav_moment = [&](u32 cell_id) -> GravMoments {
                            const Tscal *grav_moment_ptr
                                = grav_moments + cell_id * grav_moment_terms;
                            return GravMoments::load(grav_moment_ptr, 0);
                        };

                        auto store_grav_moment = [&](u32 cell_id, const GravMoments &grav_moment) {
                            Tscal *grav_moment_ptr = grav_moments + cell_id * grav_moment_terms;
                            grav_moment.store(grav_moment_ptr, 0);
                        };

                        u32 left_child  = tree_traverser.get_left_child(cell_id);
                        u32 right_child = tree_traverser.get_right_child(cell_id);

                        // run only if is_moment_complete is 1
                        // at the end set children to 1
                        u8 should_compute = is_moment_complete[cell_id] == 1
                                            && is_moment_complete[left_child] == 0
                                            && is_moment_complete[right_child] == 0;

                        if (should_compute) {

                            u32 left_child  = tree_traverser.get_left_child(cell_id);
                            u32 right_child = tree_traverser.get_right_child(cell_id);

                            Tvec s_A = shammath::AABB<Tvec>{aabb_min[cell_id], aabb_max[cell_id]}
                                           .get_center();
                            Tvec s_left
                                = shammath::AABB<Tvec>{aabb_min[left_child], aabb_max[left_child]}
                                      .get_center();
                            Tvec s_right
                                = shammath::AABB<Tvec>{aabb_min[right_child], aabb_max[right_child]}
                                      .get_center();

                            // perform L2L
                            GravMoments my_moment = load_grav_moment(cell_id);

                            GravMoments left_moment  = load_grav_moment(left_child);
                            GravMoments right_moment = load_grav_moment(right_child);

                            left_moment += shamphys::offset_dM_mat(my_moment, s_A, s_left);
                            right_moment += shamphys::offset_dM_mat(my_moment, s_A, s_right);

                            store_grav_moment(left_child, left_moment);
                            store_grav_moment(right_child, right_moment);

                            is_moment_complete[left_child]  = 1;
                            is_moment_complete[right_child] = 1;
                        }
                    });
            }

            // L2P
            auto cell_it = bvh.reduced_morton_set.get_leaf_cell_iterator();
            sham::kernel_call(
                q,
                sham::MultiRef{
                    xyz.get_buf(),
                    cell_it,
                    bvh.aabbs.buf_aabb_min,
                    bvh.aabbs.buf_aabb_max,
                    grav_moments_tree.buf_field},
                sham::MultiRef{axyz_ext.get_buf()},
                bvh.structure.get_leaf_count(),
                [leaf_offset = bvh.structure.get_internal_cell_count(),
                 G](u32 ileaf,
                    const Tvec *xyz,
                    auto cell_iter,
                    const Tvec *aabb_min,
                    const Tvec *aabb_max,
                    const Tscal *grav_moments,
                    Tvec *axyz_ext) {
                    auto load_grav_moment = [&](u32 cell_id) -> GravMoments {
                        const Tscal *grav_moment_ptr = grav_moments + cell_id * grav_moment_terms;
                        return GravMoments::load(grav_moment_ptr, 0);
                    };

                    u32 cell_id      = ileaf + leaf_offset;
                    GravMoments dM_k = load_grav_moment(cell_id);

                    Tvec s_A
                        = shammath::AABB<Tvec>{aabb_min[cell_id], aabb_max[cell_id]}.get_center();

                    cell_iter.for_each_in_leaf_cell(ileaf, [&](u32 i) {
                        Tvec a_i = xyz[i] - s_A;

                        auto a_k
                            = shammath::SymTensorCollection<Tscal, 0, mm_order - 1>::from_vec(a_i);

                        axyz_ext[i] += -G
                                       * shamphys::contract_grav_moment_to_force<Tscal, mm_order>(
                                           a_k, dM_k);
                    });
                });
#if false
            {
                auto tmp = axyz_ext.get_buf().copy_to_stdvec();
                logger::raw_ln(
                    "SPH", "axyz: \n", shambase::format_array(tmp.begin(), tmp.size(), 20, "{}"));
            }

            {
                auto p2p_off = dtt_result.ordered_result->offset_p2p.copy_to_stdvec();
                logger::raw_ln(
                    "SPH",
                    "P2P offsets: \n",
                    shambase::format_array(p2p_off.begin(), p2p_off.size(), 20, "{:6}"));
            }

            {
                auto interact = dtt_result.node_interactions_p2p.copy_to_stdvec();
                logger::raw_ln(
                    "SPH",
                    "P2P interactions: \n",
                    shambase::format_array(interact.begin(), interact.size(), 20, "{}"));
            }
#endif
            // P2P
            u32 leaf_offset = bvh.structure.get_internal_cell_count();
            auto node_it
                = bvh.reduced_morton_set.get_cell_iterator(bvh.structure.buf_endrange, leaf_offset);
            sham::kernel_call(
                q,
                sham::MultiRef{
                    xyz.get_buf(),
                    node_it,
                    dtt_result.node_interactions_p2p,
                    dtt_result.ordered_result->offset_p2p},
                sham::MultiRef{axyz_ext.get_buf()},
                bvh.structure.get_total_cell_count(),
                [leaf_offset, gpart_mass, G, gravitational_softening](
                    u32 icell,
                    const Tvec *xyz,
                    auto node_iter,
                    const u32_2 *p2p_interactions,
                    const u32 *offset_p2p,
                    Tvec *axyz_ext) {
                    for (u32 j = offset_p2p[icell]; j < offset_p2p[icell + 1]; j++) {
                        u32_2 interaction = p2p_interactions[j];
                        u32 cell_id_a     = interaction.x();
                        u32 cell_id_b     = interaction.y();

                        SHAM_ASSERT(icell == cell_id_a);

                        node_iter.for_each_in_cell(cell_id_a, [&](u32 i) {
                            node_iter.for_each_in_cell(cell_id_b, [&](u32 j) {
                                Tvec R            = xyz[j] - xyz[i];
                                const Tscal r_inv = sycl::rsqrt(
                                    R.x() * R.x() + R.y() * R.y() + R.z() * R.z()
                                    + gravitational_softening);
                                axyz_ext[i] += G * gpart_mass * r_inv * r_inv * r_inv * R;
                            });
                        });
                    }
                });
#if false
            {
                auto tmp = axyz_ext.get_buf().copy_to_stdvec();
                logger::raw_ln(
                    "SPH", "axyz: \n", shambase::format_array(tmp.begin(), tmp.size(), 20, "{}"));
            }
#endif
        });
    }
} // namespace shammodels::sph::modules

template class shammodels::sph::modules::SGSFMMPlummer<f64_3, 1>;
template class shammodels::sph::modules::SGSFMMPlummer<f64_3, 2>;
template class shammodels::sph::modules::SGSFMMPlummer<f64_3, 3>;
template class shammodels::sph::modules::SGSFMMPlummer<f64_3, 4>;
template class shammodels::sph::modules::SGSFMMPlummer<f64_3, 5>;
