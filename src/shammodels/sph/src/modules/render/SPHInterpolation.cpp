// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SPHInterpolation.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/integer.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/AABB.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/render/SPHInterpolation.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"
#include <array>
#include <cmath>
#include <limits>

namespace {

    /// capacity of the per thread buffer of particles around the interpolation point
    constexpr u32 interp_buf_size = 96;

    /// the warp flushes its buffers once a thread has this many particles in it
    constexpr u32 interp_flush_threshold = 64;

    /// capacity of the per thread queue of leaves found by the traversal
    constexpr u32 interp_leaf_queue_size = 8;

    /// work group size of the column integration kernel
    constexpr u32 interp_group_size = 128;

    /**
     * @brief SPH interpolation at one point per thread, with the threads of a sub-group
     * cooperating to stay converged
     *
     * Each thread computes exactly the same operations, in the same order, as a plain
     * `rtree_for` traversal accumulating every particle whose support contains the point. Only the
     * moment at which each part of the work is done changes:
     *  - (A) the tree is traversed until every thread of the sub-group found a leaf, threads which
     *    already have one keep traversing and queue the next leaves (in traversal order),
     *  - (B) each thread then tests the particles of its oldest leaf (cheap) and buffers the ones
     *    around the point,
     *  - (C) the expensive contributions of the buffered particles are computed by the whole
     *    sub-group at once, in buffer order.
     */
    template<class Tvec, class T, class Kernel, class ParticleLooper>
    inline void interp_warp_cooperative(
        const sycl::nd_item<1> &item,
        u32 npoints,
        const Tvec *__restrict pixel_positions,
        const Tvec *__restrict xyz,
        const shambase::VecComponent<Tvec> *__restrict hpart,
        const T *__restrict partmass_val,
        const shambase::VecComponent<Tvec> *__restrict rho_part,
        const ParticleLooper &particle_looper,
        const shambase::VecComponent<Tvec> *__restrict hmax,
        T *__restrict render_field) {

        using Tscal = shambase::VecComponent<Tvec>;

        const auto &traverser = particle_looper.tree_traverser;
        const auto &tree      = traverser.tree_traverser;

        static constexpr u32 tree_depth = std::remove_cvref_t<decltype(traverser)>::tree_depth_max;

        constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

        auto sg = item.get_sub_group();

        const u32 gid     = item.get_global_linear_id();
        const bool active = gid < npoints;

        T acc = sham::VectorProperties<T>::get_zero();

        Tvec pos_render = pixel_positions[active ? gid : 0];

        // traversal stack (empty for inactive threads)
        std::array<u32, tree_depth> id_stack;
        u32 stack_cursor = tree_depth;
        if (active) {
            stack_cursor           = tree_depth - 1;
            id_stack[stack_cursor] = 0; // On a Karras tree, the root is always 0
        }

        // queue of the leaves found by the traversal, in traversal order
        u32 leaf_queue[interp_leaf_queue_size];
        u32 queue_head = 0;
        u32 queue_cnt  = 0;

        // buffer of the particles around the point, in traversal order
        u32 buf_id[interp_buf_size];
        Tscal buf_rab2[interp_buf_size];
        u32 buf_cnt = 0;

        auto flush = [&]() {
            for (u32 i = 0; i < buf_cnt; i++) {
                u32 id_b   = buf_id[i];
                Tscal rab2 = buf_rab2[i];

                Tscal h_b = hpart[id_b];

                Tscal rab = sycl::sqrt(rab2);

                // partmass * val and rho_h(partmass, h_b, hfactd) precomputed per particle
                acc += partmass_val[id_b] * Kernel::W_3d(rab, h_b) / rho_part[id_b];
            }
            buf_cnt = 0;
        };

        auto has_nodes = [&]() {
            return stack_cursor < tree_depth;
        };

        while (sycl::any_of_group(sg, has_nodes() || queue_cnt > 0 || buf_cnt > 0)) {

            // (A) traverse until every thread has a leaf or no nodes left
            while (sycl::any_of_group(sg, has_nodes() && queue_cnt == 0)) {
                if (has_nodes() && queue_cnt < interp_leaf_queue_size) {

                    // Pop the top of the stack
                    u32 current_node_id = id_stack[stack_cursor];
                    stack_cursor++;

                    Tscal rint_cell = hmax[current_node_id] * Kernel::Rkern;

                    shammath::AABB<Tvec> node_aabb{
                        traverser.aabb_min[current_node_id], traverser.aabb_max[current_node_id]};

                    if (node_aabb.expand_all(rint_cell).contains_asymmetric(pos_render)) {
                        if (tree.is_id_leaf(current_node_id)) {
                            leaf_queue[(queue_head + queue_cnt) % interp_leaf_queue_size]
                                = current_node_id;
                            queue_cnt++;
                        } else {
                            u32 lid = tree.get_left_child(current_node_id);
                            u32 rid = tree.get_right_child(current_node_id);

                            id_stack[stack_cursor - 1] = rid;
                            stack_cursor--;

                            id_stack[stack_cursor - 1] = lid;
                            stack_cursor--;
                        }
                    }
                }
            }

            // (B) test the particles of the oldest leaf
            if (queue_cnt > 0) {
                u32 leaf_node = leaf_queue[queue_head];
                queue_head    = (queue_head + 1) % interp_leaf_queue_size;
                queue_cnt--;

                u32 leaf_id = leaf_node - tree.offset_leaf;

                particle_looper.cell_iterator.for_each_in_leaf_cell(leaf_id, [&](u32 id_b) {
                    Tvec dr    = pos_render - xyz[id_b];
                    Tscal rab2 = sycl::dot(dr, dr);
                    Tscal h_b  = hpart[id_b];

                    if (rab2 > h_b * h_b * Rker2) {
                        return;
                    }

                    buf_id[buf_cnt]   = id_b;
                    buf_rab2[buf_cnt] = rab2;
                    buf_cnt++;

                    if (buf_cnt == interp_buf_size) {
                        flush();
                    }
                });
            }

            // (C) compute the buffered contributions together
            bool lane_done = !has_nodes() && queue_cnt == 0;
            if (sycl::any_of_group(
                    sg, buf_cnt >= interp_flush_threshold || (lane_done && buf_cnt > 0))) {
                flush();
            }
        }

        if (active) {
            render_field[gid] += acc;
        }
    }

} // namespace

template<class Tvec, class T, template<class> class SPHKernel>
void shammodels::sph::modules::SPHInterpolation<Tvec, T, SPHKernel>::_impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto &part_counts = edges.part_counts.indexes;

    edges.positions.check_sizes(part_counts);
    edges.h_part.check_sizes(part_counts);
    edges.field_data.check_sizes(part_counts);

    const sham::DeviceBuffer<Tvec> &interp_points_buf = edges.interp_points.value;
    sham::DeviceBuffer<T> &output_buf                 = edges.interpolated_field.value;

    u32 npoints = interp_points_buf.get_size();
    if (output_buf.get_size() != npoints) {
        output_buf.resize_discard_data(npoints);
    }
    output_buf.fill(sham::VectorProperties<T>::get_zero());

    using u_morton = u32;
    using Tree     = shamtree::CompressedLeafBVH<u_morton, Tvec, 3>;

    Tscal partmass           = edges.gpart_mass.data;
    u32 tree_reduction_level = edges.tree_reduction_level.data;
    sham::DeviceQueue &queue = shamsys::instance::get_compute_scheduler().get_queue();
    auto dev_sched           = shamsys::instance::get_compute_scheduler_ptr();

    part_counts.for_each([&](u64 id, u32 count) {
        if (count == 0) {
            return;
        }

        PatchDataField<Tvec> &pos = edges.positions.get_field(id);
        if (pos.is_empty()) {
            return;
        }

        Tvec bmax = pos.compute_max();
        Tvec bmin = pos.compute_min();

        shammath::AABB<Tvec> aabb(bmin, bmax);

        Tscal infty = std::numeric_limits<Tscal>::infinity();

        aabb.lower[0] = std::nextafter(aabb.lower[0], -infty);
        aabb.lower[1] = std::nextafter(aabb.lower[1], -infty);
        aabb.lower[2] = std::nextafter(aabb.lower[2], -infty);
        aabb.upper[0] = std::nextafter(aabb.upper[0], infty);
        aabb.upper[1] = std::nextafter(aabb.upper[1], infty);
        aabb.upper[2] = std::nextafter(aabb.upper[2], infty);

        u32 obj_cnt = pos.get_obj_cnt();

        Tree tree = Tree::make_empty(dev_sched);
        tree.rebuild_from_positions(pos.get_buf(), obj_cnt, aabb, tree_reduction_level);

        auto &hpart_span = edges.h_part.get_spans().get(id);
        auto &field_span = edges.field_data.get_spans().get(id);
        auto &buf_hpart  = hpart_span.field_ref.get_buf();
        auto &buf_field  = field_span.field_ref.get_buf();

        auto hmax_tree = shamtree::compute_tree_field_max_field<Tscal>(
            tree.structure,
            tree.reduced_morton_set.get_leaf_cell_iterator(),
            shamtree::new_empty_karras_radix_tree_field<Tscal>(),
            buf_hpart);

        auto obj_it = tree.get_object_iterator();

        // per particle quantities independent of the point (same expressions as in the direct
        // computation `partmass * val * W / rho_h(partmass, h_b, hfactd)`, hence same bits)
        sham::DeviceBuffer<T> partmass_val_buf(obj_cnt, dev_sched);
        sham::DeviceBuffer<Tscal> rho_part_buf(obj_cnt, dev_sched);

        sham::kernel_call(
            queue,
            sham::MultiRef{buf_hpart, buf_field},
            sham::MultiRef{partmass_val_buf, rho_part_buf},
            obj_cnt,
            [partmass](
                u32 id_b,
                const Tscal *__restrict hpart,
                const T *__restrict torender,
                T *__restrict partmass_val,
                Tscal *__restrict rho_part) {
                partmass_val[id_b] = partmass * torender[id_b];
                rho_part[id_b]     = shamrock::sph::rho_h(partmass, hpart[id_b], Kernel::hfactd);
            });

        u32 group_cnt     = shambase::group_count(npoints, interp_group_size);
        u32 corrected_len = group_cnt * interp_group_size;

        sham::kernel_call_hndl(
            queue,
            sham::MultiRef{
                interp_points_buf,
                pos.get_buf(),
                buf_hpart,
                partmass_val_buf,
                rho_part_buf,
                obj_it,
                hmax_tree.buf_field},
            sham::MultiRef{output_buf},
            npoints,
            [=](u32,
                const Tvec *__restrict pixel_positions,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const T *__restrict partmass_val,
                const Tscal *__restrict rho_part,
                auto particle_looper,
                const Tscal *__restrict hmax,
                T *__restrict render_field) {
                return [=](sycl::handler &cgh) {
                    cgh.parallel_for(
                        sycl::nd_range<1>{corrected_len, interp_group_size},
                        [=](sycl::nd_item<1> item) {
                            interp_warp_cooperative<Tvec, T, Kernel>(
                                item,
                                npoints,
                                pixel_positions,
                                xyz,
                                hpart,
                                partmass_val,
                                rho_part,
                                particle_looper,
                                hmax,
                                render_field);
                        });
                };
            });
    });

    shamalgs::collective::reduce_buffer_in_place_sum(output_buf, MPI_COMM_WORLD);
}

template<class Tvec, class T, template<class> class SPHKernel>
std::string shammodels::sph::modules::SPHInterpolation<Tvec, T, SPHKernel>::_impl_get_tex() const {
    return "TODO";
}

using namespace shammath;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, M4>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, M6>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, M8>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, C2>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, C4>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, C6>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, M4>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, M6>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, M8>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, C2>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, C4>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, C6>;
