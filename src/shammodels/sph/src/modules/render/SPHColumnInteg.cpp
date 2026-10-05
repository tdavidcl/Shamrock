// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SPHColumnInteg.cpp
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
#include "shammodels/sph/modules/render/SPHColumnInteg.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"
#include <array>
#include <cmath>
#include <limits>

namespace {

    /**
     * @brief Bit identical equivalent of `Kernel::Y_3d(r, h, np)` exploiting the z symmetry
     *
     * `Y_3d` is a Riemann sum over `z = -Rkern, -Rkern + step, ...` of `f(sqrt(x^2 + z^2))`.
     * When this grid is exact and symmetric, the terms at `z` and `-z` are identical, so only
     * the samples with `z <= 0` are evaluated and the sum is then accumulated in the original
     * order, giving the same bits as `Y_3d`.
     */
    template<class Kernel, int np>
    struct ColumnZGrid {
        using Tscal = typename Kernel::Tscal;

        static constexpr Tscal start = -Kernel::Rkern;
        static constexpr Tscal end   = Kernel::Rkern;
        static constexpr Tscal step  = Kernel::Rkern / np;

        // same loop as shammath::integ_riemann_sum
        static constexpr int count() {
            int n = 0;
            for (Tscal z = start; z < end; z += step) {
                n++;
            }
            return n;
        }

        static constexpr int n = count();

        static constexpr std::array<Tscal, n> zs() {
            std::array<Tscal, n> ret{};
            int i = 0;
            for (Tscal z = start; z < end; z += step) {
                ret[i++] = z;
            }
            return ret;
        }

        static constexpr std::array<Tscal, n> z = zs();

        // index of the sample with the same z^2 and z <= 0 (-1 if there is none)
        static constexpr std::array<int, n> mirrors() {
            std::array<int, n> ret{};
            for (int i = 0; i < n; i++) {
                ret[i] = -1;
                if (z[i] <= 0) {
                    ret[i] = i;
                    continue;
                }
                for (int j = 0; j < n; j++) {
                    if (z[j] <= 0 && z[j] == -z[i]) {
                        ret[i] = j;
                    }
                }
            }
            return ret;
        }

        static constexpr std::array<int, n> mirror = mirrors();

        static constexpr bool is_symmetric() {
            for (int i = 0; i < n; i++) {
                if (mirror[i] < 0) {
                    return false;
                }
            }
            return true;
        }

        static_assert(is_symmetric(), "the Riemann grid of Y_3d must be exactly symmetric");

        static inline Tscal Y_3d(Tscal r, Tscal h) {
            Tscal x = r / h;

            // the original loop compiles `x * x + z * z` to fma(z, z, x * x), match it explicitly
            Tscal xx = x * x;

            Tscal fz[n];
#pragma unroll
            for (int i = 0; i < n; i++) {
                if (z[i] <= 0) {
                    fz[i] = Kernel::f(sqrt(sycl::fma(z[i], z[i], xx)));
                }
            }

            // same accumulation as shammath::integ_riemann_sum
            Tscal acc = {};
#pragma unroll
            for (int i = 0; i < n; i++) {
                acc += fz[mirror[i]] * step;
            }

            return Kernel::Generator::norm_3d * acc / (h * h);
        }
    };

    /// capacity of the per thread buffer of particles intersecting the ray
    constexpr u32 column_buf_size = 96;

    /// the warp flushes its buffers once a thread has this many particles in it
    constexpr u32 column_flush_threshold = 64;

    /// capacity of the per thread queue of leaves found by the traversal
    constexpr u32 column_leaf_queue_size = 8;

    /// work group size of the column integration kernel
    constexpr u32 column_group_size = 128;

    /**
     * @brief Column integration of one ray per thread, with the threads of a sub-group
     * cooperating to stay converged
     *
     * Each thread computes exactly the same operations, in the same order, as a plain
     * `rtree_for` traversal accumulating every particle intersecting the ray. Only the moment at
     * which each part of the work is done changes:
     *  - (A) the tree is traversed until every thread of the sub-group found a leaf, threads which
     *    already have one keep traversing and queue the next leaves (in traversal order),
     *  - (B) each thread then tests the particles of its oldest leaf (cheap) and buffers the ones
     *    intersecting the ray,
     *  - (C) the expensive contributions of the buffered particles are computed by the whole
     *    sub-group at once, in buffer order.
     */
    template<class Tvec, class T, class Kernel, class ParticleLooper>
    inline void column_integ_warp_cooperative(
        const sycl::nd_item<1> &item,
        u32 nrays,
        shambase::VecComponent<Tvec> partmass,
        const shammath::Ray<Tvec> *__restrict image_rays,
        const Tvec *__restrict xyz,
        const shambase::VecComponent<Tvec> *__restrict hpart,
        const T *__restrict torender,
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
        const bool active = gid < nrays;

        T acc = sham::VectorProperties<T>::get_zero();

        shammath::Ray<Tvec> ray = image_rays[active ? gid : 0];

        // traversal stack (empty for inactive threads)
        std::array<u32, tree_depth> id_stack;
        u32 stack_cursor = tree_depth;
        if (active) {
            stack_cursor           = tree_depth - 1;
            id_stack[stack_cursor] = 0; // On a Karras tree, the root is always 0
        }

        // queue of the leaves found by the traversal, in traversal order
        u32 leaf_queue[column_leaf_queue_size];
        u32 queue_head = 0;
        u32 queue_cnt  = 0;

        // buffer of the particles intersecting the ray, in traversal order
        u32 buf_id[column_buf_size];
        Tscal buf_rab2[column_buf_size];
        u32 buf_cnt = 0;

        auto flush = [&]() {
            for (u32 i = 0; i < buf_cnt; i++) {
                u32 id_b   = buf_id[i];
                Tscal rab2 = buf_rab2[i];

                Tscal h_b = hpart[id_b];

                Tscal rab = sycl::sqrt(rab2);

                T val = torender[id_b];

                Tscal rho_b = shamrock::sph::rho_h(partmass, h_b, Kernel::hfactd);

                acc += partmass * val * ColumnZGrid<Kernel, 4>::Y_3d(rab, h_b) / rho_b;
            }
            buf_cnt = 0;
        };

        auto has_nodes = [&]() {
            return stack_cursor < tree_depth;
        };

        while (sycl::any_of_group(sg, has_nodes() || queue_cnt > 0 || buf_cnt > 0)) {

            // (A) traverse until every thread has a leaf or no nodes left
            while (sycl::any_of_group(sg, has_nodes() && queue_cnt == 0)) {
                if (has_nodes() && queue_cnt < column_leaf_queue_size) {

                    // Pop the top of the stack
                    u32 current_node_id = id_stack[stack_cursor];
                    stack_cursor++;

                    Tscal rint_cell = hmax[current_node_id] * Kernel::Rkern;

                    shammath::AABB<Tvec> node_aabb{
                        traverser.aabb_min[current_node_id], traverser.aabb_max[current_node_id]};

                    if (node_aabb.expand_all(rint_cell).intersect_ray(ray)) {
                        if (tree.is_id_leaf(current_node_id)) {
                            leaf_queue[(queue_head + queue_cnt) % column_leaf_queue_size]
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
                queue_head    = (queue_head + 1) % column_leaf_queue_size;
                queue_cnt--;

                u32 leaf_id = leaf_node - tree.offset_leaf;

                particle_looper.cell_iterator.for_each_in_leaf_cell(leaf_id, [&](u32 id_b) {
                    Tvec dr = ray.origin - xyz[id_b];

                    dr -= ray.direction * sycl::dot(dr, ray.direction);

                    Tscal rab2 = sycl::dot(dr, dr);
                    Tscal h_b  = hpart[id_b];

                    if (rab2 > h_b * h_b * Rker2) {
                        return;
                    }

                    buf_id[buf_cnt]   = id_b;
                    buf_rab2[buf_cnt] = rab2;
                    buf_cnt++;

                    if (buf_cnt == column_buf_size) {
                        flush();
                    }
                });
            }

            // (C) compute the buffered contributions together
            bool lane_done = !has_nodes() && queue_cnt == 0;
            if (sycl::any_of_group(
                    sg, buf_cnt >= column_flush_threshold || (lane_done && buf_cnt > 0))) {
                flush();
            }
        }

        if (active) {
            render_field[gid] += acc;
        }
    }

} // namespace

template<class Tvec, class T, template<class> class SPHKernel>
void shammodels::sph::modules::SPHColumnInteg<Tvec, T, SPHKernel>::_impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto &part_counts = edges.part_counts.indexes;

    edges.positions.check_sizes(part_counts);
    edges.h_part.check_sizes(part_counts);
    edges.field_data.check_sizes(part_counts);

    const sham::DeviceBuffer<shammath::Ray<Tvec>> &rays_buf = edges.rays.value;
    sham::DeviceBuffer<T> &output_buf                       = edges.interpolated_field.value;

    u32 nrays = rays_buf.get_size();
    if (output_buf.get_size() != nrays) {
        output_buf.resize_discard_data(nrays);
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

        u32 group_cnt     = shambase::group_count(nrays, column_group_size);
        u32 corrected_len = group_cnt * column_group_size;

        sham::kernel_call_hndl(
            queue,
            sham::MultiRef{
                rays_buf, pos.get_buf(), buf_hpart, buf_field, obj_it, hmax_tree.buf_field},
            sham::MultiRef{output_buf},
            nrays,
            [=](u32,
                const shammath::Ray<Tvec> *__restrict image_rays,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const T *__restrict torender,
                auto particle_looper,
                const Tscal *__restrict hmax,
                T *__restrict render_field) {
                return [=](sycl::handler &cgh) {
                    cgh.parallel_for(
                        sycl::nd_range<1>{corrected_len, column_group_size},
                        [=](sycl::nd_item<1> item) {
                            column_integ_warp_cooperative<Tvec, T, Kernel>(
                                item,
                                nrays,
                                partmass,
                                image_rays,
                                xyz,
                                hpart,
                                torender,
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
std::string shammodels::sph::modules::SPHColumnInteg<Tvec, T, SPHKernel>::_impl_get_tex() const {
    return "TODO";
}

using namespace shammath;
template class shammodels::sph::modules::SPHColumnInteg<f64_3, f64, M4>;
template class shammodels::sph::modules::SPHColumnInteg<f64_3, f64, M6>;
template class shammodels::sph::modules::SPHColumnInteg<f64_3, f64, M8>;
template class shammodels::sph::modules::SPHColumnInteg<f64_3, f64, C2>;
template class shammodels::sph::modules::SPHColumnInteg<f64_3, f64, C4>;
template class shammodels::sph::modules::SPHColumnInteg<f64_3, f64, C6>;
template class shammodels::sph::modules::SPHColumnInteg<f64_3, f64_3, M4>;
template class shammodels::sph::modules::SPHColumnInteg<f64_3, f64_3, M6>;
template class shammodels::sph::modules::SPHColumnInteg<f64_3, f64_3, M8>;
template class shammodels::sph::modules::SPHColumnInteg<f64_3, f64_3, C2>;
template class shammodels::sph::modules::SPHColumnInteg<f64_3, f64_3, C4>;
template class shammodels::sph::modules::SPHColumnInteg<f64_3, f64_3, C6>;
