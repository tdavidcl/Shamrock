// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SPHAzymuthalInteg.cpp
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
#include "shammodels/sph/modules/render/SPHAzymuthalInteg.hpp"
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
    struct IntegZGrid {
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

        static constexpr Tscal Rkern2 = Kernel::Rkern * Kernel::Rkern;

        // Rkern^2 must be exact for sqrt(Rkern^2) == Rkern (Rkern has a short dyadic expansion)
        static_assert(
            Kernel::Rkern * 8 == Tscal(i64(Kernel::Rkern * 8)) && Kernel::Rkern < 64,
            "Rkern must be a multiple of 1/8 for Rkern^2 to be exact");

        static inline Tscal Y_3d(Tscal r, Tscal h) {
            Tscal x = r / h;

            // the original loop compiles `x * x + z * z` to fma(z, z, x * x), match it explicitly
            Tscal xx = x * x;

            Tscal fz[n];
#pragma unroll
            for (int i = 0; i < n; i++) {
                if (z[i] <= 0) {
                    if (z[i] == start) {
                        // y >= Rkern^2 always (or NaN), see below
                        fz[i] = 0;
                    } else {
                        Tscal y = sycl::fma(z[i], z[i], xx);
                        // f(q) is exactly 0 for q >= Rkern (and for NaN) and sqrt(y) >= Rkern
                        // whenever y >= Rkern^2, so these samples are exactly 0
                        fz[i] = (y < Rkern2) ? Kernel::f(sqrt(y)) : Tscal{0};
                    }
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

    /// capacity of the per thread buffer of particles intersecting the ring
    constexpr u32 azymuthal_buf_size = 192;

    /// the warp flushes its buffers once a thread has this many particles in it
    constexpr u32 azymuthal_flush_threshold = 160;

    /// capacity of the per thread queue of leaves found by the traversal
    constexpr u32 azymuthal_leaf_queue_size = 16;

    /// work group size of the column integration kernel
    constexpr u32 azymuthal_group_size = 128;

    /**
     * @brief Azymuthal integration of one ring ray per thread, with the threads of a sub-group
     * cooperating to stay converged
     *
     * Each thread computes exactly the same operations, in the same order, as a plain
     * `rtree_for` traversal accumulating every particle intersecting the ring. Only the moment at
     * which each part of the work is done changes:
     *  - (A) the tree is traversed until every thread of the sub-group found a leaf, threads which
     *    already have one keep traversing and queue the next leaves (in traversal order),
     *  - (B) each thread then tests the particles of its oldest leaf (cheap) and buffers the ones
     *    intersecting the ring,
     *  - (C) the expensive contributions of the buffered particles are computed by the whole
     *    sub-group at once, in buffer order.
     */
    template<class Tvec, class T, class Kernel, class ParticleLooper>
    inline void azymuthal_integ_warp_cooperative(
        const sycl::nd_item<1> &item,
        u32 nring_rays,
        const shammath::RingRay<Tvec> *__restrict ring_rays_ptr,
        const Tvec *__restrict xyz,
        const shambase::VecComponent<Tvec> *__restrict hpart,
        const T *__restrict partmass_val,
        const shambase::VecComponent<Tvec> *__restrict rho_part,
        const f32 *__restrict hsupport_up,
        const ParticleLooper &particle_looper,
        const shambase::VecComponent<Tvec> *__restrict hmax,
        T *__restrict render_field,
        u32 *__restrict staging_id,
        u32 *__restrict staging_owner,
        T *__restrict staging_term,
        u32 *__restrict staging_ok,
        Tvec *__restrict ring_centers,
        Tvec *__restrict ring_exs,
        Tvec *__restrict ring_eys,
        Tvec *__restrict ring_ezs,
        shambase::VecComponent<Tvec> *__restrict ring_radii) {

        using Tscal = shambase::VecComponent<Tvec>;

        const auto &traverser = particle_looper.tree_traverser;
        const auto &tree      = traverser.tree_traverser;

        static constexpr u32 tree_depth = std::remove_cvref_t<decltype(traverser)>::tree_depth_max;

        constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

        auto sg = item.get_sub_group();

        const u32 gid     = item.get_global_linear_id();
        const bool active = gid < nring_rays;

        T acc = sham::VectorProperties<T>::get_zero();

        shammath::RingRay<Tvec> ring_ray = ring_rays_ptr[active ? gid : 0];
        Tvec ez                          = ring_ray.get_ez();

        // traversal stack (empty for inactive threads)
        std::array<u32, tree_depth> id_stack;
        u32 stack_cursor = tree_depth;
        if (active) {
            stack_cursor           = tree_depth - 1;
            id_stack[stack_cursor] = 0; // On a Karras tree, the root is always 0
        }

        // queue of the leaves found by the traversal, in traversal order
        u32 leaf_queue[azymuthal_leaf_queue_size];
        u32 queue_head = 0;
        u32 queue_cnt  = 0;

        // buffer of the candidate particles (that may intersect the ring), in traversal order
        u32 buf_id[azymuthal_buf_size];
        u32 buf_cnt = 0;

        // Exact test + contribution of the particle id_b for the ring (center, e_x, e_y, ez,
        // radius), this is exactly the code of the direct computation. Returns false if the
        // particle does not intersect the ring
        auto contrib = [&](u32 id_b,
                           const Tvec &center,
                           const Tvec &e_x,
                           const Tvec &e_y,
                           const Tvec &e_z,
                           Tscal radius,
                           T &ret) -> bool {
            Tvec r_center = center - xyz[id_b];

            Tscal z_val = sycl::dot(r_center, e_z);
            Tscal x_val = sycl::dot(r_center, e_x);
            Tscal y_val = sycl::dot(r_center, e_y);
            Tscal r_val = sycl::sqrt(x_val * x_val + y_val * y_val);

            Tscal delta_r = r_val - radius;

            Tscal rab2_ring = z_val * z_val + delta_r * delta_r;
            Tscal h_b       = hpart[id_b];

            if (rab2_ring > h_b * h_b * Rker2) {
                return false;
            }

            Tscal rab = sycl::sqrt(rab2_ring);

            // partmass * val and rho_h(partmass, h_b, hfactd) precomputed per particle
            // TODO: account for curvature
            ret = partmass_val[id_b] * IntegZGrid<Kernel, 4>::Y_3d(rab, h_b) / rho_part[id_b];
            return true;
        };

        // flush computed by this thread only (only used if the buffer is full within a leaf)
        auto flush = [&]() {
            for (u32 i = 0; i < buf_cnt; i++) {
                T term;
                if (contrib(
                        buf_id[i],
                        ring_ray.center,
                        ring_ray.e_x,
                        ring_ray.e_y,
                        ez,
                        ring_ray.radius,
                        term)) {
                    acc += term;
                }
            }
            buf_cnt = 0;
        };

        const u32 sg_size = sg.get_local_range()[0];
        const u32 lane    = sg.get_local_linear_id();
        const u32 st_base = sg.get_group_linear_id() * sg_size;

        ring_centers[st_base + lane] = ring_ray.center;
        ring_exs[st_base + lane]     = ring_ray.e_x;
        ring_eys[st_base + lane]     = ring_ray.e_y;
        ring_ezs[st_base + lane]     = ez;
        ring_radii[st_base + lane]   = ring_ray.radius;

        // fp32 copies of the ring frame for the conservative prefilter, and the Lipschitz
        // constant L = sqrt(|e_x|^2 + |e_y|^2 + |e_z|^2) of the distance to the ring with
        // respect to the position (rounded up)
        sycl::vec<f32, 3> exf{f32(ring_ray.e_x.x()), f32(ring_ray.e_x.y()), f32(ring_ray.e_x.z())};
        sycl::vec<f32, 3> eyf{f32(ring_ray.e_y.x()), f32(ring_ray.e_y.y()), f32(ring_ray.e_y.z())};
        sycl::vec<f32, 3> ezf{f32(ez.x()), f32(ez.y()), f32(ez.z())};
        f32 radius_f     = f32(ring_ray.radius);
        f32 lip_f        = f32(sycl::sqrt(
                               sycl::dot(ring_ray.e_x, ring_ray.e_x)
                               + sycl::dot(ring_ray.e_y, ring_ray.e_y) + sycl::dot(ez, ez)))
                           * (1.f + 1.f / 1048576.f);
        f32 radius_abs_f = sycl::fabs(radius_f);

        // flush of the whole sub-group: the buffered candidates of all threads are computed by
        // all the threads (sg_size entries per round, through the staging area), then each
        // thread adds its own results in its buffer order
        auto flush_cooperative = [&]() {
            u32 off = sycl::exclusive_scan_over_group(sg, buf_cnt, sycl::plus<u32>{});
            u32 tot = sycl::reduce_over_group(sg, buf_cnt, sycl::plus<u32>{});

            for (u32 r0 = 0; r0 < tot; r0 += sg_size) {

                // range of my entries in this round
                u32 jbeg = (off < r0) ? sycl::min(r0 - off, buf_cnt) : 0;
                u32 jend = (off + buf_cnt > r0 + sg_size)
                               ? ((r0 + sg_size > off) ? r0 + sg_size - off : 0)
                               : buf_cnt;

                for (u32 j = jbeg; j < jend; j++) {
                    u32 w                      = off + j - r0;
                    staging_id[st_base + w]    = buf_id[j];
                    staging_owner[st_base + w] = lane;
                }

                sycl::group_barrier(sg);

                if (r0 + lane < tot) {
                    u32 owner = st_base + staging_owner[st_base + lane];
                    T term;
                    bool ok = contrib(
                        staging_id[st_base + lane],
                        ring_centers[owner],
                        ring_exs[owner],
                        ring_eys[owner],
                        ring_ezs[owner],
                        ring_radii[owner],
                        term);
                    staging_term[st_base + lane] = term;
                    staging_ok[st_base + lane]   = ok;
                }

                sycl::group_barrier(sg);

                for (u32 j = jbeg; j < jend; j++) {
                    u32 w = st_base + off + j - r0;
                    if (staging_ok[w]) {
                        acc += staging_term[w];
                    }
                }

                sycl::group_barrier(sg);
            }

            buf_cnt = 0;
        };

        auto has_nodes = [&]() {
            return stack_cursor < tree_depth;
        };

        while (sycl::any_of_group(sg, has_nodes() || queue_cnt > 0 || buf_cnt > 0)) {

            // (A) traverse until every thread has a leaf or no nodes left
            while (sycl::any_of_group(sg, has_nodes() && queue_cnt == 0)) {
                if (has_nodes() && queue_cnt < azymuthal_leaf_queue_size) {

                    // Pop the top of the stack
                    u32 current_node_id = id_stack[stack_cursor];
                    stack_cursor++;

                    Tscal rint_cell = hmax[current_node_id] * Kernel::Rkern;

                    shammath::AABB<Tvec> node_aabb{
                        traverser.aabb_min[current_node_id], traverser.aabb_max[current_node_id]};

                    if (node_aabb.expand_all(rint_cell).intersect_ring_ray_approx(ring_ray)) {
                        if (tree.is_id_leaf(current_node_id)) {
                            leaf_queue[(queue_head + queue_cnt) % azymuthal_leaf_queue_size]
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
                queue_head    = (queue_head + 1) % azymuthal_leaf_queue_size;
                queue_cnt--;

                u32 leaf_id = leaf_node - tree.offset_leaf;

                particle_looper.cell_iterator.for_each_in_leaf_cell(leaf_id, [&](u32 id_b) {
                    // same fp64 operation as the exact test (A = center - x_b)
                    Tvec a = ring_ray.center - xyz[id_b];

                    // Conservative fp32 prefilter: rejects only particles that the exact test
                    // rejects. With e = 2^-24, g(A) = |(sqrt(x^2 + y^2) - R, z)| the distance to
                    // the ring ((x, y, z) = (A.e_x, A.e_y, A.e_z)) is L-Lipschitz in A, and the
                    // fp32 evaluation from af = f32(A) differs from g(A) by at most
                    // Eg = e (16 L |af|_1 + 8 |R|) (rounding of A, of the frame, of the dot
                    // products, of the sqrt and of r - R). So
                    // g2_32 > (sqrt(H) + Eg)^2 (1 + 16 e) + 2e-15 (L |af|_1 + |R|)^2 (the later
                    // bounding the fp64 rounding of the exact test) implies that the fp64 test
                    // rejects the particle. NaN / inf never reject.
                    sycl::vec<f32, 3> af{f32(a.x()), f32(a.y()), f32(a.z())};
                    f32 an = sycl::fabs(af.x()) + sycl::fabs(af.y()) + sycl::fabs(af.z());

                    f32 xf  = sycl::dot(af, exf);
                    f32 yf  = sycl::dot(af, eyf);
                    f32 zf  = sycl::dot(af, ezf);
                    f32 rf  = sycl::sqrt(xf * xf + yf * yf);
                    f32 drf = rf - radius_f;
                    f32 g2f = zf * zf + drf * drf;

                    constexpr f32 e = 1.f / 16777216.f; // 2^-24

                    f32 scale = lip_f * an + radius_abs_f;
                    f32 eg    = e * (16.f * lip_f * an + 8.f * radius_abs_f);
                    f32 lim   = hsupport_up[id_b] + eg;

                    if (an < 1e15f
                        && g2f > lim * lim * (1.f + 16.f * e) + 2e-15f * scale * scale + 1e-30f) {
                        return;
                    }

                    buf_id[buf_cnt] = id_b;
                    buf_cnt++;

                    if (buf_cnt == azymuthal_buf_size) {
                        flush();
                    }
                });
            }

            // (C) compute the buffered contributions together
            bool lane_done = !has_nodes() && queue_cnt == 0;
            if (sycl::any_of_group(
                    sg, buf_cnt >= azymuthal_flush_threshold || (lane_done && buf_cnt > 0))) {
                flush_cooperative();
            }
        }

        if (active) {
            render_field[gid] += acc;
        }
    }

} // namespace

template<class Tvec, class T, template<class> class SPHKernel>
void shammodels::sph::modules::SPHAzymuthalInteg<Tvec, T, SPHKernel>::_impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto &part_counts = edges.part_counts.indexes;

    edges.positions.check_sizes(part_counts);
    edges.h_part.check_sizes(part_counts);
    edges.field_data.check_sizes(part_counts);

    const sham::DeviceBuffer<shammath::RingRay<Tvec>> &ring_rays_buf = edges.ring_rays.value;
    sham::DeviceBuffer<T> &output_buf = edges.interpolated_field.value;

    u32 nring_rays = ring_rays_buf.get_size();
    if (output_buf.get_size() != nring_rays) {
        output_buf.resize_discard_data(nring_rays);
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

        // per particle quantities independent of the ring ray (same expressions as in the direct
        // computation `partmass * val * Y / rho_h(partmass, h_b, hfactd)`, hence same bits)
        sham::DeviceBuffer<T> partmass_val_buf(obj_cnt, dev_sched);
        sham::DeviceBuffer<Tscal> rho_part_buf(obj_cnt, dev_sched);
        sham::DeviceBuffer<f32> hsupport_up_buf(obj_cnt, dev_sched);

        sham::kernel_call(
            queue,
            sham::MultiRef{buf_hpart, buf_field},
            sham::MultiRef{partmass_val_buf, rho_part_buf, hsupport_up_buf},
            obj_cnt,
            [partmass](
                u32 id_b,
                const Tscal *__restrict hpart,
                const T *__restrict torender,
                T *__restrict partmass_val,
                Tscal *__restrict rho_part,
                f32 *__restrict hsupport_up) {
                Tscal h_b = hpart[id_b];

                partmass_val[id_b] = partmass * torender[id_b];
                rho_part[id_b]     = shamrock::sph::rho_h(partmass, h_b, Kernel::hfactd);

                // fp32 upper bound of sqrt(h_b * h_b * Rker2) (the exact test threshold)
                hsupport_up[id_b] = f32(h_b * Kernel::Rkern) * (1.f + 1.f / 1048576.f);
            });

        u32 group_cnt     = shambase::group_count(nring_rays, azymuthal_group_size);
        u32 corrected_len = group_cnt * azymuthal_group_size;

        sham::kernel_call_hndl(
            queue,
            sham::MultiRef{
                ring_rays_buf,
                pos.get_buf(),
                buf_hpart,
                partmass_val_buf,
                rho_part_buf,
                hsupport_up_buf,
                obj_it,
                hmax_tree.buf_field},
            sham::MultiRef{output_buf},
            nring_rays,
            [=](u32,
                const shammath::RingRay<Tvec> *__restrict ring_rays_ptr,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const T *__restrict partmass_val,
                const Tscal *__restrict rho_part,
                const f32 *__restrict hsupport_up,
                auto particle_looper,
                const Tscal *__restrict hmax,
                T *__restrict render_field) {
                return [=](sycl::handler &cgh) {
                    sycl::local_accessor<u32> staging_id{azymuthal_group_size, cgh};
                    sycl::local_accessor<u32> staging_owner{azymuthal_group_size, cgh};
                    sycl::local_accessor<T> staging_term{azymuthal_group_size, cgh};
                    sycl::local_accessor<u32> staging_ok{azymuthal_group_size, cgh};
                    sycl::local_accessor<Tvec> ring_centers{azymuthal_group_size, cgh};
                    sycl::local_accessor<Tvec> ring_exs{azymuthal_group_size, cgh};
                    sycl::local_accessor<Tvec> ring_eys{azymuthal_group_size, cgh};
                    sycl::local_accessor<Tvec> ring_ezs{azymuthal_group_size, cgh};
                    sycl::local_accessor<Tscal> ring_radii{azymuthal_group_size, cgh};

                    cgh.parallel_for(
                        sycl::nd_range<1>{corrected_len, azymuthal_group_size},
                        [=](sycl::nd_item<1> item) {
                            azymuthal_integ_warp_cooperative<Tvec, T, Kernel>(
                                item,
                                nring_rays,
                                ring_rays_ptr,
                                xyz,
                                hpart,
                                partmass_val,
                                rho_part,
                                hsupport_up,
                                particle_looper,
                                hmax,
                                render_field,
                                &(staging_id[0]),
                                &(staging_owner[0]),
                                &(staging_term[0]),
                                &(staging_ok[0]),
                                &(ring_centers[0]),
                                &(ring_exs[0]),
                                &(ring_eys[0]),
                                &(ring_ezs[0]),
                                &(ring_radii[0]));
                        });
                };
            });
    });

    shamalgs::collective::reduce_buffer_in_place_sum(output_buf, MPI_COMM_WORLD);
}

template<class Tvec, class T, template<class> class SPHKernel>
std::string shammodels::sph::modules::SPHAzymuthalInteg<Tvec, T, SPHKernel>::_impl_get_tex() const {
    return "TODO";
}

using namespace shammath;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, M4>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, M6>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, M8>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, C2>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, C4>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, C6>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, M4>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, M6>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, M8>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, C2>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, C4>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, C6>;
