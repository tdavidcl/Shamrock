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

        sham::kernel_call(
            queue,
            sham::MultiRef{
                ring_rays_buf, pos.get_buf(), buf_hpart, buf_field, obj_it, hmax_tree.buf_field},
            sham::MultiRef{output_buf},
            nring_rays,
            [=](u32 gid,
                const shammath::RingRay<Tvec> *__restrict ring_rays_ptr,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const T *__restrict torender,
                auto particle_looper,
                const Tscal *__restrict hmax,
                T *__restrict render_field) {
                T acc = sham::VectorProperties<T>::get_zero();

                shammath::RingRay<Tvec> ring_ray = ring_rays_ptr[gid];
                Tvec ez                          = ring_ray.get_ez();

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                particle_looper.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        Tscal rint_cell = hmax[node_id] * Kernel::Rkern;

                        return node_aabb.expand_all(rint_cell).intersect_ring_ray_approx(ring_ray);
                    },
                    [&](u32 id_b) {
                        Tvec r_center = ring_ray.center - xyz[id_b];

                        Tscal z_val = sycl::dot(r_center, ez);
                        Tscal x_val = sycl::dot(r_center, ring_ray.e_x);
                        Tscal y_val = sycl::dot(r_center, ring_ray.e_y);
                        Tscal r_val = sycl::sqrt(x_val * x_val + y_val * y_val);

                        Tscal delta_r = r_val - ring_ray.radius;

                        Tscal rab2_ring = z_val * z_val + delta_r * delta_r;
                        Tscal h_b       = hpart[id_b];

                        if (rab2_ring > h_b * h_b * Rker2) {
                            return;
                        }

                        Tscal rab = sycl::sqrt(rab2_ring);

                        T val = torender[id_b];

                        Tscal rho_b = shamrock::sph::rho_h(partmass, h_b, Kernel::hfactd);

                        // TODO: account for curvature
                        acc += partmass * val * IntegZGrid<Kernel, 4>::Y_3d(rab, h_b) / rho_b;
                    });

                render_field[gid] += acc;
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
