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

#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/Device.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/AABB.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/render/SPHColumnInteg.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"
#include <cmath>
#include <limits>

namespace {

    // persistent kernel helpers, same as in the persistent neighbour cache (PR #2417)

    u32 get_persistent_worker_group_count(sham::DeviceProperties &dev) {
        return dev.max_compute_units * 2;
    }

    u32 get_persistent_thread_count(sham::DeviceProperties &dev) {
        return get_persistent_worker_group_count(dev)
               * (dev.type == sham::DeviceType::GPU ? 256 : 1);
    }

    template<class T>
    T global_fetch_add_relaxed(T *ptr, T offset) {
        sycl::atomic_ref<
            T,
            sycl::memory_order_relaxed,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>
            ref(*ptr);

        return ref.fetch_add(offset);
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

    // persistent kernel handling
    u32 persistent_tcount = get_persistent_thread_count(queue.get_device_prop());
    sham::DeviceBuffer<u32> work_index(1, dev_sched);

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

        // reset the work queue to 0 to start the persistent kernel
        work_index.fill(0);

        sham::kernel_call(
            queue,
            sham::MultiRef{
                rays_buf, pos.get_buf(), buf_hpart, buf_field, obj_it, hmax_tree.buf_field},
            sham::MultiRef{work_index, output_buf},
            persistent_tcount,
            [=](u32,
                const shammath::Ray<Tvec> *__restrict image_rays,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const T *__restrict torender,
                auto particle_looper,
                const Tscal *__restrict hmax,
                u32 *work_index,
                T *__restrict render_field) {
                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                // current work item state, set by next_work
                u32 ray_id = nrays;
                shammath::Ray<Tvec> ray;
                T acc;

                particle_looper.rtree_for_persistent(
                    [&]() -> bool {
                        // store the result of the finished ray (if any)
                        if (ray_id < nrays) {
                            render_field[ray_id] += acc;
                        }

                        // fetch the next ray
                        ray_id = global_fetch_add_relaxed<u32>(work_index, 1);
                        if (ray_id >= nrays) {
                            return false;
                        }

                        ray = image_rays[ray_id];
                        acc = sham::VectorProperties<T>::get_zero();
                        return true;
                    },
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        Tscal rint_cell = hmax[node_id] * Kernel::Rkern;

                        return node_aabb.expand_all(rint_cell).intersect_ray(ray);
                    },
                    [&](u32 id_b) {
                        Tvec dr = ray.origin - xyz[id_b];

                        dr -= ray.direction * sycl::dot(dr, ray.direction);

                        Tscal rab2 = sycl::dot(dr, dr);
                        Tscal h_b  = hpart[id_b];

                        if (rab2 > h_b * h_b * Rker2) {
                            return;
                        }

                        Tscal rab = sycl::sqrt(rab2);

                        T val = torender[id_b];

                        Tscal rho_b = shamrock::sph::rho_h(partmass, h_b, Kernel::hfactd);

                        acc += partmass * val * Kernel::Y_3d(rab, h_b, 4) / rho_b;
                    });
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
