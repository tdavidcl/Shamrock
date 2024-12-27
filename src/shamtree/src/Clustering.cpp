// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Clustering.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shamtree/Clustering.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/sfc/hilbert.hpp"
#include "shammath/sfc/morton.hpp"

namespace shamtree {

    template<class Tvec, u32 cluster_obj_count>
    ClusterList<cluster_obj_count> cluster_objects_hilbert(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<Tvec> &pos,
        shammath::AABB<Tvec> pos_bounding_box,
        ClusteringOptions<Tvec> options) {

        // generate Hilbert codes
        auto hilbert_coord_transf = shamrock::sfc::MortonConverter<u64, Tvec, 3>::get_transform(
            pos_bounding_box.lower, pos_bounding_box.upper);

        sham::DeviceBuffer<u64> hilbert_codes{pos.get_size(), sched};

        sham::kernel_call(
            sched->get_queue(),
            sham::MultiRef{pos},
            sham::MultiRef{hilbert_codes},
            pos.get_size(),
            [hilbert_coord_transf](u32 i, Tvec &pos, u64 &hilbert_code) {
                auto pos_igrid = hilbert_coord_transf.reverse_transform(pos);
                hilbert_code   = shamrock::sfc::HilbertCurve<u64, 3>::icoord_to_hilbert(
                    pos_igrid.x(), pos_igrid.y(), pos_igrid.z());
            });

        // sort hilbert codes
        sham::DeviceBuffer<u32> object_index_map(pos.get_size(), sched);

        sham::kernel_call(
            sched->get_queue(),
            sham::MultiRef{},
            sham::MultiRef{object_index_map},
            pos.get_size(),
            [](u32 i, u32 &idx) {
                idx = i;
            });

        // sycl_sort_morton_key_pair(
        //     queue, morton_len, out_buf_particle_index_map, out_buf_morton);
        shambase::throw_unimplemented();

        // compute cluster count
        u32 cluster_count = pos.get_size() / cluster_obj_count;
        if (pos.get_size() % cluster_obj_count != 0) {
            cluster_count++;
        }

        u32 cluster_ids_count = cluster_count * cluster_obj_count;

        // compute cluster ids
        sham::DeviceBuffer<u32> cluster_ids{cluster_ids_count, sched};

        shambase::throw_unimplemented();

        // generate return object
        return ClusterList<cluster_obj_count>{cluster_count, std::move(cluster_ids)};
    }

    template<class Tvec, u32 cluster_count>
    ClusterList<cluster_count> cluster_objects(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<Tvec> &pos,
        shammath::AABB<Tvec> pos_bounding_box,
        ClusteringStrategy strategy,
        ClusteringOptions<Tvec> options) {

        if (strategy == ClusteringStrategy::Hilbert) {
            return cluster_objects_hilbert<Tvec, cluster_count>(
                sched, pos, pos_bounding_box, options);
        } else {
            shambase::throw_unimplemented("Clustering strategy not implemented");
        }

        // just to shut up the compiler
        return ClusterList<cluster_count>{0, sham::DeviceBuffer<u32>{0, sched}};
    }

    template ClusterList<4> cluster_objects<f64_3, 4>(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<f64_3> &pos,
        shammath::AABB<f64_3> bounding_box,
        ClusteringStrategy strategy,
        ClusteringOptions<f64_3> options);

} // namespace shamtree
