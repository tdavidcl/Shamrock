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
#include "shamalgs/details/algorithm/algorithm.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/mpi.hpp"
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

        u32 hilbert_count       = pos.get_size();
        u32 hilbert_count_round = sham::roundup_pow2_clz(hilbert_count);

        using Thilbert = u64;

        sham::DeviceBuffer<Thilbert> hilbert_codes{hilbert_count_round, sched};

        sham::kernel_call(
            sched->get_queue(),
            sham::MultiRef{pos},
            sham::MultiRef{hilbert_codes},
            hilbert_count_round,
            [hilbert_coord_transf, hilbert_count](u32 i, const Tvec *pos, Thilbert *hilbert_code) {
                if (i < hilbert_count) {
                    auto pos_igrid  = hilbert_coord_transf.reverse_transform(pos[i]);
                    hilbert_code[i] = shamrock::sfc::HilbertCurve<Thilbert, 3>::icoord_to_hilbert(
                        pos_igrid.x(), pos_igrid.y(), pos_igrid.z());
                } else {
                    hilbert_code[i] = shambase::primitive_type_info<Thilbert>::max;
                }
            });

        // sort hilbert codes
        sham::DeviceBuffer<u32> object_index_map
            = shamalgs::algorithm::gen_buffer_index_usm(sched, hilbert_count_round);
        shamalgs::algorithm::sort_by_key(
            sched, hilbert_codes, object_index_map, hilbert_count_round);

        // compute cluster count
        auto round_up = [](u32 x, u32 mult) {
            if (x % mult != 0) {
                return x + mult - (x % mult);
            } else {
                return x;
            }
        };

        u32 cluster_ids_count = round_up(hilbert_count, cluster_obj_count);

        u32 cluster_count = cluster_ids_count / cluster_obj_count;

        // compute cluster ids
        sham::DeviceBuffer<u32> cluster_ids{cluster_ids_count, sched};

        sham::kernel_call(
            sched->get_queue(),
            sham::MultiRef{object_index_map},
            sham::MultiRef{cluster_ids},
            cluster_ids_count,
            [N = hilbert_count](u32 i, const u32 *idxs, u32 *cluster_ids) {
                cluster_ids[i] = (i < N) ? idxs[i] : ClusterList<cluster_obj_count>::err_id;
            });

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

    template ClusterList<1> cluster_objects<f64_3, 1>(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<f64_3> &pos,
        shammath::AABB<f64_3> bounding_box,
        ClusteringStrategy strategy,
        ClusteringOptions<f64_3> options);

    template ClusterList<8> cluster_objects<f64_3, 8>(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<f64_3> &pos,
        shammath::AABB<f64_3> bounding_box,
        ClusteringStrategy strategy,
        ClusteringOptions<f64_3> options);

    template<class Tvec, u32 cluster_obj_count>
    ClusterListAABBs<Tvec, cluster_obj_count> get_cluster_aabbs(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<Tvec> &pos,
        ClusterList<cluster_obj_count> &clusters) {

        ClusterListAABBs<Tvec, cluster_obj_count> ret{
            clusters.cluster_count,
            sham::DeviceBuffer<Tvec>{clusters.cluster_count, sched},
            sham::DeviceBuffer<Tvec>{clusters.cluster_count, sched}};

        sham::kernel_call(
            sched->get_queue(),
            sham::MultiRef{pos, clusters},
            sham::MultiRef{ret.clusters_aabb_min, ret.clusters_aabb_max},
            clusters.cluster_count,
            [](u32 cluster_id,
               const Tvec *pos,
               auto clusters,
               Tvec *cluster_aabbs_lower,
               Tvec *cluster_aabbs_upper) {
                auto cluster_ids = clusters.get_cluster_ids(cluster_id);

                shammath::AABB<Tvec> aabb_cluster
                    = cluster_ids.template get_aabb<Tvec>([&](u32 id) -> shammath::AABB<Tvec> {
                          auto r_a    = pos[id];
                          auto aabb_a = shammath::AABB<Tvec>(r_a, r_a);
                          return aabb_a;
                      });

                cluster_aabbs_lower[cluster_id] = aabb_cluster.lower;
                cluster_aabbs_upper[cluster_id] = aabb_cluster.upper;
            });

        return ret;
    }

    template ClusterListAABBs<f64_3, 1> get_cluster_aabbs(
        sham::DeviceScheduler_ptr sched, sham::DeviceBuffer<f64_3> &pos, ClusterList<1> &clusters);

    template ClusterListAABBs<f64_3, 4> get_cluster_aabbs(
        sham::DeviceScheduler_ptr sched, sham::DeviceBuffer<f64_3> &pos, ClusterList<4> &clusters);

    template ClusterListAABBs<f64_3, 8> get_cluster_aabbs(
        sham::DeviceScheduler_ptr sched, sham::DeviceBuffer<f64_3> &pos, ClusterList<8> &clusters);

    template<class Tvec, u32 cluster_obj_count>
    sham::DeviceBuffer<shambase::VecComponent<Tvec>> compute_compression_ratios(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<Tvec> &pos,
        ClusterList<cluster_obj_count> &clusters,
        sham::DeviceBuffer<shambase::VecComponent<Tvec>> &interact_radius) {

        using Tscal = shambase::VecComponent<Tvec>;
        sham::DeviceBuffer<Tscal> ret{clusters.cluster_count, sched};

        sham::kernel_call(
            sched->get_queue(),
            sham::MultiRef{pos, interact_radius, clusters},
            sham::MultiRef{ret},
            clusters.cluster_count,
            [](u32 cluster_id, const Tvec *xyz, const Tscal *rint, auto clusters, Tscal *ratios) {
                auto cluster_ids = clusters.get_cluster_ids(cluster_id);

                Tscal sum_vol_indiv = Tscal(0);

                shammath::AABB<Tvec> aabb_cluster
                    = cluster_ids.template get_aabb<Tvec>([&](u32 id) -> shammath::AABB<Tvec> {
                          auto rint_a = rint[id];
                          auto pos_a  = xyz[id];
                          auto aabb_a = shammath::AABB<Tvec>(pos_a, pos_a).expand_all(rint_a);
                          sum_vol_indiv += aabb_a.get_volume();
                          return aabb_a;
                      });

                Tscal vol_cluster = aabb_cluster.get_volume();

                ratios[cluster_id] = vol_cluster / sum_vol_indiv;
            });

        return ret;
    }

    template sham::DeviceBuffer<f64> compute_compression_ratios<f64_3, 1>(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<f64_3> &pos,
        ClusterList<1> &clusters,
        sham::DeviceBuffer<f64> &interact_radius);

    template sham::DeviceBuffer<f64> compute_compression_ratios<f64_3, 4>(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<f64_3> &pos,
        ClusterList<4> &clusters,
        sham::DeviceBuffer<f64> &interact_radius);

    template sham::DeviceBuffer<f64> compute_compression_ratios<f64_3, 8>(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<f64_3> &pos,
        ClusterList<8> &clusters,
        sham::DeviceBuffer<f64> &interact_radius);

} // namespace shamtree
