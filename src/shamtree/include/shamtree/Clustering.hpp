// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Clustering.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambase/optional.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/AABB.hpp"
#include <array>
namespace shamtree {

    template<u32 cluster_obj_count>
    struct ClusterList {
        static constexpr u32 err_id = shambase::primitive_type_info<u32>::max;

        u32 cluster_count;
        sham::DeviceBuffer<u32> cluster_ids;

        struct accessed_ro {
            const u32 *cluster_ids;

            struct id_list {
                std::array<u32, cluster_obj_count> ids;

                template<class Lambda>
                void for_each_object(const Lambda &&func) {
                    // clang-format off
                    #pragma unroll cluster_obj_count
                    for (u32 i = 0; i < cluster_obj_count; i++) {
                        if(ids[i] != err_id) func(ids[i]);
                    }
                    // clang-format on
                }

                template<class Tvec, class AABBGetter>
                shammath::AABB<Tvec> get_aabb(AABBGetter &&get_id_aabb) {

                    shammath::AABB<Tvec> ret = shammath::AABB<Tvec>::empty();
                    for_each_object([&](u32 id) {
                        ret.include(get_id_aabb(id));
                    });

                    return ret;
                }
            };

            id_list get_cluster_ids(u32 cluster_id) {
                // clang-format off
                std::array<u32, cluster_obj_count> ret;
                #pragma unroll cluster_obj_count
                for (u32 i = 0; i < cluster_obj_count; i++) {
                    ret[i] = cluster_ids[cluster_id * cluster_obj_count + i];
                }
                // clang-format on
                return id_list{ret};
            }

            u32 get_cluster_obj_id(u32 cluster_id, u32 internal_cluster_id) {
                return cluster_ids[cluster_id * cluster_obj_count + internal_cluster_id];
            }

            template<class T, class Lambda>
            T get_cluster_obj_value(u32 cluster_id, u32 internal_cluster_id, Lambda &&get_value) {
                u32 id = get_cluster_obj_id(cluster_id, internal_cluster_id);
                return (id != err_id) ? get_value(id) : T();
            }

            template<class T>
            T load_cluster_obj_value(u32 cluster_id, u32 internal_cluster_id, const T *field) {
                return get_cluster_obj_value(cluster_id, internal_cluster_id, [&](u32 id) {
                    return field[id];
                });
            }

            template<class T>
            void
            store_cluster_obj_value(u32 cluster_id, u32 internal_cluster_id, T *field, T value) {
                u32 id = get_cluster_obj_id(cluster_id, internal_cluster_id);
                if (id != err_id)
                    field[id] = value;
            }
        };

        accessed_ro get_read_access(sham::EventList &depends_list) {
            return {cluster_ids.get_read_access(depends_list)};
        }

        void complete_event_state(sycl::event e) { cluster_ids.complete_event_state(e); }
    };

    ///////////////////////////////////////////////////////////////////////
    // Cluster gen
    ///////////////////////////////////////////////////////////////////////

    enum class ClusteringStrategy { Hilbert };

    template<class Tvec>
    struct ClusteringOptions {
        using Tscal = shambase::VecComponent<Tvec>;

        std::optional<Tscal> cutoff_volume_factor;
    };

    template<class Tvec, u32 cluster_obj_count>
    ClusterList<cluster_obj_count> cluster_objects(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<Tvec> &pos,
        shammath::AABB<Tvec> pos_bounding_box,
        ClusteringStrategy strategy,
        ClusteringOptions<Tvec> options = {});

    ///////////////////////////////////////////////////////////////////////
    // Compute clusters AABB
    ///////////////////////////////////////////////////////////////////////

    template<class Tvec, u32 cluster_obj_count>
    struct ClusterListAABBs {
        u32 cluster_count;
        sham::DeviceBuffer<Tvec> clusters_aabb_min;
        sham::DeviceBuffer<Tvec> clusters_aabb_max;

        struct accessed_ro {
            const Tvec *clusters_aabb_min;
            const Tvec *clusters_aabb_max;

            shammath::AABB<Tvec> get_aabb(u32 cluster_id) {
                return shammath::AABB<Tvec>{
                    clusters_aabb_min[cluster_id], clusters_aabb_max[cluster_id]};
            }
        };

        accessed_ro get_read_access(sham::EventList &depends_list) {
            return {
                clusters_aabb_min.get_read_access(depends_list),
                clusters_aabb_max.get_read_access(depends_list)};
        }

        void complete_event_state(sycl::event e) {
            clusters_aabb_min.complete_event_state(e);
            clusters_aabb_max.complete_event_state(e);
        }

        sham::DeviceBuffer<Tvec> get_cluster_centers(sham::DeviceScheduler_ptr sched) {
            sham::DeviceBuffer<Tvec> ret(cluster_count, sched);
            sham::kernel_call(
                sched->get_queue(),
                sham::MultiRef{*this},
                sham::MultiRef{ret},
                cluster_count,
                [](u32 cluster_id, accessed_ro aabbs, Tvec *center) {
                    center[cluster_id] = aabbs.get_aabb(cluster_id).get_center();
                });
            return ret;
        }
    };

    template<class Tvec, u32 cluster_obj_count>
    ClusterListAABBs<Tvec, cluster_obj_count> get_cluster_aabbs(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<Tvec> &pos,
        ClusterList<cluster_obj_count> &clusters);

    /// compute for each cluser
    /// (aabb + interact radius)_cluster / sum (aabb + interact radius)_objects
    template<class Tvec, u32 cluster_obj_count>
    sham::DeviceBuffer<shambase::VecComponent<Tvec>> compute_compression_ratios(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<Tvec> &pos,
        ClusterList<cluster_obj_count> &clusters,
        sham::DeviceBuffer<shambase::VecComponent<Tvec>> &interact_radius);

    ///////////////////////////////////////////////////////////////////////
    // Cluster field mirror
    ///////////////////////////////////////////////////////////////////////

    /// cluster field mirror
    /// mirror a field to a cluster
    /// aka mirror_field[i*4 + j] = field_ids[cluster_ids[i*4 + j]]
    /// for a cluster of 4 objects
    /// the mirror values fit in a burst buffer and are contiguous
    /// this allows for peak bandwidth usage in theory
    template<class T, u32 cluster_obj_count>
    struct ClusterFieldMirror {
        u32 cluster_count;
        sham::DeviceBuffer<T> mirrored_field;
    };

    template<class T, u32 cluster_obj_count>
    inline ClusterFieldMirror<T, cluster_obj_count> mirror_field_to_cluster(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<T> &field,
        ClusterList<cluster_obj_count> clusters) {

        ClusterFieldMirror<T, cluster_obj_count> ret{
            clusters.cluster_count,
            sham::DeviceBuffer<T>{clusters.cluster_count * cluster_obj_count, sched}};

        sham::kernel_call(
            sched->get_queue(),
            sham::MultiRef{field, clusters},
            sham::MultiRef{ret.mirrored_field},
            clusters.cluster_count * cluster_obj_count,
            [](u32 i, const T *field, auto clusters, T *mirrored_field) {
                u32 cluster_id          = i / cluster_obj_count;
                u32 internal_cluster_id = i % cluster_obj_count;
                mirrored_field[i]
                    = clusters.load_cluster_obj_value(cluster_id, internal_cluster_id, field);
            });
    }

    template<class T, u32 cluster_obj_count>
    inline void store_field_from_cluster(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<T> &field,
        ClusterFieldMirror<T, cluster_obj_count> clusters,
        ClusterFieldMirror<T, cluster_obj_count> mirrored_field) {

        sham::kernel_call(
            sched->get_queue(),
            sham::MultiRef{mirrored_field, clusters},
            sham::MultiRef{field},
            clusters.cluster_count * cluster_obj_count,
            [](u32 i, const T *mirrored_field, auto clusters, T *field) {
                u32 cluster_id          = i / cluster_obj_count;
                u32 internal_cluster_id = i % cluster_obj_count;
                clusters.store_cluster_obj_value(
                    cluster_id, internal_cluster_id, field, mirrored_field[i]);
            });
    }

} // namespace shamtree
