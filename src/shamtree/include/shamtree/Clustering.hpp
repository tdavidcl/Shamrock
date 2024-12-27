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
#include "shammath/AABB.hpp"
#include <array>
namespace shamtree {

    template<u32 cluster_obj_count>
    struct ClusterList {
        static constexpr u32 err_id = shambase::primitive_type_info<u32>::max;

        u32 cluster_count;
        sham::DeviceBuffer<u32> cluster_ids;

        class accessed_ro {
            const u32 *cluster_ids;

            std::array<u32, cluster_obj_count> get_cluster_ids(u32 cluster_id) {
                // clang-format off
                std::array<u32, cluster_obj_count> ret;
                #pragma unroll cluster_obj_count
                for (u32 i = 0; i < cluster_obj_count; i++) {
                    ret[i] = cluster_ids[cluster_id * cluster_obj_count + i];
                }
                // clang-format on
                return ret;
            }

            template<class Tvec, class AABBGetter>
            shammath::AABB<Tvec> get_aabb(u32 cluster_id, AABBGetter &&get_id_aabb) {

                std::array<u32, cluster_obj_count> cluster_ids = get_cluster_ids(cluster_id);

                shammath::AABB<Tvec> ret = shammath::AABB<Tvec>::empty();

                // clang-format off
                #pragma unroll cluster_obj_count
                for (u32 i = 0; i < cluster_obj_count; i++) {
                    if(cluster_ids[i] != err_id) ret.include(get_id_aabb(cluster_ids[i]));
                }
                // clang-format on

                return ret;
            }
        };

        accessed_ro get_read_access(sham::EventList &depends_list) {
            return {cluster_ids.get_read_access(depends_list)};
        }
    };

    enum class ClusteringStrategy { Hilbert };

    template<class Tvec>
    struct ClusteringOptions {
        using Tscal = shambase::VecComponent<Tvec>;

        shambase::opt_ref<shammath::AABB<Tvec>> base_bounding_boxes;
        std::optional<Tscal> cutoff_volume_factor;
    };

    template<class Tvec, u32 cluster_obj_count>
    ClusterList<cluster_obj_count> cluster_objects(
        sham::DeviceScheduler_ptr sched,
        sham::DeviceBuffer<Tvec> &pos,
        shammath::AABB<Tvec> pos_bounding_box,
        ClusteringStrategy strategy,
        ClusteringOptions<Tvec> options = {});

} // namespace shamtree
