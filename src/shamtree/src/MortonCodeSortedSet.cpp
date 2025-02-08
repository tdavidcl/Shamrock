// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MortonCodeSortedSet.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shamtree/MortonCodeSortedSet.hpp"
#include "shamalgs/details/algorithm/algorithm.hpp"
#include <utility>

namespace shamtree {

    template<class Tmorton, class Tvec, u32 dim>
    MortonCodeSortedSet<Tmorton, Tvec, dim> sort_morton_set(
        sham::DeviceScheduler_ptr dev_sched, MortonCodeSet<Tmorton, Tvec, dim> &&morton_codes_set) {

        shammath::AABB<Tvec> bounding_box                = std::move(morton_codes_set.bounding_box);
        u32 cnt_obj                                      = std::move(morton_codes_set.cnt_obj);
        u32 morton_count                                 = std::move(morton_codes_set.morton_count);
        sham::DeviceBuffer<Tmorton> morton_codes_to_sort = std::move(morton_codes_set.morton_codes);

        auto map_morton_id_to_obj_id
            = shamalgs::algorithm::gen_buffer_index_usm(dev_sched, morton_count);

        shamalgs::algorithm::sort_by_key(
            dev_sched, morton_codes_to_sort, map_morton_id_to_obj_id, morton_count);

        return MortonCodeSortedSet<Tmorton, Tvec, dim>(
            std::move(bounding_box),
            std::move(cnt_obj),
            std::move(morton_count),
            std::move(morton_codes_to_sort),
            std::move(map_morton_id_to_obj_id));
    }

} // namespace shamtree

template class shamtree::MortonCodeSortedSet<u32, f64_3, 3>;
template class shamtree::MortonCodeSortedSet<u64, f64_3, 3>;

template shamtree::MortonCodeSortedSet<u32, f64_3, 3> shamtree::sort_morton_set<u32, f64_3, 3>(
    sham::DeviceScheduler_ptr dev_sched, shamtree::MortonCodeSet<u32, f64_3, 3> &&morton_codes_set);
template shamtree::MortonCodeSortedSet<u64, f64_3, 3> shamtree::sort_morton_set<u64, f64_3, 3>(
    sham::DeviceScheduler_ptr dev_sched, shamtree::MortonCodeSet<u64, f64_3, 3> &&morton_codes_set);
