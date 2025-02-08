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
 * @file MortonCodeSortedSet.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shamtree/MortonCodeSet.hpp"

namespace shamtree {

    /**
     * @brief Class representing a set of Morton codes with associated bounding box and position
     * data
     *
     * @tparam Tmorton The type used for Morton codes
     * @tparam Tvec The vector type representing positions
     * @tparam dim The dimensionality, inferred from Tvec if not provided
     */
    template<class Tmorton, class Tvec, u32 dim = shambase::VectorProperties<Tvec>::dimension>
    class MortonCodeSortedSet {
        public:
        /// The axis-aligned bounding box for the set of positions
        shammath::AABB<Tvec> bounding_box;

        /// The count of objects represented in the set
        u32 cnt_obj;

        /// The count of Morton codes in the set (rounded to a power of 2)
        u32 morton_count;

        /// Device buffer holding the sorted Morton codes
        sham::DeviceBuffer<Tmorton> sorted_morton_codes;

        /// Device buffer holding the map from sorted Morton code to object id
        sham::DeviceBuffer<u32> map_morton_id_to_obj_id;

        /// Move constructor from each members
        MortonCodeSortedSet(
            shammath::AABB<Tvec> &&bounding_box,
            u32 &&cnt_obj,
            u32 &&morton_count,
            sham::DeviceBuffer<Tmorton> &&sorted_morton_codes,
            sham::DeviceBuffer<u32> &&map_morton_id_to_obj_id)
            : bounding_box(std::move(bounding_box)), cnt_obj(std::move(cnt_obj)),
              morton_count(std::move(morton_count)),
              sorted_morton_codes(std::move(sorted_morton_codes)),
              map_morton_id_to_obj_id(std::move(map_morton_id_to_obj_id)) {}
    };

    /**
     * @brief Constructs a MortonCodeSet
     *
     * @param dev_sched The device scheduler for managing SYCL operations
     * @param bounding_box The bounding box encapsulating the input positions
     * @param pos_buf The buffer containing the input positions
     * @param cnt_obj The number of positions in the buffer
     */
    template<class Tmorton, class Tvec, u32 dim>
    MortonCodeSortedSet<Tmorton, Tvec, dim> sort_morton_set(
        sham::DeviceScheduler_ptr dev_sched, MortonCodeSet<Tmorton, Tvec, dim> &&morton_codes_set);

} // namespace shamtree
