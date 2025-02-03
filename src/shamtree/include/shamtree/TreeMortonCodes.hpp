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
 * @file TreeMortonCodes.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/reduction.hpp"
#include "shamalgs/serialize.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shammath/CoordRange.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/MortonCodeSet.hpp"
#include "shamtree/MortonCodeSortedSet.hpp"
#include "shamtree/RadixTreeMortonBuilder.hpp"
#include <stdexcept>
#include <utility>

namespace shamrock::tree {

    template<class u_morton>
    class TreeMortonCodes {
        public:
        u32 obj_cnt;

        sham::DeviceBuffer<u_morton> buf_morton;
        sham::DeviceBuffer<u32> buf_particle_index_map;

        static TreeMortonCodes create(
            u32 obj_cnt,
            sham::DeviceBuffer<u_morton> &&buf_morton,
            sham::DeviceBuffer<u32> &&buf_particle_index_map) {
            return {
                obj_cnt,
                std::forward<sham::DeviceBuffer<u_morton>>(buf_morton),
                std::forward<sham::DeviceBuffer<u32>>(buf_particle_index_map)};
        }

        template<class T>
        static TreeMortonCodes create(
            sycl::queue &queue,
            shammath::CoordRange<T> coord_range,
            u32 obj_cnt,
            sycl::buffer<T> &pos_buf) {

            StackEntry stack_loc{};

            sham::DeviceBuffer<T> pos_buf2(obj_cnt, shamsys::instance::get_compute_scheduler_ptr());
            pos_buf2.copy_from_sycl_buffer(pos_buf);

            shamtree::MortonCodeSet<u_morton, T> mset{
                shamsys::instance::get_compute_scheduler_ptr(),
                {coord_range.lower, coord_range.upper},
                pos_buf2,
                obj_cnt,
                sham::roundup_pow2_clz(obj_cnt)};

            shamtree::MortonCodeSortedSet<u_morton, T> mset_sorted{
                shamsys::instance::get_compute_scheduler_ptr(), std::move(mset)};

            return create(
                obj_cnt,
                std::move(mset_sorted.sorted_morton_codes),
                std::move(mset_sorted.map_morton_id_to_obj_id));
        }

        template<class T>
        static TreeMortonCodes create(
            sham::DeviceScheduler_ptr dev_sched,
            shammath::CoordRange<T> coord_range,
            u32 obj_cnt,
            sham::DeviceBuffer<T> &pos_buf) {

            StackEntry stack_loc{};

            shamtree::MortonCodeSet<u_morton, T> mset{
                shamsys::instance::get_compute_scheduler_ptr(),
                {coord_range.lower, coord_range.upper},
                pos_buf,
                obj_cnt,
                sham::roundup_pow2_clz(obj_cnt)};

            shamtree::MortonCodeSortedSet<u_morton, T> mset_sorted{
                shamsys::instance::get_compute_scheduler_ptr(), std::move(mset)};

            return create(
                obj_cnt,
                std::move(mset_sorted.sorted_morton_codes),
                std::move(mset_sorted.map_morton_id_to_obj_id));
        }

        template<class T>
        TreeMortonCodes(
            sycl::queue &queue,
            shammath::CoordRange<T> coord_range,
            u32 obj_cnt,
            sycl::buffer<T> &pos_buf)
            : TreeMortonCodes{create(queue, coord_range, obj_cnt, pos_buf)} {}

        template<class T>
        TreeMortonCodes(
            sham::DeviceScheduler_ptr dev_sched,
            shammath::CoordRange<T> coord_range,
            u32 obj_cnt,
            sham::DeviceBuffer<T> &pos_buf)
            : TreeMortonCodes{create(dev_sched, coord_range, obj_cnt, pos_buf)} {}

        [[nodiscard]] inline u64 memsize() const {
            u64 sum = 0;

            auto add_ptr = [&](auto &a) {
                if (a) {
                    sum += a.get_mem_usage();
                }
            };

            sum += sizeof(obj_cnt);

            add_ptr(buf_morton);
            add_ptr(buf_particle_index_map);

            return sum;
        }

        inline TreeMortonCodes() = default;

        inline TreeMortonCodes(const TreeMortonCodes &other)
            : obj_cnt(other.obj_cnt), buf_morton(other.buf_morton->copy()),
              buf_particle_index_map(other.buf_particle_index_map->copy()) {}

        inline TreeMortonCodes &operator=(TreeMortonCodes &&other) noexcept {
            obj_cnt                = std::move(other.obj_cnt);
            buf_morton             = std::move(other.buf_morton);
            buf_particle_index_map = std::move(other.buf_particle_index_map);

            return *this;
        } // move assignment

        inline friend bool operator==(const TreeMortonCodes &t1, const TreeMortonCodes &t2) {
            bool cmp = true;

            cmp = cmp && (t1.obj_cnt == t2.obj_cnt);

            using namespace shamalgs::reduction;

            cmp = cmp
                  && equals(
                      shamsys::instance::get_compute_queue(),
                      t1.buf_morton,
                      t2.buf_morton,
                      t1.obj_cnt);
            cmp = cmp
                  && equals(
                      shamsys::instance::get_compute_queue(),
                      t1.buf_particle_index_map,
                      t2.buf_particle_index_map,
                      t1.obj_cnt);

            return cmp;
        }

        /**
         * @brief serialize a TreeMortonCodes object
         *
         * @param serializer
         */
        inline void serialize(shamalgs::SerializeHelper &serializer) {
            StackEntry stack_loc{};

            serializer.write(obj_cnt);
            if (!buf_morton) {
                throw shambase::make_except_with_loc<std::runtime_error>("missing buffer");
            }
            // serializer.write(buf_morton->size());
            serializer.write_buf(*buf_morton, obj_cnt);
            if (!buf_particle_index_map) {
                throw shambase::make_except_with_loc<std::runtime_error>("missing buffer");
            }
            serializer.write_buf(*buf_particle_index_map, obj_cnt);
        }

        /**
         * @brief deserialize a TreeMortonCodes object
         * Note : here since the initial buffer is a pow of 2
         * with trailling terms for the bitonic sort, when
         * deserializing we are not loading the last values
         * the buffer size is obj_cnt here
         *
         * @param serializer
         * @return TreeMortonCodes
         */
        inline static TreeMortonCodes deserialize(shamalgs::SerializeHelper &serializer) {
            StackEntry stack_loc{};
            TreeMortonCodes ret;
            serializer.load(ret.obj_cnt);

            // u32 morton_len;
            // serializer.load(morton_len);
            ret.buf_morton = std::make_unique<sham::DeviceBuffer<u_morton>>(
                ret.obj_cnt, shamsys::instance::get_compute_scheduler_ptr());
            ret.buf_particle_index_map = std::make_unique<sham::DeviceBuffer<u32>>(
                ret.obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

            serializer.load_buf(*ret.buf_morton, ret.obj_cnt);
            serializer.load_buf(*ret.buf_particle_index_map, ret.obj_cnt);

            return ret;
        }

        /**
         * @brief give the size of the serialized object
         *
         * @return u64
         */
        inline shamalgs::SerializeSize serialize_byte_size() {
            using H = shamalgs::SerializeHelper;
            return H::serialize_byte_size<u32>() + H::serialize_byte_size<u32>(obj_cnt)
                   + H::serialize_byte_size<u_morton>(obj_cnt);
        }
    };

} // namespace shamrock::tree
