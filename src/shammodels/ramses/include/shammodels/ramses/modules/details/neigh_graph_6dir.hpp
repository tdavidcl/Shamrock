// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file neigh_graph_6dir.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Build the offsets of the 6 directional neighbour graphs with a single scan
 * (NeighGraphStrategy::NeighGraphOpt)
 */

#include "shamalgs/details/numeric/numeric.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/kernel_call.hpp"
#include <array>
#include <memory>

namespace shammodels::basegodunov::modules::details {

    /**
     * @brief Index of the link count of object i for direction dir in the buffer of the 6
     * directions
     *
     * The counts of a direction are followed by a separator slot, and the buffer ends with one
     * more slot (see neigh_6dir_count_size)
     */
    inline u32 neigh_6dir_count_idx(u32 dir, u32 i, u32 obj_cnt) { return dir * (obj_cnt + 1) + i; }

    /// Size of the buffer of the link counts of the 6 directions
    inline u32 neigh_6dir_count_size(u32 obj_cnt) { return 6 * (obj_cnt + 1) + 1; }

    /// Per direction offsets of the links of each object, and number of links
    struct NeighGraph6DirOffsets {
        std::array<std::unique_ptr<sham::DeviceBuffer<u32>>, 6> node_link_offset;
        std::array<u32, 6> link_count;
    };

    /**
     * @brief Exclusive scan of the link counts of the 6 directions with one scan
     *
     * `counts` (neigh_6dir_count_size(obj_cnt) entries) must hold the link count of object i for
     * direction dir at neigh_6dir_count_idx(dir, i, obj_cnt); the separator slots are set to 0
     * here. For each direction the returned node_link_offset (obj_cnt + 1 entries) is the
     * exclusive scan of its counts, the last entry being its link count, exactly as if it had
     * been scanned on its own: the scan of the concatenated counts minus its value at the start
     * of the direction.
     */
    inline NeighGraph6DirOffsets scan_link_counts_6dir(
        const sham::DeviceScheduler_ptr &dev_sched, sham::DeviceBuffer<u32> &counts, u32 obj_cnt) {

        auto &q = dev_sched->get_queue();

        u32 stride = obj_cnt + 1;
        u32 len    = neigh_6dir_count_size(obj_cnt);

        // separators : after the counts of each direction, and the last slot
        sham::kernel_call(
            q, sham::MultiRef{}, sham::MultiRef{counts}, 7, [stride](u32 k, u32 *__restrict c) {
                c[k * stride + ((k < 6) ? stride - 1 : 0)] = 0;
            });

        sham::DeviceBuffer<u32> scanned = shamalgs::numeric::scan_exclusive(dev_sched, counts, len);

        // start of each direction in the scan (the 7th is the total)
        sham::DeviceBuffer<u32> starts(7, dev_sched);
        sham::kernel_call(
            q,
            sham::MultiRef{scanned},
            sham::MultiRef{starts},
            7,
            [stride](u32 k, const u32 *__restrict s, u32 *__restrict st) {
                st[k] = s[k * stride];
            });
        std::vector<u32> base = starts.copy_to_stdvec();

        NeighGraph6DirOffsets ret;
        for (u32 dir = 0; dir < 6; dir++) {
            ret.node_link_offset[dir]
                = std::make_unique<sham::DeviceBuffer<u32>>(stride, dev_sched);
            ret.link_count[dir] = base[dir + 1] - base[dir];
        }

        sham::kernel_call(
            q,
            sham::MultiRef{scanned},
            sham::MultiRef{
                *ret.node_link_offset[0],
                *ret.node_link_offset[1],
                *ret.node_link_offset[2],
                *ret.node_link_offset[3],
                *ret.node_link_offset[4],
                *ret.node_link_offset[5]},
            stride,
            [stride,
             b0 = base[0],
             b1 = base[1],
             b2 = base[2],
             b3 = base[3],
             b4 = base[4],
             b5 = base[5]](
                u32 i,
                const u32 *__restrict s,
                u32 *__restrict o0,
                u32 *__restrict o1,
                u32 *__restrict o2,
                u32 *__restrict o3,
                u32 *__restrict o4,
                u32 *__restrict o5) {
                o0[i] = s[0 * stride + i] - b0;
                o1[i] = s[1 * stride + i] - b1;
                o2[i] = s[2 * stride + i] - b2;
                o3[i] = s[3 * stride + i] - b3;
                o4[i] = s[4 * stride + i] - b4;
                o5[i] = s[5 * stride + i] - b5;
            });

        return ret;
    }

} // namespace shammodels::basegodunov::modules::details
