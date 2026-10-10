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
 * @file ReattributeDataUtility.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/string.hpp"
#include "shamalgs/collective/distributedDataComm.hpp"
#include "shamalgs/numeric.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shamalgs/primitives/scan_exclusive_sum_in_place.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/comm/details/CommunicationBufferImpl.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <unordered_map>
#include <map>
#include <vector>

namespace shamrock {

    /**
     * @brief Utility class used to move the objects between patches.
     *
     * The class is used to recompute the ownership of the objects in the patches
     * based on their position in space.
     *
     */
    class ReattributeDataUtility {
        PatchScheduler &sched; ///< Scheduler to bind onto

        public:
        /**
         * @brief Constructor
         *
         * @param sched The PatchScheduler to work on.
         */
        ReattributeDataUtility(PatchScheduler &sched) : sched(sched) {}

        /**
         * @brief Computes the new patch owner IDs for the objects in the patches based on their
         * position in space.
         *
         * @param sptree The SerialPatchTree used to compute the patch owners.
         * @param ipos The index of the position field in the PatchData.
         *
         * @return A DistributedData containing the new patch IDs for each patch.
         *
         * @throws std::runtime_error If a new ID could not be computed for an object (out of
         * bound).
         */
        template<class T>
        shambase::DistributedData<sycl::buffer<u64>> compute_new_pid(
            SerialPatchTree<T> &sptree, u32 ipos) {

            StackEntry stack_loc{};

            shambase::DistributedData<sycl::buffer<u64>> newid_buf_map;

            sched.patch_data.for_each_patchdata([&](u64 id, shamrock::patch::PatchDataLayer &pdat) {
                if (!pdat.is_empty()) {

                    PatchDataField<T> &pos_field = pdat.get_field<T>(ipos);

                    if (pos_field.get_nvar() != 1) {
                        shambase::throw_unimplemented();
                    }

                    newid_buf_map.add_obj(
                        id,
                        sptree.compute_patch_owner(
                            shamsys::instance::get_compute_scheduler_ptr(),
                            pos_field.get_buf(),
                            pos_field.get_obj_cnt()));

                    bool err_id_in_newid = false;
                    {
                        sycl::host_accessor nid{newid_buf_map.get(id), sycl::read_only};
                        for (u32 i = 0; i < pdat.get_obj_cnt(); i++) {
                            bool err        = nid[i] == u64_max;
                            err_id_in_newid = err_id_in_newid || (err);
                        }
                    }

                    if (err_id_in_newid) {
                        throw shambase::make_except_with_loc<std::runtime_error>(
                            "a new id could not be computed");
                    }
                }
            });

            return newid_buf_map;
        }

        /**
         * @brief Flag the objects of a patch whose new patch id differs from the current one.
         *
         * @param current_pid The id of the patch holding the objects.
         * @param new_pid The new patch id of each object.
         * @param cnt The number of objects in the patch.
         *
         * @return A device buffer of size cnt, 1 if the object leaves the patch, 0 otherwise.
         */
        static sham::DeviceBuffer<u32> flag_moved_objects(
            u64 current_pid, sycl::buffer<u64> &new_pid, u32 cnt) {
            StackEntry stack_loc{};

            auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
            sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

            sham::DeviceBuffer<u32> flag_moved(cnt, dev_sched);

            sham::EventList depends_list;
            u32 *flag = flag_moved.get_write_access(depends_list);

            auto e = q.submit(depends_list, [&, current_pid](sycl::handler &cgh) {
                sycl::accessor nid{new_pid, cgh, sycl::read_only};
                shambase::parallel_for(cgh, cnt, "flag moved objects", [=](u32 i) {
                    flag[i] = (nid[i] != current_pid) ? 1 : 0;
                });
            });

            flag_moved.complete_event_state(e);

            return flag_moved;
        }

        /**
         * @brief Count the objects of a patch whose new patch id differs from the current one.
         *
         * @param flag_moved The flags returned by flag_moved_objects.
         * @param cnt The number of objects in the patch.
         *
         * @return The number of objects leaving the patch.
         */
        static u32 count_moved_objects(const sham::DeviceBuffer<u32> &flag_moved, u32 cnt) {
            return shamalgs::primitives::sum(
                shamsys::instance::get_compute_scheduler_ptr(), flag_moved, 0, cnt);
        }

        /**
         * @brief Extracts elements that do not belong to a patch from the patch data based on the
         * new patch IDs.
         *
         * For each patch, the objects whose new patch id differs from the current one are counted
         * on the device. If none is leaving, the patch is left untouched. Otherwise the index
         * lists of the objects to keep and to move are built on the device from a single exclusive
         * scan of the flags, the moved objects are appended to one PatchDataLayer per destination
         * patch and the remaining ones are kept in place.
         *
         * @param new_pid A distributed data object containing the new patch IDs.
         *
         * @return A shared distributed data object containing the extracted patch data.
         */
        inline shambase::DistributedDataShared<shamrock::patch::PatchDataLayer> extract_elements(
            shambase::DistributedData<sycl::buffer<u64>> new_pid) {
            shambase::DistributedDataShared<patch::PatchDataLayer> part_exchange;

            StackEntry stack_loc{};

            using namespace shamrock::patch;

            auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
            sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

            std::unordered_map<u64, u64> histogram_extract;

            sched.patch_data.for_each_patchdata([&](u64 current_pid, PatchDataLayer &pdat) {
                histogram_extract[current_pid] = 0;

                if (pdat.is_empty()) {
                    return;
                }

                const u32 cnt              = pdat.get_obj_cnt();
                sycl::buffer<u64> &nid_buf = new_pid.get(current_pid);

                // flag = 1 if the object leaves the patch
                sham::DeviceBuffer<u32> flag_moved = flag_moved_objects(current_pid, nid_buf, cnt);

                const u32 moved_cnt = count_moved_objects(flag_moved, cnt);

                histogram_extract[current_pid] = moved_cnt;

                // fast path: no object leaves the patch, nothing to extract or to reallocate
                if (moved_cnt == 0) {
                    return;
                }

                const u32 keep_cnt = cnt - moved_cnt;

                // exclusive scan of the flags: rank of each object among the moved ones, and
                // i - rank its rank among the kept ones
                shamalgs::primitives::scan_exclusive_sum_in_place(flag_moved, cnt);

                sham::DeviceBuffer<u32> moved_ids(moved_cnt, dev_sched);
                sham::DeviceBuffer<u64> moved_new_pid(moved_cnt, dev_sched);
                sham::DeviceBuffer<u32> keep_ids(keep_cnt, dev_sched);
                {
                    sham::EventList depends_list;
                    const u32 *rank = flag_moved.get_read_access(depends_list);
                    u32 *moved      = moved_ids.get_write_access(depends_list);
                    u64 *moved_pid  = moved_new_pid.get_write_access(depends_list);
                    u32 *keep       = keep_ids.get_write_access(depends_list);

                    auto e = q.submit(depends_list, [&, current_pid](sycl::handler &cgh) {
                        sycl::accessor nid{nid_buf, cgh, sycl::read_only};
                        shambase::parallel_for(cgh, cnt, "split moved / kept objects", [=](u32 i) {
                            u64 pid = nid[i];
                            u32 r   = rank[i];
                            if (pid != current_pid) {
                                moved[r]     = i;
                                moved_pid[r] = pid;
                            } else {
                                keep[i - r] = i;
                            }
                        });
                    });

                    flag_moved.complete_event_state(e);
                    moved_ids.complete_event_state(e);
                    moved_new_pid.complete_event_state(e);
                    keep_ids.complete_event_state(e);
                }

                // the destinations are neighbouring patches so there are only a few of them
                std::map<u64, u32> dest_counts;
                for (u64 dest : moved_new_pid.copy_to_stdvec()) {
                    dest_counts[dest]++;
                }

                for (auto &[dest, dest_cnt] : dest_counts) {

                    auto it = part_exchange.add_obj(
                        current_pid, dest, PatchDataLayer(sched.get_layout_ptr_old()));
                    PatchDataLayer &pdat_send = it->second;

                    if (dest_counts.size() == 1) {
                        pdat.append_subset_to(moved_ids, dest_cnt, pdat_send);
                        continue;
                    }

                    // select the moved objects going to dest
                    sham::DeviceBuffer<u32> flag_dest(moved_cnt, dev_sched);
                    sham::kernel_call(
                        q,
                        sham::MultiRef{moved_new_pid},
                        sham::MultiRef{flag_dest},
                        moved_cnt,
                        [dest](u32 i, const u64 *__restrict pid, u32 *__restrict flag) {
                            flag[i] = (pid[i] == dest) ? 1 : 0;
                        });

                    sham::DeviceBuffer<u32> sel
                        = shamalgs::numeric::stream_compact(dev_sched, flag_dest, moved_cnt);

                    sham::DeviceBuffer<u32> dest_ids(dest_cnt, dev_sched);
                    sham::kernel_call(
                        q,
                        sham::MultiRef{sel, moved_ids},
                        sham::MultiRef{dest_ids},
                        dest_cnt,
                        [](u32 i,
                           const u32 *__restrict sel_idx,
                           const u32 *__restrict ids,
                           u32 *__restrict out_ids) {
                            out_ids[i] = ids[sel_idx[i]];
                        });

                    pdat.append_subset_to(dest_ids, dest_cnt, pdat_send);
                }

                pdat.keep_ids(keep_ids, keep_cnt);
            });

            for (auto &[k, v] : histogram_extract) {
                shamlog_debug_ln("ReattributeDataUtility", "patch", k, "extract=", v);
            }

            return part_exchange;
        }

        /**
         * @brief Reattribute objects based on a given position field.
         *
         * This function computes new patch IDs for each object in the PatchData,
         * extracts elements to be exchanged between patches, and then updates the patch data
         * with the received elements.
         *
         * @param sptree the SerialPatchTree
         * @param position_field the name of the main field used to determine the new patch IDs
         */
        template<class T>
        inline void reatribute_patch_objects(
            SerialPatchTree<T> &sptree, std::string position_field) {
            StackEntry stack_loc{};

            using namespace shambase;
            using namespace shamrock::patch;

            u32 ipos = sched.pdl_old().get_field_idx<T>(position_field);

            DistributedData<sycl::buffer<u64>> new_pid = compute_new_pid(sptree, ipos);

            DistributedDataShared<patch::PatchDataLayer> part_exchange = extract_elements(new_pid);

            part_exchange.for_each([](u64 sender, u64 receiver, PatchDataLayer &pdat) {
                shamlog_debug_ln("ReattributeDataUtility", sender, receiver, pdat.get_obj_cnt());
            });

            DistributedDataShared<patch::PatchDataLayer> recv_dat;

            shamalgs::collective::DDSCommCache cache;

            shamalgs::collective::serialize_sparse_comm<PatchDataLayer>(
                shamsys::instance::get_compute_scheduler_ptr(),
                std::move(part_exchange),
                recv_dat,
                [&](u64 id) {
                    return sched.get_patch_rank_owner(id);
                },
                [](PatchDataLayer &pdat) {
                    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
                    ser.allocate(pdat.serialize_buf_byte_size());
                    pdat.serialize_buf(ser);
                    return ser.finalize();
                },
                [&](sham::DeviceBuffer<u8> &&buf) {
                    // exchange the buffer held by the distrib data and give it to the serializer
                    shamalgs::SerializeHelper ser(
                        shamsys::instance::get_compute_scheduler_ptr(),
                        std::forward<sham::DeviceBuffer<u8>>(buf));
                    return PatchDataLayer::deserialize_buf(ser, sched.get_layout_ptr_old());
                },
                cache);

            recv_dat.for_each([&](u64 sender, u64 receiver, PatchDataLayer &pdat) {
                shamlog_debug_ln("Part Exchanges", format("send = {} recv = {}", sender, receiver));
                sched.patch_data.get_pdat(receiver).insert_elements(pdat);
            });
        }
    };

} // namespace shamrock
