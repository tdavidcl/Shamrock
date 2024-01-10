// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GhostZones.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shammodels/amr/basegodunov/modules/GhostZones.hpp"
#include "shamalgs/numeric.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/sycl_utils.hpp"
#include "shammath/AABB.hpp"
#include "shammath/CoordRange.hpp"
#include "shammodels/amr/basegodunov/GhostZoneData.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::basegodunov::modules::GhostZones<Tvec, TgridVec>;

namespace shammodels::basegodunov::modules {
    /**
     * @brief find interfaces corresponding to shared surface between domains
     *
     * @tparam Tvec
     * @tparam TgridVec
     */
    template<class Tvec, class TgridVec>
    auto find_interfaces(PatchScheduler &sched, SerialPatchTree<TgridVec> &sptree) {

        using namespace shamrock::patch;
        using namespace shammath;

        using GZData              = shammodels::basegodunov::GhostZonesData<Tvec, TgridVec>;
        static constexpr u32 dim  = shambase::VectorProperties<Tvec>::dimension;
        using InterfaceBuildInfos = typename GZData::InterfaceBuildInfos;
        using GeneratorMap        = typename GZData::GeneratorMap;

        StackEntry stack_loc{};

        i32 repetition_x = 1;
        i32 repetition_y = 1;
        i32 repetition_z = 1;

        GeneratorMap results;

        shamrock::patch::SimulationBoxInfo &sim_box = sched.get_sim_box();

        PatchCoordTransform<TgridVec> patch_coord_transf = sim_box.get_patch_transform<TgridVec>();
        TgridVec bsize = sim_box.get_bounding_box_size<TgridVec>();

        for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
            for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
                for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {

                    // sender translation
                    TgridVec periodic_offset =
                        TgridVec{xoff * bsize.x(), yoff * bsize.y(), zoff * bsize.z()};

                    sched.for_each_local_patch([&](const Patch psender) {
                        CoordRange<TgridVec> sender_bsize =
                            patch_coord_transf.to_obj_coord(psender);
                        CoordRange<TgridVec> sender_bsize_off =
                            sender_bsize.add_offset(periodic_offset);

                        shammath::AABB<TgridVec> sender_bsize_off_aabb{
                            sender_bsize_off.lower, sender_bsize_off.upper};

                        using PtNode = typename SerialPatchTree<TgridVec>::PtNode;

                        logger::debug_sycl_ln(
                            "AMR:interf",
                            "find_interfaces -",
                            psender.id_patch,
                            sender_bsize_off_aabb.lower,
                            sender_bsize_off_aabb.upper);

                        sptree.host_for_each_leafs(
                            [&](u64 tree_id, PtNode n) {
                                shammath::AABB<TgridVec> tree_cell{n.box_min, n.box_max};

                                bool result =
                                    tree_cell.get_intersect(sender_bsize_off_aabb).is_not_empty();

                                return result;
                            },
                            [&](u64 id_found, PtNode n) {
                                if ((id_found == psender.id_patch) && (xoff == 0) && (yoff == 0) &&
                                    (zoff == 0)) {
                                    return;
                                }

                                InterfaceBuildInfos ret{
                                    periodic_offset,
                                    {xoff, yoff, zoff},
                                    shammath::AABB<TgridVec>{
                                        n.box_min - periodic_offset, n.box_max - periodic_offset}};

                                results.add_obj(psender.id_patch, id_found, std::move(ret));
                            });
                    });
                }
            }
        }

        return results;
    }
} // namespace shammodels::godunov::modules

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::build_ghost_cache() {

    StackEntry stack_loc{};

    using GZData = GhostZonesData<Tvec, TgridVec>;

    storage.ghost_zone_infos.set(GZData{});
    GZData &gen_ghost = storage.ghost_zone_infos.get();

    // get ids of cells that will be on the surface of another patch.
    // for cells corresponding to fixed boundary they will be generated after the exhange
    // and appended to the interface list a poosteriori

    gen_ghost.ghost_gen_infos =
        find_interfaces<Tvec, TgridVec>(scheduler(), storage.serial_patch_tree.get());

    using InterfaceBuildInfos = typename GZData::InterfaceBuildInfos;
    using InterfaceIdTable    = typename GZData::InterfaceIdTable;

    // if(logger::log_debug);
    gen_ghost.ghost_gen_infos.for_each([&](u64 sender, u64 receiver, InterfaceBuildInfos &build) {
        std::string log;

        log = shambase::format(
            "{} -> {} : off = {}, {} -> {}",
            sender,
            receiver,
            build.offset,
            build.volume_target.lower,
            build.volume_target.upper);

        logger::debug_ln("AMRgodunov", log);
    });

    sycl::queue &q = shamsys::instance::get_compute_queue();

    gen_ghost.ghost_gen_infos.for_each([&](u64 sender, u64 receiver, InterfaceBuildInfos &build) {
        shamrock::patch::PatchData &src = scheduler().patch_data.get_pdat(sender);

        sycl::buffer<u32> is_in_interf{src.get_obj_cnt()};

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{src.get_field_buf_ref<TgridVec>(0), cgh, sycl::read_only};
            sycl::accessor cell_max{src.get_field_buf_ref<TgridVec>(1), cgh, sycl::read_only};
            sycl::accessor flag{is_in_interf, cgh, sycl::write_only, sycl::no_init};

            shammath::AABB<TgridVec> check_volume = build.volume_target;

            shambase::parralel_for(cgh, src.get_obj_cnt(), "check if in interf", [=](u32 id_a) {
                flag[id_a] = shammath::AABB<TgridVec>(cell_min[id_a], cell_max[id_a])
                                 .get_intersect(check_volume)
                                 .is_not_empty();
            });
        });

        auto resut = shamalgs::numeric::stream_compact(q, is_in_interf, src.get_obj_cnt());
        f64 ratio  = f64(std::get<1>(resut)) / f64(src.get_obj_cnt());

        std::string s = shambase::format(
            "{} -> {} : off = {}, test volume = {} -> {}",
            sender,
            receiver,
            build.offset,
            build.volume_target.lower,
            build.volume_target.upper);
        s += shambase::format("\n    found N = {}, ratio = {} %", std::get<1>(resut), ratio);

        logger::debug_ln("AMR interf", s);

        std::unique_ptr<sycl::buffer<u32>> ids =
            std::make_unique<sycl::buffer<u32>>(shambase::extract_value(std::get<0>(resut)));

        gen_ghost.ghost_id_build_map.add_obj(
            sender, receiver, InterfaceIdTable{build, std::move(ids), ratio});
    });
}

template<class Tvec, class TgridVec>
shambase::DistributedDataShared<shamrock::patch::PatchData>
Module<Tvec, TgridVec>::communicate_pdat(
    shamrock::patch::PatchDataLayout &pdl,
    shambase::DistributedDataShared<shamrock::patch::PatchData> &&interf) {
    StackEntry stack_loc{};

    shambase::DistributedDataShared<shamrock::patch::PatchData> recv_dat;

    shamalgs::collective::serialize_sparse_comm<shamrock::patch::PatchData>(
        std::forward<shambase::DistributedDataShared<shamrock::patch::PatchData>>(interf),
        recv_dat,
        [&](u64 id) {
            return scheduler().get_patch_rank_owner(id);
        },
        [](shamrock::patch::PatchData &pdat) {
            shamalgs::SerializeHelper ser;
            ser.allocate(pdat.serialize_buf_byte_size());
            pdat.serialize_buf(ser);
            return ser.finalize();
        },
        [&](std::unique_ptr<sycl::buffer<u8>> &&buf) {
            // exchange the buffer held by the distrib data and give it to the serializer
            shamalgs::SerializeHelper ser(std::forward<std::unique_ptr<sycl::buffer<u8>>>(buf));
            return shamrock::patch::PatchData::deserialize_buf(ser, pdl);
        });

    return recv_dat;
}

template<class Tvec, class TgridVec>
template<class T>
shambase::DistributedDataShared<PatchDataField<T>> Module<Tvec, TgridVec>::communicate_pdat_field(
    shambase::DistributedDataShared<PatchDataField<T>> &&interf) {
    StackEntry stack_loc{};

    shambase::DistributedDataShared<PatchDataField<T>> recv_dat;

    shamalgs::collective::serialize_sparse_comm<PatchDataField<T>>(
        std::forward<shambase::DistributedDataShared<PatchDataField<T>>>(interf),
        recv_dat,
        [&](u64 id) {
            return scheduler().get_patch_rank_owner(id);
        },
        [](PatchDataField<T> &pdat) {
            shamalgs::SerializeHelper ser;
            ser.allocate(pdat.serialize_full_byte_size());
            pdat.serialize_buf(ser);
            return ser.finalize();
        },
        [&](std::unique_ptr<sycl::buffer<u8>> &&buf) {
            // exchange the buffer held by the distrib data and give it to the serializer
            shamalgs::SerializeHelper ser(std::forward<std::unique_ptr<sycl::buffer<u8>>>(buf));
            return PatchDataField<T>::deserialize_full(ser);
        });

    return recv_dat;
}

template<class Tvec, class TgridVec>
template<class T, class Tmerged>
shambase::DistributedData<Tmerged> Module<Tvec, TgridVec>::merge_native(
    shambase::DistributedDataShared<T> &&interfs,
    std::function<Tmerged(const shamrock::patch::Patch, shamrock::patch::PatchData &pdat)> init,
    std::function<void(Tmerged &, T &)> appender) {

    StackEntry stack_loc{};

    shambase::DistributedData<Tmerged> merge_f;

    scheduler().for_each_patchdata_nonempty(
        [&](const shamrock::patch::Patch p, shamrock::patch::PatchData &pdat) {
            Tmerged tmp_merge = init(p, pdat);

            interfs.for_each([&](u64 sender, u64 receiver, T &interface) {
                if (receiver == p.id_patch) {
                    appender(tmp_merge, interface);
                }
            });

            merge_f.add_obj(p.id_patch, std::move(tmp_merge));
        });

    return merge_f;
}

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::exchange_ghost() {

    StackEntry stack_loc{};

    shambase::Timer timer_interf;
    timer_interf.start();

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    using GZData              = GhostZonesData<Tvec, TgridVec>;
    using InterfaceBuildInfos = typename GZData::InterfaceBuildInfos;
    using InterfaceIdTable    = typename GZData::InterfaceIdTable;

    using AMRBlock = typename Config::AMRBlock;

    // setup ghost layout
    storage.ghost_layout.set(shamrock::patch::PatchDataLayout{});
    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();

    ghost_layout.add_field<TgridVec>("cell_min", 1);
    ghost_layout.add_field<TgridVec>("cell_max", 1);
    ghost_layout.add_field<Tscal>("rho", AMRBlock::block_size);
    ghost_layout.add_field<Tscal>("rhoetot", AMRBlock::block_size);
    ghost_layout.add_field<Tvec>("rhovel", AMRBlock::block_size);

    u32 icell_min_interf = ghost_layout.get_field_idx<TgridVec>("cell_min");
    u32 icell_max_interf = ghost_layout.get_field_idx<TgridVec>("cell_max");
    u32 irho_interf      = ghost_layout.get_field_idx<Tscal>("rho");
    u32 irhoetot_interf     = ghost_layout.get_field_idx<Tscal>("rhoetot");
    u32 irhovel_interf      = ghost_layout.get_field_idx<Tvec>("rhovel");

    // load layout info
    PatchDataLayout &pdl = scheduler().pdl;

    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot     = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel      = pdl.get_field_idx<Tvec>("rhovel");

    // generate send buffers
    GZData &gen_ghost = storage.ghost_zone_infos.get();
    auto pdat_interf  = gen_ghost.template build_interface_native<PatchData>(
        [&](u64 sender, u64, InterfaceBuildInfos binfo, sycl::buffer<u32> &buf_idx, u32 cnt) {
            PatchData &sender_patch = scheduler().patch_data.get_pdat(sender);

            PatchData pdat(ghost_layout);

            pdat.reserve(cnt);

            sender_patch.get_field<TgridVec>(icell_min).append_subset_to(
                buf_idx, cnt, pdat.get_field<TgridVec>(icell_min_interf));

            sender_patch.get_field<TgridVec>(icell_max).append_subset_to(
                buf_idx, cnt, pdat.get_field<TgridVec>(icell_max_interf));

            sender_patch.get_field<Tscal>(irho).append_subset_to(
                buf_idx, cnt, pdat.get_field<Tscal>(irho_interf));

            sender_patch.get_field<Tscal>(irhoetot).append_subset_to(
                buf_idx, cnt, pdat.get_field<Tscal>(irhoetot_interf));

            sender_patch.get_field<Tvec>(irhovel).append_subset_to(
                buf_idx, cnt, pdat.get_field<Tvec>(irhovel_interf));

            pdat.check_field_obj_cnt_match();

            pdat.get_field<TgridVec>(icell_min_interf).apply_offset(binfo.offset);
            pdat.get_field<TgridVec>(icell_max_interf).apply_offset(binfo.offset);

            return pdat;
        });

    // communicate buffers
    shambase::DistributedDataShared<PatchData> interf_pdat =
        communicate_pdat(ghost_layout, std::move(pdat_interf));

    std::map<u64, u64> sz_interf_map;
    interf_pdat.for_each([&](u64 s, u64 r, PatchData &pdat_interf) {
        sz_interf_map[r] += pdat_interf.get_obj_cnt();
    });

    storage.merged_patchdata_ghost.set(merge_native<PatchData, MergedPatchData>(
        std::move(interf_pdat),
        [&](const shamrock::patch::Patch p, shamrock::patch::PatchData &pdat) {
            logger::debug_ln("Merged patch init", p.id_patch);

            PatchData pdat_new(ghost_layout);

            u32 or_elem = pdat.get_obj_cnt();
            pdat_new.reserve(or_elem + sz_interf_map[p.id_patch]);
            u32 total_elements = or_elem;

            pdat_new.get_field<TgridVec>(icell_min_interf)
                .insert(pdat.get_field<TgridVec>(icell_min));
            pdat_new.get_field<TgridVec>(icell_max_interf)
                .insert(pdat.get_field<TgridVec>(icell_max));
            pdat_new.get_field<Tscal>(irho_interf).insert(pdat.get_field<Tscal>(irho));
            pdat_new.get_field<Tscal>(irhoetot_interf).insert(pdat.get_field<Tscal>(irhoetot));
            pdat_new.get_field<Tvec>(irhovel_interf).insert(pdat.get_field<Tvec>(irhovel));

            pdat_new.check_field_obj_cnt_match();

            return MergedPatchData{or_elem, total_elements, std::move(pdat_new), ghost_layout};
        },
        [](MergedPatchData &mpdat, PatchData &pdat_interf) {
            mpdat.total_elements += pdat_interf.get_obj_cnt();
            mpdat.pdat.insert_elements(pdat_interf);
        }));

    storage.merged_patchdata_ghost.get().for_each([](u64 id, shamrock::MergedPatchData &mpdat) {
        logger::debug_ln(
            "Merged patch", id, ",", mpdat.original_elements, "->", mpdat.total_elements);
    });

    timer_interf.end();
    storage.timings_details.interface += timer_interf.elasped_sec();
}

template<class Tvec, class TgridVec>
template<class T>
shamrock::ComputeField<T>
Module<Tvec, TgridVec>::exchange_compute_field(shamrock::ComputeField<T> &in) {

    StackEntry stack_loc{};

    shambase::Timer timer_interf;
    timer_interf.start();

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    using GZData              = GhostZonesData<Tvec, TgridVec>;
    using InterfaceBuildInfos = typename GZData::InterfaceBuildInfos;
    using InterfaceIdTable    = typename GZData::InterfaceIdTable;

    using AMRBlock = typename Config::AMRBlock;

    // generate send buffers
    GZData &gen_ghost = storage.ghost_zone_infos.get();
    auto pdat_interf  = gen_ghost.template build_interface_native<PatchDataField<T>>(
        [&](u64 sender, u64, InterfaceBuildInfos binfo, sycl::buffer<u32> &buf_idx, u32 cnt) {
            PatchDataField<T> &sender_patch = in.get_field(sender);

            PatchDataField<T> pdat(sender_patch.get_name(), sender_patch.get_nvar(), cnt);

            return pdat;
        });

    // communicate buffers
    shambase::DistributedDataShared<PatchDataField<T>> interf_pdat =
        communicate_pdat_field<T>(std::move(pdat_interf));

    std::map<u64, u64> sz_interf_map;
    interf_pdat.for_each([&](u64 s, u64 r, PatchDataField<T> &pdat_interf) {
        sz_interf_map[r] += pdat_interf.get_obj_cnt();
    });

    ComputeField<T> out;
    scheduler().for_each_patchdata_nonempty(
        [&](const shamrock::patch::Patch p, shamrock::patch::PatchData &pdat) {
            PatchDataField<T> &receiver_patch = in.get_field(p.id_patch);

            PatchDataField<T> new_pdat(receiver_patch);

            interf_pdat.for_each([&](u64 sender, u64 receiver, PatchDataField<T> &interface) {
                if (receiver == p.id_patch) {
                    new_pdat.insert(interface);
                }
            });

            out.field_data.add_obj(p.id_patch, std::move(new_pdat));
        });

    timer_interf.end();
    storage.timings_details.interface += timer_interf.elasped_sec();
    return out;
}

template class shammodels::basegodunov::modules::GhostZones<f64_3, i64_3>;
template shamrock::ComputeField<f64_8>
shammodels::basegodunov::modules::GhostZones<f64_3, i64_3>::exchange_compute_field<f64_8>(
    shamrock::ComputeField<f64_8> &in);
template shambase::DistributedDataShared<PatchDataField<f64_8>>
shammodels::basegodunov::modules::GhostZones<f64_3, i64_3>::communicate_pdat_field<f64_8>(
    shambase::DistributedDataShared<PatchDataField<f64_8>> &&interf);