// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"

#include "shamalgs/memory.hpp"
#include "shamalgs/random.hpp"
#include "shammath/AABB.hpp"
#include "shammath/crystalLattice.hpp"
#include "shamrock/amr/AMRGrid.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/sfc/MortonKernels.hpp"
#include "shamrock/sfc/morton.hpp"
#include "shamrock/tree/RadixTree.hpp"
#include "shamrock/tree/TreeStructureWalker.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shamrock/tree/TreeTests.hpp"

template<class Tvec>
sycl::buffer<Tvec> generate_positions(sham::VecComponent<Tvec> dr, Tvec boxmin, Tvec boxmax) {

    using Lattice     = shammath::LatticeHCP<Tvec>;
    using LatticeIter = typename shammath::LatticeHCP<Tvec>::IteratorDiscontinuous;

    auto [idxs_min, idxs_max] = Lattice::get_box_index_bounds(dr, boxmin, boxmax);

    LatticeIter gen = LatticeIter(dr, idxs_min, idxs_max);

    std::vector<Tvec> positions{};

    while (!gen.is_done()) {
        std::vector<Tvec> to_ins = gen.next_n(1000000);
        positions.insert(positions.end(), to_ins.begin(), to_ins.end());
    }

    return shamalgs::memory::vec_to_buf(positions);
}

struct Treebuildperf {
    f64 times_morton;
    f64 times_reduc;
    f64 times_karras;
    f64 times_compute_int_range;
    f64 times_compute_coord_range;
    f64 times_morton_build;
    f64 times_trailling_fill;
    f64 times_index_gen;
    f64 times_morton_sort;
    f64 times_full_tree;
};

template<class Tvec, class Tmorton>
Treebuildperf
benchmark_tree(sycl::buffer<Tvec> &positions, Tvec boxmin, Tvec boxmax, u32 reduc_lev) {

    Treebuildperf ret;

    auto get_repetition_count = [](f64 cnt) {
        if (cnt < 1e5)
            return 100;
        return 30;
    };

    shammath::CoordRange<Tvec> coord_range = {boxmin, boxmax};

    u32 rep_count = get_repetition_count(positions.size());

    logger::debug_ln("TestTreePerf", positions.size());
    for (u32 rep_count = 0; rep_count < get_repetition_count(positions.size()); rep_count++) {

        shambase::Timer timer;
        u32 cnt_obj = positions.size();

        auto time_func = [](auto f) {
            shamsys::instance::get_compute_queue().wait();
            shambase::Timer timer;
            timer.start();

            f();
            shamsys::instance::get_compute_queue().wait();

            timer.end();
            return timer.nanosec / 1.e9;
        };

        {
            shamrock::tree::TreeMortonCodes<Tmorton> tree_morton_codes;
            shamrock::tree::TreeReducedMortonCodes<Tmorton> tree_reduced_morton_codes;
            shamrock::tree::TreeStructure<Tmorton> tree_struct;

            ret.times_morton += (time_func([&]() {
                tree_morton_codes.build(
                    shamsys::instance::get_compute_queue(), coord_range, cnt_obj, positions);
            }));

            bool one_cell_mode;
            ret.times_reduc += (time_func([&]() {
                tree_reduced_morton_codes.build(
                    shamsys::instance::get_compute_queue(),
                    cnt_obj,
                    reduc_lev,
                    tree_morton_codes,
                    one_cell_mode);
            }));

            ret.times_karras += (time_func([&]() {
                if (!one_cell_mode) {
                    tree_struct.build(
                        shamsys::instance::get_compute_queue(),
                        tree_reduced_morton_codes.tree_leaf_count - 1,
                        *tree_reduced_morton_codes.buf_tree_morton);
                } else {
                    tree_struct.build_one_cell_mode();
                }
            }));
        }

        {
            RadixTree<Tmorton, Tvec> rtree = RadixTree<Tmorton, Tvec>(
                shamsys::instance::get_compute_queue(),
                {coord_range.lower, coord_range.upper},
                positions,
                cnt_obj,
                reduc_lev);

            ret.times_compute_int_range += (time_func([&]() {
                rtree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
            }));

            shamsys::instance::get_compute_queue().wait();
            ret.times_compute_coord_range += (time_func([&]() {
                rtree.convert_bounding_box(shamsys::instance::get_compute_queue());
            }));
        }

        {

            using namespace shamrock::sfc;

            u32 morton_len = shambase::roundup_pow2_clz(cnt_obj);

            auto out_buf_morton = std::make_unique<sycl::buffer<Tmorton>>(morton_len);

            ret.times_morton_build += (time_func([&]() {
                MortonKernels<Tmorton, Tvec, 3>::sycl_xyz_to_morton(
                    shamsys::instance::get_compute_queue(),
                    cnt_obj,
                    positions,
                    coord_range.lower,
                    coord_range.upper,
                    out_buf_morton);
            }));

            ret.times_trailling_fill += (time_func([&]() {
                MortonKernels<Tmorton, Tvec, 3>::sycl_fill_trailling_buffer(
                    shamsys::instance::get_compute_queue(), cnt_obj, morton_len, out_buf_morton);
            }));

            std::unique_ptr<sycl::buffer<u32>> out_buf_particle_index_map;

            ret.times_index_gen += (time_func([&]() {
                out_buf_particle_index_map
                    = std::make_unique<sycl::buffer<u32>>(shamalgs::algorithm::gen_buffer_index(
                        shamsys::instance::get_compute_queue(), morton_len));
            }));

            ret.times_morton_sort += (time_func([&]() {
                sycl_sort_morton_key_pair(
                    shamsys::instance::get_compute_queue(),
                    morton_len,
                    out_buf_particle_index_map,
                    out_buf_morton);
            }));
        }

        {
            shamsys::instance::get_compute_queue().wait();
            shambase::Timer timer2;
            timer2.start();

            RadixTree<Tmorton, Tvec> rtree = RadixTree<Tmorton, Tvec>(
                shamsys::instance::get_compute_queue(),
                {coord_range.lower, coord_range.upper},
                positions,
                cnt_obj,
                reduc_lev);

            rtree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
            rtree.convert_bounding_box(shamsys::instance::get_compute_queue());
            shamsys::instance::get_compute_queue().wait();
            timer2.end();
            ret.times_full_tree += (timer2.nanosec / 1.e9);
        }
    }

    ret.times_morton /= get_repetition_count(positions.size());
    ret.times_reduc /= get_repetition_count(positions.size());
    ret.times_karras /= get_repetition_count(positions.size());
    ret.times_compute_int_range /= get_repetition_count(positions.size());
    ret.times_compute_coord_range /= get_repetition_count(positions.size());
    ret.times_morton_build /= get_repetition_count(positions.size());
    ret.times_trailling_fill /= get_repetition_count(positions.size());
    ret.times_index_gen /= get_repetition_count(positions.size());
    ret.times_morton_sort /= get_repetition_count(positions.size());
    ret.times_full_tree /= get_repetition_count(positions.size());

    return ret;
}

template<class Tvec, class Tmorton>
void do_benchmark_build(u32 reduc_level){
    using Tscal = sham::VecComponent<Tvec>;

    Tvec box_min {-1,-1,-1};
    Tvec box_max {1,1,1};


    std::vector<f64> times_morton              ;
    std::vector<f64> times_reduc               ;
    std::vector<f64> times_karras              ;
    std::vector<f64> times_compute_int_range   ;
    std::vector<f64> times_compute_coord_range ;
    std::vector<f64> times_morton_build        ;
    std::vector<f64> times_trailling_fill      ;
    std::vector<f64> times_index_gen           ;
    std::vector<f64> times_morton_sort         ;
    std::vector<f64> times_full_tree           ;

    std::vector<f64> Npart;

    for(Tscal dr = 0.25; dr > 0.02; dr /= 1.1){

        sycl::buffer<Tvec> pos = generate_positions(dr, box_min, box_max);

        shamalgs::memory::move_buffer_on_queue(
            shamsys::instance::get_compute_queue(), 
            pos);

        Treebuildperf ret = benchmark_tree<Tvec, Tmorton>(pos, box_min, box_max, reduc_level);

        Npart.push_back(pos.size());
        times_morton             .push_back(ret.times_morton             ) ;
        times_reduc              .push_back(ret.times_reduc              ) ;
        times_karras             .push_back(ret.times_karras             ) ;
        times_compute_int_range  .push_back(ret.times_compute_int_range  ) ;
        times_compute_coord_range.push_back(ret.times_compute_coord_range) ;
        times_morton_build       .push_back(ret.times_morton_build       ) ;
        times_trailling_fill     .push_back(ret.times_trailling_fill     ) ;
        times_index_gen          .push_back(ret.times_index_gen          ) ;
        times_morton_sort        .push_back(ret.times_morton_sort        ) ;
        times_full_tree          .push_back(ret.times_full_tree          ) ;

    }

    std::string name = "";
    if(std::is_same_v<Tmorton, u64>){
        name += "u64_";
    }
    if(std::is_same_v<Tmorton, u32>){
        name += "u32_";
    }

    name += "tree_perf_reduc_"+std::to_string(reduc_level);


    PyScriptHandle hdnl{};

    hdnl.data()["Npart"] = Npart;

    hdnl.data()["times_morton"]   = times_morton;
    hdnl.data()["times_reduc"]   = times_reduc;
    hdnl.data()["times_karras"]   = times_karras;
    hdnl.data()["times_compute_int_range"]   = times_compute_int_range;
    hdnl.data()["times_compute_coord_range"]   = times_compute_coord_range;
    hdnl.data()["times_morton_build"]   = times_morton_build;
    hdnl.data()["times_trailling_fill"]   = times_trailling_fill;
    hdnl.data()["times_index_gen"]   = times_index_gen;
    hdnl.data()["times_morton_sort"]   = times_morton_sort;
    hdnl.data()["times_full_tree"]   = times_full_tree;
    hdnl.data()["name"]       = name;

    hdnl.exec(R"(
        import numpy as np
        import matplotlib.pyplot as plt

        plt.style.use('custom_style.mplstyle')

        Npart = np.array(Npart)
        times_morton = np.array(times_morton)
        times_reduc = np.array(times_reduc)
        times_karras = np.array(times_karras)
        times_compute_int_range = np.array(times_compute_int_range)
        times_compute_coord_range = np.array(times_compute_coord_range)
        times_morton_build = np.array(times_morton_build)
        times_trailling_fill = np.array(times_trailling_fill)
        times_index_gen = np.array(times_index_gen)
        times_morton_sort = np.array(times_morton_sort)
        times_full_tree = np.array(times_full_tree)

        np.save("tests/figures/"+name+"Npart.npy",Npart)
        np.save("tests/figures/"+name+"times_morton.npy",times_morton)
        np.save("tests/figures/"+name+"times_reduc.npy",times_reduc)
        np.save("tests/figures/"+name+"times_karras.npy",times_karras)
        np.save("tests/figures/"+name+"times_compute_int_range.npy",times_compute_int_range)
        np.save("tests/figures/"+name+"times_compute_coord_range.npy",times_compute_coord_range)
        np.save("tests/figures/"+name+"times_morton_build.npy",times_morton_build)
        np.save("tests/figures/"+name+"times_trailling_fill.npy",times_trailling_fill)
        np.save("tests/figures/"+name+"times_index_gen.npy",times_index_gen)
        np.save("tests/figures/"+name+"times_morton_sort.npy",times_morton_sort)
        np.save("tests/figures/"+name+"times_full_tree.npy",times_full_tree)

        plt.plot(Npart, Npart / times_morton, lw=2, label="morton build")
        plt.plot(Npart, Npart / times_reduc, lw=2, label="reduction")
        plt.plot(Npart, Npart / times_karras, lw=2, label="T. Karras")
        plt.plot(Npart, Npart / times_compute_int_range, lw=2, label="int range")
        plt.plot(Npart, Npart / times_compute_coord_range, lw=2, label="coord range")
        plt.plot(Npart, Npart / times_morton_build, lw=2, label="morton build")
        plt.plot(Npart, Npart / times_trailling_fill, lw=2, label="trailling fill")
        plt.plot(Npart, Npart / times_index_gen, lw=2, label="index gen")
        plt.plot(Npart, Npart / times_morton_sort, lw=2, label="morton sort")
        plt.plot(Npart, Npart / times_full_tree,"--",c="black", lw=3, label="full tree")

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"$N$")
        plt.ylabel(r"$t/N$")
        plt.legend()

        ax = plt.axis()
        plt.axis((ax[0],ax[1],ax[3],ax[2]))

        plt.tight_layout()  

        plt.savefig("tests/figures/"+name+".pdf", dpi = 300)

        plt.cla()

    )");


}

TestStart(Benchmark, "tree_build_benchmark", tree_build_benchmark, 1){

    do_benchmark_build<f64_3,u32>(0);
    do_benchmark_build<f64_3,u64>(0);
    do_benchmark_build<f64_3,u32>(1);
    do_benchmark_build<f64_3,u64>(1);
    do_benchmark_build<f64_3,u32>(2);
    do_benchmark_build<f64_3,u64>(2);
    do_benchmark_build<f64_3,u32>(3);
    do_benchmark_build<f64_3,u64>(3);

}