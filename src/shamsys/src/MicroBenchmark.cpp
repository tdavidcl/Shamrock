// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file MicroBenchmark.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "sham/format/human_readable.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/Device.hpp"
#include "shambackends/benchmarks/cache_chase.hpp"
#include "shambackends/benchmarks/fma_chains.hpp"
#include "shambackends/benchmarks/int_chains.hpp"
#include "shambackends/benchmarks/saxpy.hpp"
#include "shambackends/benchmarks/warp_divergence.hpp"
#include "shambackends/comm/CommunicationBuffer.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/wrapper.hpp"
#include "shamsys/MicroBenchmark.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <stdexcept>
#include <vector>

namespace {

    std::unordered_map<std::string, double> microbench_results = {};

}

namespace shamsys::microbench {
    /// MPI point-to-point bandwidth benchmark
    void p2p_bandwidth(u32 wr_sender, u32 wr_receiv);

    /// MPI point-to-point latency benchmark
    void p2p_latency(u32 wr1, u32 wr2);

    /// SAXPY benchmark, to get the maximum bandwidth
    template<typename T>
    void saxpy();

    /// FMA chains benchmark to get the maximum floating point performance
    template<typename T>
    void fma_chains_rotation();

    /// Integer chains benchmark to get the maximum integer performance
    template<typename T, sham::benchmarks::IntChainOp op>
    void int_chains_rotation();

    /// Pointer chasing benchmark, to expose the cache tiers of the device
    void cache_latency(u64 working_set_bytes, const std::string &label);

    /// Branch divergence benchmark, to get the cost of a divergent branch
    void branch_divergence(u32 n_paths);

    /// Vector allgather benchmark
    void vector_allgather(u32 el_per_rank);

} // namespace shamsys::microbench

void shamsys::run_micro_benchmark() {
    StackEntry stack_loc{};

    if (shamcomm::world_rank() == 0) {
        logger::raw_ln("Running micro benchmarks:");
    }

    u32 wr1 = 0;
    u32 wr2 = shamcomm::world_size() - 1;

    microbench::p2p_bandwidth(wr1, wr2);
    if (shamcomm::world_size() > 1) {
        microbench::p2p_latency(wr1, wr2);
    }
    microbench::saxpy<f32>();
    microbench::saxpy<f64>();
    microbench::saxpy<f32_2>();
    microbench::saxpy<f64_2>();
    microbench::saxpy<f32_3>();
    microbench::saxpy<f64_3>();
    microbench::saxpy<f32_4>();
    microbench::saxpy<f64_4>();
    microbench::fma_chains_rotation<f32>();
    microbench::fma_chains_rotation<f64>();
    microbench::fma_chains_rotation<f32_2>();
    microbench::fma_chains_rotation<f64_2>();
    microbench::fma_chains_rotation<f32_3>();
    microbench::fma_chains_rotation<f64_3>();
    microbench::fma_chains_rotation<f32_4>();
    microbench::fma_chains_rotation<f64_4>();
    microbench::int_chains_rotation<u32, sham::benchmarks::IntChainOp::Mul>();
    microbench::int_chains_rotation<u64, sham::benchmarks::IntChainOp::Mul>();
    microbench::int_chains_rotation<u32, sham::benchmarks::IntChainOp::Add>();
    microbench::int_chains_rotation<u64, sham::benchmarks::IntChainOp::Add>();
    microbench::cache_latency(16 * 1024, "16KiB");
    microbench::cache_latency(128 * 1024, "128KiB");
    microbench::cache_latency(1024 * 1024, "1MiB");
    microbench::cache_latency(4 * 1024 * 1024, "4MiB");
    microbench::cache_latency(16 * 1024 * 1024, "16MiB");
    microbench::cache_latency(64 * 1024 * 1024, "64MiB");
    microbench::cache_latency(256 * 1024 * 1024, "256MiB");
    microbench::branch_divergence(2);
    microbench::branch_divergence(4);
    microbench::branch_divergence(8);
    microbench::branch_divergence(16);
    microbench::branch_divergence(32);
    microbench::branch_divergence(64);
    microbench::vector_allgather(1);
    microbench::vector_allgather(8);
    microbench::vector_allgather(64);
    microbench::vector_allgather(128);
    microbench::vector_allgather(150);
    microbench::vector_allgather(1024);
}

void shamsys::microbench::p2p_bandwidth(u32 wr_sender, u32 wr_receiv) {
    StackEntry stack_loc{};

    u32 wr = shamcomm::world_rank();

    u64 length = 1024UL * 1014UL * 8UL; // 8MB messages
    shamcomm::CommunicationBuffer buf_recv{length, instance::get_compute_scheduler_ptr()};
    shamcomm::CommunicationBuffer buf_send{length, instance::get_compute_scheduler_ptr()};

    std::vector<MPI_Request> rqs;

    f64 t        = 0;
    u64 loops    = 0;
    bool is_used = false;
    do {
        loops++;

        mpi::barrier(MPI_COMM_WORLD);
        f64 t_start = MPI_Wtime();

        if (wr == wr_sender) {
            rqs.push_back(MPI_Request{});
            u32 rq_index = rqs.size() - 1;
            auto &rq     = rqs[rq_index];
            shamcomm::mpi::Isend(
                buf_send.get_ptr(), length, MPI_BYTE, wr_receiv, 0, MPI_COMM_WORLD, &rq);
            is_used = true;
        }

        if (wr == wr_receiv) {
            MPI_Status s;
            shamcomm::mpi::Recv(
                buf_recv.get_ptr(), length, MPI_BYTE, wr_sender, 0, MPI_COMM_WORLD, &s);
            is_used = true;
        }

        if (!is_used) {
            t = 1;
        }
        std::vector<MPI_Status> st_lst(rqs.size());
        if (rqs.size() > 0) {
            shamcomm::mpi::Waitall(rqs.size(), rqs.data(), st_lst.data());
        }
        f64 t_end = MPI_Wtime();
        t += t_end - t_start;

    } while (shamalgs::collective::allreduce_min(t) < 1);

    f64 bw = f64(length * loops) / t;

    microbench_results["p2p_bandwidth"] = bw;

    if (shamcomm::world_rank() == 0) {
        auto hr_bw = sham::to_human_readable<false>(bw);
        logger::raw_ln(
            sham::format(
                " - p2p bandwidth    : {:.2f} {}B.s^-1 (ranks : {} -> {}) (loops : {})",
                hr_bw.value,
                hr_bw.prefix,
                wr_sender,
                wr_receiv,
                loops));
    }
}

void shamsys::microbench::p2p_latency(u32 wr1, u32 wr2) {
    StackEntry stack_loc{};

    if (wr1 == wr2) {
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "can not launch this test with same ranks");
    }

    u32 wr = shamcomm::world_rank();

    u64 length = 8ULL; // 8B messages
    shamcomm::CommunicationBuffer buf_recv{length, instance::get_compute_scheduler_ptr()};
    shamcomm::CommunicationBuffer buf_send{length, instance::get_compute_scheduler_ptr()};

    shambase::Timer bench_timer;
    bench_timer.start();

    f64 t        = 0;
    u64 loops    = 0;
    bool is_used = false;
    do {
        loops++;

        mpi::barrier(MPI_COMM_WORLD);
        f64 t_start = MPI_Wtime();

        if (wr == wr1) {
            MPI_Status s;
            shamcomm::mpi::Send(buf_send.get_ptr(), length, MPI_BYTE, wr2, 0, MPI_COMM_WORLD);
            shamcomm::mpi::Recv(buf_recv.get_ptr(), length, MPI_BYTE, wr2, 1, MPI_COMM_WORLD, &s);
            is_used = true;
        }

        if (wr == wr2) {
            MPI_Status s;
            shamcomm::mpi::Recv(buf_recv.get_ptr(), length, MPI_BYTE, wr1, 0, MPI_COMM_WORLD, &s);
            shamcomm::mpi::Send(buf_send.get_ptr(), length, MPI_BYTE, wr1, 1, MPI_COMM_WORLD);
            is_used = true;
        }

        if (!is_used) {
            t = 1;
        }
        f64 t_end = MPI_Wtime();
        t += t_end - t_start;

        bench_timer.stop();

    } while (shamalgs::collective::allreduce_min(bench_timer.elapsed_sec()) < 1);

    f64 latency                       = t / f64(loops);
    microbench_results["p2p_latency"] = latency;

    if (shamcomm::world_rank() == 0) {
        logger::raw_ln(
            sham::format(
                " - p2p latency     : {:.4e} s (ranks : {} <-> {}) (loops : {})",
                latency,
                wr1,
                wr2,
                loops));
    }
}

template<typename T>
void shamsys::microbench::saxpy() {
    int Tsize = sizeof(T);

    std::string type_name;
    T init_x, init_y, a;
    if constexpr (std::is_same_v<T, f32>) {
        type_name = "f32";
        init_x    = 1.0f;
        init_y    = 2.0f;
        a         = 2.0f;
    } else if constexpr (std::is_same_v<T, f64>) {
        type_name = "f64";
        init_x    = 1.0;
        init_y    = 2.0;
        a         = 2.0;
    } else if constexpr (std::is_same_v<T, f32_2>) {
        type_name = "f32_2";
        init_x    = {1.0f, 1.0f};
        init_y    = {2.0f, 2.0f};
        a         = {2.0f, 2.0f};
    } else if constexpr (std::is_same_v<T, f64_2>) {
        type_name = "f64_2";
        init_x    = {1.0, 1.0};
        init_y    = {2.0, 2.0};
        a         = {2.0, 2.0};
    } else if constexpr (std::is_same_v<T, f32_3>) {
        type_name = "f32_3";
        init_x    = {1.0f, 1.0f, 1.0f};
        init_y    = {2.0f, 2.0f, 2.0f};
        a         = {2.0f, 2.0f, 2.0f};
    } else if constexpr (std::is_same_v<T, f64_3>) {
        type_name = "f64_3";
        init_x    = {1.0, 1.0, 1.0};
        init_y    = {2.0, 2.0, 2.0};
        a         = {2.0, 2.0, 2.0};
    } else if constexpr (std::is_same_v<T, f32_4>) {
        type_name = "f32_4";
        init_x    = {1.0f, 1.0f, 1.0f, 1.0f};
        init_y    = {2.0f, 2.0f, 2.0f, 2.0f};
        a         = {2.0f, 2.0f, 2.0f, 2.0f};
    } else if constexpr (std::is_same_v<T, f64_4>) {
        type_name = "f64_4";
        init_x    = {1.0, 1.0, 1.0, 1.0};
        init_y    = {2.0, 2.0, 2.0, 2.0};
        a         = {2.0, 2.0, 2.0, 2.0};
    } else {
        throw shambase::make_except_with_loc<std::invalid_argument>("unsupported type");
    }

    auto bench_step = [&](int N) {
        return sham::benchmarks::saxpy_bench<T>(
            instance::get_compute_scheduler_ptr(), N, init_x, init_y, a, Tsize, N < (1 << 17));
    };

    auto benchmark = [&]() {
        size_t N = (1 << 15);

        auto &dev_ctx = shambase::get_check_ref(instance::get_compute_scheduler().ctx);
        auto &dev_ptr = dev_ctx.device;
        auto &dev     = shambase::get_check_ref(dev_ptr);

        size_t max_alloc
            = std::min<size_t>(dev.prop.max_mem_alloc_size_dev, dev.prop.global_mem_size);
        double max_size = double(max_alloc) / (Tsize * 4); // there is 2 allocations so /4
        if (max_size >= (1 << 30)) {
            max_size = (1 << 30);
        }

        auto result = bench_step(shambase::narrow_or_throw<i32>(N));

        for (; N <= (1 << 30) && static_cast<double>(N) <= max_size; N *= 2) {
            result = bench_step(shambase::narrow_or_throw<i32>(N));

            // std::cout << N << " " << result_new.seconds << " " << result_new.bandwidth
            //           << std::endl;

            if (result.seconds > 1e-3) {
                break;
            }
        }

        return result;
    };

    auto result = benchmark();

    f64 bw = result.bandwidth * 1e9;

    f64 min_bw = shamalgs::collective::allreduce_min(bw);
    f64 max_bw = shamalgs::collective::allreduce_max(bw);
    f64 sum_bw = shamalgs::collective::allreduce_sum(bw);
    f64 avg_bw = sum_bw / (f64) shamcomm::world_size();

    microbench_results["saxpy_" + type_name] = sum_bw;

    if (shamcomm::world_rank() == 0) {
        auto hr_bw = sham::to_human_readable<false>(sum_bw);
        logger::raw_ln(
            sham::format(
                " - saxpy ({})   : {:.2f} {}B.s^-1 (min = {:.1e}, max = {:.1e}, avg = {:.1e}) "
                "({:.1e} ms, {})",
                type_name,
                hr_bw.value,
                hr_bw.prefix,
                min_bw,
                max_bw,
                avg_bw,
                result.seconds * 1e3,
                shambase::readable_sizeof(result.byte_used)));
    }
}

template<typename T>
void shamsys::microbench::fma_chains_rotation() {
    int N = (1 << 22);

    auto result
        = sham::benchmarks::fma_chains_bench<T>(instance::get_compute_scheduler_ptr(), N, 0.2);

    std::string type_name;
    f64 flops_multiplier = 1;
    if constexpr (std::is_same_v<T, f32>) {
        type_name        = "f32";
        flops_multiplier = 1;
    } else if constexpr (std::is_same_v<T, f64>) {
        type_name        = "f64";
        flops_multiplier = 1;
    } else if constexpr (std::is_same_v<T, f32_2>) {
        type_name        = "f32_2";
        flops_multiplier = 2;
    } else if constexpr (std::is_same_v<T, f64_2>) {
        type_name        = "f64_2";
        flops_multiplier = 2;
    } else if constexpr (std::is_same_v<T, f32_3>) {
        type_name        = "f32_3";
        flops_multiplier = 3;
    } else if constexpr (std::is_same_v<T, f64_3>) {
        type_name        = "f64_3";
        flops_multiplier = 3;
    } else if constexpr (std::is_same_v<T, f32_4>) {
        type_name        = "f32_4";
        flops_multiplier = 4;
    } else if constexpr (std::is_same_v<T, f64_4>) {
        type_name        = "f64_4";
        flops_multiplier = 4;
    } else {
        throw shambase::make_except_with_loc<std::invalid_argument>("unsupported type");
    }

    f64 min_flop = shamalgs::collective::allreduce_min(result.flops);
    f64 max_flop = shamalgs::collective::allreduce_max(result.flops);
    f64 sum_flop = shamalgs::collective::allreduce_sum(result.flops);
    f64 avg_flop = sum_flop / (f64) shamcomm::world_size();

    microbench_results["fma_chains_" + type_name] = sum_flop * flops_multiplier;

    if (shamcomm::world_rank() == 0) {
        auto hr_flop = sham::to_human_readable<false>(sum_flop * flops_multiplier);
        logger::raw_ln(
            sham::format(
                " - fma_chains ({}) : {:.2f} {}flops (min = {:.1e}, max = {:.1e}, avg = {:.1e}) "
                "({:.1e} ms, rotations = {})",
                type_name,
                hr_flop.value,
                hr_flop.prefix,
                min_flop * flops_multiplier,
                max_flop * flops_multiplier,
                avg_flop * flops_multiplier,
                result.seconds * 1e3,
                result.nrotations));
    }
}

template<typename T, sham::benchmarks::IntChainOp op>
void shamsys::microbench::int_chains_rotation() {
    int N = (1 << 22);

    auto result
        = sham::benchmarks::int_chains_bench<T, op>(instance::get_compute_scheduler_ptr(), N, 0.2);

    std::string type_name;
    if constexpr (std::is_same_v<T, u32>) {
        type_name = "u32";
    } else if constexpr (std::is_same_v<T, u64>) {
        type_name = "u64";
    } else {
        throw shambase::make_except_with_loc<std::invalid_argument>("unsupported type");
    }

    std::string op_name = (op == sham::benchmarks::IntChainOp::Mul) ? "mul" : "add";

    f64 min_iops = shamalgs::collective::allreduce_min(result.iops);
    f64 max_iops = shamalgs::collective::allreduce_max(result.iops);
    f64 sum_iops = shamalgs::collective::allreduce_sum(result.iops);
    f64 avg_iops = sum_iops / (f64) shamcomm::world_size();

    microbench_results["int_" + op_name + "_chains_" + type_name] = sum_iops;

    if (shamcomm::world_rank() == 0) {
        auto hr_iops = sham::to_human_readable<false>(sum_iops);
        logger::raw_ln(
            sham::format(
                " - int_{}_chains ({}) : {:.2f} {}iops (min = {:.1e}, max = {:.1e}, avg = {:.1e}) "
                "({:.1e} ms, rotations = {})",
                op_name,
                type_name,
                hr_iops.value,
                hr_iops.prefix,
                min_iops,
                max_iops,
                avg_iops,
                result.seconds * 1e3,
                result.nrotations));
    }
}

void shamsys::microbench::cache_latency(u64 working_set_bytes, const std::string &label) {
    StackEntry stack_loc{};

    auto &dev_ctx = shambase::get_check_ref(instance::get_compute_scheduler().ctx);
    auto &dev     = shambase::get_check_ref(dev_ctx.device);

    u64 max_working_set
        = std::min<u64>(dev.prop.max_mem_alloc_size_dev, dev.prop.global_mem_size / 4);

    if (working_set_bytes > max_working_set) {
        if (shamcomm::world_rank() == 0) {
            logger::raw_ln(
                sham::format(
                    " - cache_chase ({:>6}) : skipped, larger than the device can hold", label));
        }
        return;
    }

    u32 n_elem   = u32(working_set_bytes / sizeof(u32));
    u32 n_chains = 1 << 16;

    auto result = sham::benchmarks::cache_chase_bench(
        instance::get_compute_scheduler_ptr(), n_elem, n_chains, 0.1);

    f64 sum_lat  = shamalgs::collective::allreduce_sum(result.latency);
    f64 avg_lat  = sum_lat / (f64) shamcomm::world_size();
    f64 sum_rate = shamalgs::collective::allreduce_sum(result.hop_rate);

    microbench_results["cache_chase_lat_" + label]  = avg_lat;
    microbench_results["cache_chase_rate_" + label] = sum_rate;

    if (shamcomm::world_rank() == 0) {
        auto hr_rate = sham::to_human_readable<false>(sum_rate);
        logger::raw_ln(
            sham::format(
                " - cache_chase ({:>6}) : {:.3e} s/hop, {:.2f} {}hops.s^-1 (chains = {}, hops = "
                "{})",
                label,
                avg_lat,
                hr_rate.value,
                hr_rate.prefix,
                result.n_chains,
                result.nsteps));
    }
}

void shamsys::microbench::branch_divergence(u32 n_paths) {
    StackEntry stack_loc{};

    int N = (1 << 22);

    // kept low on purpose: the divergent run costs up to n_paths times this
    auto result = sham::benchmarks::divergence_bench<f32>(
        instance::get_compute_scheduler_ptr(), N, n_paths, 0.05);

    f64 min_penalty = shamalgs::collective::allreduce_min(result.penalty);
    f64 max_penalty = shamalgs::collective::allreduce_max(result.penalty);
    f64 sum_penalty = shamalgs::collective::allreduce_sum(result.penalty);
    f64 avg_penalty = sum_penalty / (f64) shamcomm::world_size();

    microbench_results["divergence_penalty_" + std::to_string(n_paths)] = avg_penalty;

    if (shamcomm::world_rank() == 0) {
        logger::raw_ln(
            sham::format(
                " - divergence (n_paths={:3}) : {:.3f} x slowdown (min = {:.2f}, max = {:.2f}) "
                "({:.1e} ms uniform, {:.1e} ms divergent)",
                n_paths,
                avg_penalty,
                min_penalty,
                max_penalty,
                result.seconds_uniform * 1e3,
                result.seconds_divergent * 1e3));
    }
}

void shamsys::microbench::vector_allgather(u32 el_per_rank) {

    using T = u64;
    std::vector<u64> send_data(el_per_rank);

    std::vector<u64> recv_data;

    f64 t     = 0;
    u64 loops = 0;

    auto benchmark_step = [&]() {
        shamcomm::mpi::Barrier(MPI_COMM_WORLD);
        f64 t_start = MPI_Wtime();
        shamalgs::collective::vector_allgatherv(send_data, recv_data, MPI_COMM_WORLD);
        f64 t_end = MPI_Wtime();
        t += t_end - t_start;
        loops++;
    };

    do {
        benchmark_step();
    } while (shamalgs::collective::allreduce_min(t) < 0.1);

    t /= loops;

    f64 min_t = shamalgs::collective::allreduce_min(t);
    f64 max_t = shamalgs::collective::allreduce_max(t);
    f64 sum_t = shamalgs::collective::allreduce_sum(t);
    f64 avg_t = sum_t / (f64) shamcomm::world_size();

    microbench_results["vector_allgather_u64_" + std::to_string(el_per_rank)] = avg_t;

    if (shamcomm::world_rank() == 0) {
        logger::raw_ln(
            sham::format(
                " - vector_allgather (u64, n={:4}) : {:.3e} s (min = {:.2e}, max = {:.2e}, loops = "
                "{})",
                el_per_rank,
                avg_t,
                min_t,
                max_t,
                loops));
    }
}

const std::unordered_map<std::string, double> &shamsys::get_microbench_results(bool allow_run) {
    if (allow_run && microbench_results.empty()) {
        run_micro_benchmark();
    }
    return microbench_results;
}
