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
 * @file Solver.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/print.hpp"
#include "SolverConfig.hpp"
#include "sham/format/format.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/SolverLog.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/SPHUtilities.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamsolvergraph/edge/IDataEdgeSerializable.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/TreeTraversalCache.hpp"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>
namespace shammodels::sph {

    struct TimestepLog {
        i32 rank;
        f64 rate;
        u64 npart;
        f64 tcompute;

        inline f64 rate_sum() { return shamalgs::collective::allreduce_sum(rate); }

        inline u64 npart_sum() { return shamalgs::collective::allreduce_sum(npart); }

        inline f64 tcompute_max() { return shamalgs::collective::allreduce_max(tcompute); }
    };

    struct EvolveUntilResults {
        bool reach_target_time;
        bool reach_niter_max;
        bool reach_max_walltime;

        i32 iter_count;
    };

    /**
     * @brief The shamrock SPH model
     *
     * @tparam Tvec
     * @tparam SPHKernel
     */
    template<class Tvec, template<class> class SPHKernel>
    class Solver {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config = SolverConfig<Tvec, SPHKernel>;

        using u_morton = typename Config::u_morton;

        static constexpr Tscal Rkern = Kernel::Rkern;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        SolverStorage<Tvec, u_morton> storage{};

        Config solver_config;
        SolverLog solve_logs;

        /// Access synchronized simulation time (scheduler edge "time")
        inline Tscal &time_edge_value() {
            return scheduler()
                .synchronized_data
                .template get_edge_ref<shamrock::solvergraph::IDataEdgeSerializable<Tscal>>("time")
                .data;
        }

        /// Access synchronized next dt (scheduler edge "dt", not solver_graph "dt")
        inline Tscal &dt_edge_value() {
            return scheduler()
                .synchronized_data
                .template get_edge_ref<shamrock::solvergraph::IDataEdgeSerializable<Tscal>>("dt")
                .data;
        }

        /// Access synchronized CFL multiplier (scheduler edge "cfl_multiplier")
        inline Tscal &cfl_multiplier_edge_value() {
            return scheduler()
                .synchronized_data
                .template get_edge_ref<shamrock::solvergraph::IDataEdgeSerializable<Tscal>>(
                    "cfl_multiplier")
                .data;
        }

        inline Tscal get_time() { return time_edge_value(); }
        inline void set_time(Tscal t) { time_edge_value() = t; }
        inline Tscal get_dt_sph() { return dt_edge_value(); }
        inline void set_next_dt(Tscal dt) { dt_edge_value() = dt; }
        inline Tscal get_cfl_multipler() { return cfl_multiplier_edge_value(); }
        inline void set_cfl_multipler(Tscal lambda) { cfl_multiplier_edge_value() = lambda; }

        /// Register time/dt/cfl_multiplier synchronized edges if missing (idempotent)
        inline void ensure_time_state_edges() {
            auto &sync    = scheduler().synchronized_data;
            auto names    = sync.get_edge_names();
            auto has_edge = [&](const std::string &name) {
                return std::find(names.begin(), names.end(), name) != names.end();
            };

            if (!has_edge("time")) {
                auto edge = sync.register_edge(
                    "time", shamrock::solvergraph::IDataEdgeSerializable<Tscal>("time", "t"));
                edge->data = 0;
            }
            if (!has_edge("dt")) {
                auto edge = sync.register_edge(
                    "dt", shamrock::solvergraph::IDataEdgeSerializable<Tscal>("dt", "dt"));
                edge->data = 0;
            }
            if (!has_edge("cfl_multiplier")) {
                auto edge = sync.register_edge(
                    "cfl_multiplier",
                    shamrock::solvergraph::IDataEdgeSerializable<Tscal>(
                        "cfl_multiplier", "C_{\\rm CFL}"));
                edge->data = 1e-2;
            }
        }

        struct SolverStepCallback {
            std::optional<std::function<void(void)>> step_begin_callback;
            std::optional<std::function<void(void)>> step_end_callback;
        };
        std::vector<SolverStepCallback> timestep_callbacks{};

        inline void init_required_fields() { solver_config.set_layout(context.get_pdl_write()); }

        // serial patch tree control
        void gen_serial_patch_tree();
        inline void reset_serial_patch_tree() { storage.serial_patch_tree.reset(); }

        // interface_control
        using GhostHandle      = sph::BasicSPHGhostHandler<Tvec>;
        using GhostHandleCache = typename GhostHandle::CacheMap;

        inline void gen_ghost_handler(Tscal time_val) {

            using CfgClass = sph::BasicSPHGhostHandlerConfig<Tvec>;
            using BCConfig = typename CfgClass::Variant;

            using BCFree             = typename CfgClass::Free;
            using BCPeriodic         = typename CfgClass::Periodic;
            using BCShearingPeriodic = typename CfgClass::ShearingPeriodic;

            using SolverConfigBC           = typename Config::BCConfig;
            using SolverBCFree             = typename SolverConfigBC::Free;
            using SolverBCPeriodic         = typename SolverConfigBC::Periodic;
            using SolverBCShearingPeriodic = typename SolverConfigBC::ShearingPeriodic;

            // boundary condition selections
            if (SolverBCFree *c
                = std::get_if<SolverBCFree>(&solver_config.boundary_config.config)) {
                storage.ghost_handler.set(
                    GhostHandle{
                        scheduler(),
                        BCFree{},
                        storage.patch_rank_owner,
                        storage.xyzh_ghost_layout});
            } else if (
                SolverBCPeriodic *c
                = std::get_if<SolverBCPeriodic>(&solver_config.boundary_config.config)) {
                storage.ghost_handler.set(
                    GhostHandle{
                        scheduler(),
                        BCPeriodic{},
                        storage.patch_rank_owner,
                        storage.xyzh_ghost_layout});
            } else if (
                SolverBCShearingPeriodic *c
                = std::get_if<SolverBCShearingPeriodic>(&solver_config.boundary_config.config)) {
                storage.ghost_handler.set(
                    GhostHandle{
                        scheduler(),
                        BCShearingPeriodic{
                            c->shear_base, c->shear_dir, c->shear_speed * time_val, c->shear_speed},
                        storage.patch_rank_owner,
                        storage.xyzh_ghost_layout});
            }
        }
        inline void reset_ghost_handler() { storage.ghost_handler.reset(); }

        /// @brief Builds ghost particle interface cache for inter-patch communication
        void build_ghost_cache();
        /// @brief Clears ghost particle cache to free memory
        void clear_ghost_cache();

        /// @brief Merges ghost particle positions from neighboring patches
        void merge_position_ghost();

        // trees
        using RTree = typename Config::RTree;
        /// @brief Builds spatial BVH trees for merged positions including ghosts
        void build_merged_pos_trees();
        /// @brief Clears merged position trees to free memory
        void clear_merged_pos_trees();

        /// @brief Computes maximum smoothing length in tree nodes for neighbor search
        void compute_presteps_rint();
        /// @brief Resets tree radius interval field
        void reset_presteps_rint();

        /// @brief Builds neighbor particle cache for SPH calculations
        void start_neighbors_cache();
        /// @brief Resets neighbor cache
        void reset_neighbors_cache();

        /// @brief Performs pre-step operations for SPH timestep
        void sph_prestep(Tscal time_val, Tscal dt);

        /// @brief Applies position-based boundary conditions
        void apply_position_boundary(Tscal time_val);

        /// @brief Updates artificial viscosity coefficients for shock capturing
        void update_artificial_viscosity(Tscal dt);

        /// @brief Initializes data layout for ghost particle fields
        void init_ghost_layout();

        /// @brief Communicates and merges ghost particle fields across processes
        void communicate_merge_ghosts_fields();
        /// @brief Resets merged ghost field data
        void reset_merge_ghosts_fields();

        /// @brief Computes equation of state fields (pressure, sound speed)
        void compute_eos_fields();

        /// @brief Frees memory allocated for EOS fields
        void reset_eos_fields();

        /// @brief Saves old derivative fields for predictor-corrector integration
        void prepare_corrector();
        /// @brief Updates time derivatives and applies external forces
        void update_derivs(Tscal dt_hydro);
        /**
         * @brief
         *
         * @return true corrector is converged
         * @return false corrector is not converged
         */
        bool apply_corrector(Tscal dt, u64 Npart_all);

        /// @brief Updates load balancing values and synchronizes patch ownership
        void update_sync_load_values();

        Solver(ShamrockCtx &context) : context(context) {}

        /// @brief Initializes the solver graph for computation pipeline
        void init_solver_graph();

        /// @brief Writes VTK dump file for visualization
        void vtk_do_dump(std::string filename, bool add_patch_world_id);

        void set_debug_dump(bool _do_debug_dump, std::string _debug_dump_filename) {
            solver_config.set_debug_dump(_do_debug_dump, _debug_dump_filename);
        }

        inline void print_timestep_logs() {
            if (shamcomm::world_rank() == 0) {
                logger::info_ln("SPH", "iteration since start :", solve_logs.get_iteration_count());
                logger::info_ln("SPH", "time since start :", shambase::details::get_wtime(), "(s)");
            }
        }

        /// @brief Performs one complete SPH timestep evolution
        TimestepLog evolve_once();

        /// @brief Evolves system by one explicit timestep with specified time and dt
        Tscal evolve_once_time_expl(Tscal t_current, Tscal dt_input) {
            set_time(t_current);
            set_next_dt(dt_input);
            evolve_once();
            return get_dt_sph();
        }

        private:
        /**
         * @brief Tracks the wall-clock budget for evolve_until(): decides when the next
         * walltime check is due, and whether the limit has been exceeded.
         *
         * NFC extraction of logic previously inlined in evolve_until().
         */
        struct WalltimeLimiter {
            bool active;
            f64 max_walltime;
            f64 start_wall_time;
            i32 next_check_iter;

            inline WalltimeLimiter(bool active, f64 max_walltime)
                : active(active), max_walltime(max_walltime) {
                start_wall_time = active ? synced_wtime() : 0;
                next_check_iter = active ? 1 : std::numeric_limits<i32>::max();
            }

            inline f64 synced_wtime() {
                if (active) {
                    return shamalgs::collective::allreduce_max(shambase::details::get_wtime());
                }
                return 0;
            }

            /// Rank-local elapsed walltime since start (no MPI reduction, for display only)
            inline f64 elapsed_local() const {
                return active ? (shambase::details::get_wtime() - start_wall_time) : 0;
            }

            /// True if the next walltime check is due at this iteration count
            inline bool due(i32 iter_count) const {
                return active && iter_count >= next_check_iter;
            }

            /// Must only be called when due(iter_count) is true. Returns true if the walltime
            /// limit has been reached, otherwise updates the next check iteration estimate.
            inline bool exceeded(i32 iter_count) {
                f64 global_walltime = synced_wtime();

                // if the global walltime is greater than the max walltime
                if (global_walltime >= max_walltime) {
                    if (shamcomm::world_rank() == 0) {
                        logger::info_ln(
                            "SPH",
                            sham::format(
                                "stopping evolve until because of "
                                "max_walltime = {:.2f}s > {:.2f}s",
                                global_walltime,
                                max_walltime));
                    }
                    return true;
                }

                f64 sec_per_iter
                    = (global_walltime - start_wall_time) / static_cast<f64>(iter_count);

                auto get_remaining_iters = [&](f64 delta_walltime, f64 factor) -> i32 {
                    if (sec_per_iter > 0) {
                        f64 tmp = factor * delta_walltime / sec_per_iter;
                        if (tmp > std::numeric_limits<i32>::max()) {
                            return std::numeric_limits<i32>::max();
                        }
                        return static_cast<i32>(tmp);
                    }
                    return 1000; // default to 1000 iterations if sec_per_iter is 0
                };

                i32 iters_to_limit      = get_remaining_iters(max_walltime - global_walltime, 0.25);
                i32 iters_to_next_check = iters_to_limit;

                next_check_iter = iter_count + std::max(1, iters_to_next_check);

                if (shamcomm::world_rank() == 0) {
                    logger::info_ln(
                        "SPH",
                        sham::format(
                            "next walltime check in {:.2f}s (niter = {}) global walltime = "
                            "{:.2f}s (max_walltime = {:.2f}s)",
                            iters_to_next_check * sec_per_iter,
                            iters_to_next_check,
                            global_walltime,
                            max_walltime));
                }

                return false;
            }
        };

        public:
        inline EvolveUntilResults evolve_until(
            Tscal target_time, i32 niter_max, f64 max_walltime = -1) {

            const Tscal t_start = get_time();

            const bool niter_limit_active    = (niter_max >= 0);
            const bool walltime_limit_active = (max_walltime >= 0);

            if (shamcomm::world_rank() == 0) {
                logger::info_ln(
                    "SPH",
                    sham::format(
                        "evolve_until (target_time = {:.2f}s, niter_max = {}, max_walltime = "
                        "{:.2f}s)",
                        target_time,
                        niter_max,
                        max_walltime));
            }

            auto step = [&]() {
                Tscal dt = get_dt_sph();
                Tscal t  = get_time();

                if (t > target_time) {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "the target time is higher than the current time");
                }

                if (t + dt > target_time) {
                    set_next_dt(target_time - t);
                }
                return evolve_once();
            };

            WalltimeLimiter walltime_limiter(walltime_limit_active, max_walltime);

            i32 iter_count = 0;

            struct SelfUpdatingLogBlock {
                uint64_t last_print_counter = shambase::get_max<u64>();

                void print(const std::string &s, u64 line_count) {
                    bool has_printed_since = last_print_counter != shambase::print_counter();

                    if (has_printed_since) {
                        logger::raw_ln("-----------------------------------");
                    } else {
                        std::string clear_seq = "\x1b[K";
                        for (u64 i = 0; i < line_count + 1; i++) {
                            clear_seq += "\x1b[1A\x1b[K";
                        }
                        shambase::print(clear_seq);
                    }
                    logger::raw_ln(s, "\n----------------------------");

                    last_print_counter = shambase::print_counter();
                }
            };

            static SelfUpdatingLogBlock block = {};

            auto make_progress_bar = [](Tscal current, Tscal target, u32 width) -> std::string {
                f64 frac = (target > 0) ? f64(current) / f64(target) : 0.0;
                frac     = std::clamp(frac, 0.0, 1.0);

                u32 filled = static_cast<u32>(frac * width);
                std::string bar_str;
                bar_str.reserve(width + 8);
                bar_str += "[";
                bar_str += std::string(filled, '#');
                bar_str += std::string(width - filled, '-');
                bar_str += "]";
                bar_str += sham::format(" {:5.1f}%", frac * 100);
                return bar_str;
            };

            std::optional<TimestepLog> last_step_log = {};

            auto update_state = [&]() {
                std::string stats_str
                    = last_step_log ? sham::format(
                                          "rate = {:.3e} npart = {} tcompute = {:e}",
                                          last_step_log->rate,
                                          f64(last_step_log->npart),
                                          last_step_log->tcompute)
                                    : "rate = ....... npart = ....... tcompute = ....... [s]";

                std::string tsimhr_str
                    = (last_step_log && last_step_log->tcompute > 0)
                          ? sham::format(
                                " tsim/hr = {:.3e} [t/hr]",
                                get_dt_sph() * (3600.0 / last_step_log->tcompute))
                          : " tsim/hr = ....... [t/hr]";

                std::vector<std::string> criteria;
                criteria.push_back(
                    sham::format("{:.3e}/{:.3e}", get_time() - t_start, target_time - t_start));
                if (niter_limit_active) {
                    criteria.push_back(sham::format("{}/{}", iter_count, niter_max));
                }
                if (walltime_limit_active) {
                    criteria.push_back(
                        sham::format(
                            "{:.2f}/{:.2f}",
                            walltime_limiter.elapsed_local(),
                            max_walltime - walltime_limiter.start_wall_time));
                }

                tsimhr_str += " ";
                for (size_t i = 0; i < criteria.size(); i++) {
                    if (i > 0) {
                        tsimhr_str += " | ";
                    }
                    tsimhr_str += criteria[i];
                }

                block.print(
                    sham::format(
                        "t = {:.5e} dt = {:.5e} {}\n{}{}",
                        get_time(),
                        get_dt_sph(),
                        stats_str,
                        make_progress_bar(get_time() - t_start, target_time - t_start, 40),
                        tsimhr_str),
                    2);
            };

            while (get_time() < target_time) {

                update_state();

                last_step_log = step();
                iter_count++;

                // if the iteration count is greater than the maximum iteration count
                if (niter_limit_active && iter_count >= niter_max) {
                    if (shamcomm::world_rank() == 0) {
                        logger::info_ln(
                            "SPH", "stopping evolve until because of niter =", iter_count);
                    }
                    return {
                        .reach_target_time  = false,
                        .reach_niter_max    = true,
                        .reach_max_walltime = false,
                        .iter_count         = iter_count,
                    };
                }

                // if walltime limit is active and the next walltime check is due
                if (walltime_limiter.due(iter_count) && walltime_limiter.exceeded(iter_count)) {
                    return {
                        .reach_target_time  = false,
                        .reach_niter_max    = false,
                        .reach_max_walltime = true,
                        .iter_count         = iter_count,
                    };
                }
            }

            update_state();

            print_timestep_logs();

            return {
                .reach_target_time  = true,
                .reach_niter_max    = false,
                .reach_max_walltime = false,
                .iter_count         = iter_count,
            };
        }
    };

} // namespace shammodels::sph
