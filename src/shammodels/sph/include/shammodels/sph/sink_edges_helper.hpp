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
 * @file sink_edges_helper.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Helpers to access SPH sink particles stored as SoA synchronized data edges.
 *
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambase/type_name_info.hpp"
#include "shambackends/type_traits.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/SinkPartStruct.hpp"
#include "shamsolvergraph/SolverGraphSerializable.hpp"
#include "shamsolvergraph/edge/IDataEdgeSerializable.hpp"
#include <algorithm>
#include <vector>

namespace shambase {
    template<class T>
    struct TypeNameInfo<std::vector<T>> {
        inline static const std::string name = "std::vector<" + get_type_name<T>() + ">";
    };
} // namespace shambase

namespace shammodels::sph {

    template<class Tvec>
    struct SinkEdges {
        using Tscal = shambase::VecComponent<Tvec>;

        std::vector<Tvec> &pos;
        std::vector<Tvec> &vel;
        std::vector<Tvec> &acc_sph;
        std::vector<Tvec> &acc_ext;
        std::vector<Tscal> &mass;
        std::vector<Tvec> &angular_momentum;
        std::vector<Tscal> &accretion_radius;

        /// Accreted dust mass per species, flattened as [i_sink * ndust + j_dust]
        std::vector<Tscal> &accreted_dust_mass;

        inline bool has_sinks() const { return !pos.empty(); }
        inline size_t size() const { return pos.size(); }
    };

    namespace details {

        template<class T>
        inline void ensure_sync_data_edge(
            shamrock::solvergraph::SolverGraphSerializable &sync_data,
            const std::string &name,
            const std::string &tex_symbol,
            T init_value) {

            if (!sync_data.has_edge(name)) {
                auto edge = sync_data.register_edge(
                    name, shamrock::solvergraph::IDataEdgeSerializable<T>(name, tex_symbol));
                edge->data = std::move(init_value);
            }
        }

        template<class T>
        inline T &get_sync_data(
            shamrock::solvergraph::SolverGraphSerializable &sync, const std::string &name) {
            return sync.template get_edge_ref<shamrock::solvergraph::IDataEdgeSerializable<T>>(name)
                .data;
        }

    } // namespace details

    /**
     * @brief Register sink SoA synchronized edges if missing (idempotent).
     */
    template<class Tvec>
    inline void ensure_sink_edges(shamrock::solvergraph::SolverGraphSerializable &sync) {
        using Tscal = shambase::VecComponent<Tvec>;

        details::ensure_sync_data_edge<std::vector<Tvec>>(
            sync, "sink_pos", "\\bf{r}_{\\mathrm{sink}}", {});
        details::ensure_sync_data_edge<std::vector<Tvec>>(
            sync, "sink_vel", "\\bf{v}_{\\mathrm{sink}}", {});
        details::ensure_sync_data_edge<std::vector<Tvec>>(
            sync, "sink_acc_sph", "\\bf{a}_{\\mathrm{sink, sph}}", {});
        details::ensure_sync_data_edge<std::vector<Tvec>>(
            sync, "sink_acc_ext", "\\bf{a}_{\\mathrm{sink, ext}}", {});
        details::ensure_sync_data_edge<std::vector<Tscal>>(
            sync, "sink_mass", "m_{\\mathrm{sink}}", {});
        details::ensure_sync_data_edge<std::vector<Tvec>>(
            sync, "sink_angular_momentum", "\\bf{L}_{\\mathrm{sink}}", {});
        details::ensure_sync_data_edge<std::vector<Tscal>>(
            sync, "sink_accretion_radius", "r_{\\mathrm{accretion}}", {});
        details::ensure_sync_data_edge<std::vector<Tscal>>(
            sync, "sink_accreted_dust_mass", "m_{\\mathrm{sink, acc}, j}", {});
    }

    /**
     * @brief Named SoA getters (edges must already exist; call ensure_sink_edges first).
     * Prefer these when a function only needs a subset of sink fields.
     */
    template<class Tvec>
    inline std::vector<Tvec> &get_sink_pos(shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<Tvec>>(sync, "sink_pos");
    }

    template<class Tvec>
    inline std::vector<Tvec> &get_sink_vel(shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<Tvec>>(sync, "sink_vel");
    }

    template<class Tvec>
    inline std::vector<Tvec> &get_sink_acc_sph(
        shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<Tvec>>(sync, "sink_acc_sph");
    }

    template<class Tvec>
    inline std::vector<Tvec> &get_sink_acc_ext(
        shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<Tvec>>(sync, "sink_acc_ext");
    }

    template<class Tvec>
    inline std::vector<shambase::VecComponent<Tvec>> &get_sink_mass(
        shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<shambase::VecComponent<Tvec>>>(sync, "sink_mass");
    }

    template<class Tvec>
    inline std::vector<Tvec> &get_sink_angular_momentum(
        shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<Tvec>>(sync, "sink_angular_momentum");
    }

    template<class Tvec>
    inline std::vector<shambase::VecComponent<Tvec>> &get_sink_accretion_radius(
        shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<shambase::VecComponent<Tvec>>>(
            sync, "sink_accretion_radius");
    }

    template<class Tvec>
    inline std::vector<shambase::VecComponent<Tvec>> &get_sink_accreted_dust_mass(
        shamrock::solvergraph::SolverGraphSerializable &sync) {
        return details::get_sync_data<std::vector<shambase::VecComponent<Tvec>>>(
            sync, "sink_accreted_dust_mass");
    }

    /**
     * @brief Resize the flattened per species accreted dust mass to sink_count * ndust.
     *
     * Only pads with zeros the sinks that have no entry yet (sinks added since the last call, or
     * loaded from a dump that predates the tracking). Throws if the stored layout does not match
     * ndust, e.g. if the number of dust species changed after dust has been tracked.
     */
    template<class Tscal>
    inline void resize_sink_accreted_dust_mass(
        std::vector<Tscal> &accreted_dust_mass, size_t sink_count, u32 ndust) {
        size_t target_size = sink_count * ndust;
        if (accreted_dust_mass.size() == target_size) {
            return;
        }
        if (ndust == 0 || accreted_dust_mass.size() > target_size
            || accreted_dust_mass.size() % ndust != 0) {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "sink accreted dust mass layout mismatch: size={}, sink_count={}, ndust={}",
                accreted_dust_mass.size(),
                sink_count,
                ndust));
        }
        accreted_dust_mass.resize(target_size, Tscal{0});
    }

    /**
     * @brief Number of dust species tracked per sink in the flattened accreted dust mass.
     */
    template<class Tscal>
    inline size_t get_sink_accreted_dust_nvar(
        const std::vector<Tscal> &accreted_dust_mass, size_t sink_count) {
        if (sink_count == 0 || accreted_dust_mass.empty()) {
            return 0;
        }
        if (accreted_dust_mass.size() % sink_count != 0) {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "sink accreted dust mass size ({}) is not a multiple of the sink count ({})",
                accreted_dust_mass.size(),
                sink_count));
        }
        return accreted_dust_mass.size() / sink_count;
    }

    /**
     * @brief Check whether any sinks are present by inspecting sink_pos only.
     *
     * Prefer this for early-exit checks instead of fetching the full SinkEdges struct.
     * Edges must already exist (call ensure_sink_edges first).
     */
    template<class Tvec>
    inline bool has_sinks(shamrock::solvergraph::SolverGraphSerializable &sync) {
        return !get_sink_pos<Tvec>(sync).empty();
    }

    /**
     * @brief Fetch mutable references to the sink SoA synchronized edges.
     *
     * Edges must already exist (call ensure_sink_edges first).
     * Prefer named getters when only a subset of fields is needed.
     */
    template<class Tvec>
    inline SinkEdges<Tvec> get_sink_edges(shamrock::solvergraph::SolverGraphSerializable &sync) {
        return SinkEdges<Tvec>{
            get_sink_pos<Tvec>(sync),
            get_sink_vel<Tvec>(sync),
            get_sink_acc_sph<Tvec>(sync),
            get_sink_acc_ext<Tvec>(sync),
            get_sink_mass<Tvec>(sync),
            get_sink_angular_momentum<Tvec>(sync),
            get_sink_accretion_radius<Tvec>(sync),
            get_sink_accreted_dust_mass<Tvec>(sync),
        };
    }

    /**
     * @brief Build an AoS sink list from the current SoA edges (Python API / dump helpers).
     */
    template<class Tvec>
    inline std::vector<SinkParticle<Tvec>> to_sink_particles(const SinkEdges<Tvec> &e) {
        size_t ndust = get_sink_accreted_dust_nvar(e.accreted_dust_mass, e.size());

        std::vector<SinkParticle<Tvec>> out;
        out.reserve(e.size());
        for (size_t i = 0; i < e.size(); i++) {
            auto dust_begin = e.accreted_dust_mass.begin() + i * ndust;
            out.push_back(
                SinkParticle<Tvec>{
                    e.pos[i],
                    e.vel[i],
                    e.acc_sph[i],
                    e.acc_ext[i],
                    e.mass[i],
                    e.angular_momentum[i],
                    e.accretion_radius[i],
                    {dust_begin, dust_begin + ndust},
                });
        }
        return out;
    }

    /**
     * @brief Replace SoA sink edge contents from an AoS sink list (legacy dump migration).
     */
    template<class Tvec>
    inline void set_sink_particles(
        SinkEdges<Tvec> &e, const std::vector<SinkParticle<Tvec>> &sinks) {
        e.pos.clear();
        e.vel.clear();
        e.acc_sph.clear();
        e.acc_ext.clear();
        e.mass.clear();
        e.angular_momentum.clear();
        e.accretion_radius.clear();
        e.accreted_dust_mass.clear();

        e.pos.reserve(sinks.size());
        e.vel.reserve(sinks.size());
        e.acc_sph.reserve(sinks.size());
        e.acc_ext.reserve(sinks.size());
        e.mass.reserve(sinks.size());
        e.angular_momentum.reserve(sinks.size());
        e.accretion_radius.reserve(sinks.size());

        for (const auto &s : sinks) {
            e.pos.push_back(s.pos);
            e.vel.push_back(s.velocity);
            e.acc_sph.push_back(s.sph_acceleration);
            e.acc_ext.push_back(s.ext_acceleration);
            e.mass.push_back(s.mass);
            e.angular_momentum.push_back(s.angular_momentum);
            e.accretion_radius.push_back(s.accretion_radius);
        }

        // sinks without dust tracking (e.g. legacy dumps) are zero padded to the widest one
        size_t ndust = 0;
        for (const auto &s : sinks) {
            ndust = std::max(ndust, s.accreted_dust_mass.size());
        }
        e.accreted_dust_mass.assign(sinks.size() * ndust, 0);
        for (size_t i = 0; i < sinks.size(); i++) {
            const auto &dust = sinks[i].accreted_dust_mass;
            if (!dust.empty() && dust.size() != ndust) {
                throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                    "sink {} tracks {} dust species, expected {}", i, dust.size(), ndust));
            }
            std::copy(dust.begin(), dust.end(), e.accreted_dust_mass.begin() + i * ndust);
        }
    }

    /**
     * @brief Append one sink to the SoA edges.
     *
     * ndust is the number of dust species whose accreted mass is tracked (0 without dust).
     */
    template<class Tvec>
    inline void add_sink(
        SinkEdges<Tvec> &e,
        typename SinkEdges<Tvec>::Tscal mass,
        Tvec pos,
        Tvec velocity,
        typename SinkEdges<Tvec>::Tscal accretion_radius,
        u32 ndust) {

        e.pos.push_back(pos);
        e.vel.push_back(velocity);
        e.acc_sph.push_back({});
        e.acc_ext.push_back({});
        e.mass.push_back(mass);
        e.angular_momentum.push_back({});
        e.accretion_radius.push_back(accretion_radius);
        resize_sink_accreted_dust_mass(e.accreted_dust_mass, e.size(), ndust);
    }

} // namespace shammodels::sph
