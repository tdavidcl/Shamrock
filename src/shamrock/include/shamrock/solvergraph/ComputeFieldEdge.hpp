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
 * @file ComputeFieldEdge.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/exception.hpp"
#include "shambase/sets.hpp"
#include "shamcomm/logs.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include <unordered_set>
#include <stdexcept>
#include <string>
#include <vector>

namespace shamrock::solvergraph {

    class IDataEdgeNamed : public IDataEdge {
        std::string name;
        std::string texsymbol;

        public:
        IDataEdgeNamed(std::string name, std::string texsymbol)
            : name(name), texsymbol(texsymbol) {}

        virtual std::string _impl_get_dot_label() { return name; }
        virtual std::string _impl_get_tex_symbol() { return texsymbol; }
    };

    template<class T1, class T2, class FuncMatch, class FuncMissing, class FuncExtra>
    inline void ensure_matching(
        shambase::DistributedData<T1> &dd,
        const shambase::DistributedData<T2> &reference,
        FuncMatch &&func_missing,
        FuncMissing &&func_match,
        FuncExtra &&func_extra) {

        std::vector<u64> dd_ids;
        std::vector<u64> ref_ids;

        dd.for_each([&](u32 id, T1 &data) {
            dd_ids.push_back(id);
        });

        reference.for_each([&](u32 id, const T2 &data) {
            ref_ids.push_back(id);
        });

        std::vector<u64> missing;
        std::vector<u64> matching;
        std::vector<u64> extra;

        shambase::set_diff(dd_ids, ref_ids, missing, matching, extra);

        for (auto id : missing) {
            func_missing(id);
        }

        for (auto id : matching) {
            func_match(id);
        }

        for (auto id : extra) {
            func_extra(id);
        }
    }

    template<class T>
    class FieldSpan : public IDataEdgeNamed {
        public:
        using IDataEdgeNamed::IDataEdgeNamed;
        shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> spans;

        inline void ensure_sizes(shambase::DistributedData<u32> &sizes) {
            ensure_matching(
                spans,
                sizes,
                [](u64 id) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Missing field span in distributed data at id " + std::to_string(id));
                },
                [](u64 id) {},
                [](u64 id) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Extra field span in distributed data at id " + std::to_string(id));
                });
        }
    };

    template<class Tint>
    class Indexes : public IDataEdgeNamed {
        public:
        using IDataEdgeNamed::IDataEdgeNamed;
        shambase::DistributedData<Tint> indexes;
    };

    template<class T, class Func>
    class EntryPointNode : public INode {
        Func functor;

        public:
        inline void set_ro_edges() { __internal_set_ro_edges({}); }
        inline void set_rw_edges(std::shared_ptr<T> attach_to) {
            __internal_set_rw_edges({attach_to});
        }

        EntryPointNode(Func &&f) : functor(std::forward<Func>(f)) {}

        inline void _impl_evaluate_internal() { functor(get_rw_edge<T>(0)); }
    };

} // namespace shamrock::solvergraph
