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
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include <unordered_set>
#include <stdexcept>
#include <string>
#include <vector>

namespace shamrock::solvergraph {

    template<class T>
    class FieldSpan : public IDataEdgeNamed {
        public:
        using IDataEdgeNamed::IDataEdgeNamed;
        shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> spans;

        inline virtual void check_sizes(const shambase::DistributedData<u32> &sizes) const {
            on_distributeddata_diff(
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

        inline virtual void ensure_sizes(const shambase::DistributedData<u32> &sizes) {
            check_sizes(sizes);
        }
    };

    template<class T>
    class Field : public FieldSpan<T> {

        u32 nvar;
        std::string name;
        ComputeField<T> field;

        public:
        Field(u32 nvar, std::string name, std::string texsymbol)
            : nvar(nvar), name(name), FieldSpan<T>(name, texsymbol) {}

        // overload only the non
        inline virtual void ensure_sizes(const shambase::DistributedData<u32> &sizes) {

            auto new_patchdatafield = [&](u32 size) {
                auto ret = PatchDataField<T>(name, nvar);
                ret.resize(size);
                return ret;
            };

            auto ensure_patchdatafield_sizes = [&](u32 size, auto &pdatfield) {
                if (pdatfield.get_obj_cnt() != size) {
                    pdatfield.resize(size);
                }
            };

            on_distributeddata_diff(
                this->spans,
                sizes,
                [&](u64 id) {
                    field.field_data.add_obj(id, new_patchdatafield(sizes.get(id)));
                },
                [&](u64 id) {
                    ensure_patchdatafield_sizes(sizes.get(id), field.field_data.get(id));
                },
                [&](u64 id) {
                    field.field_data.erase(id);
                });

            this->spans = field.field_data.template map<shamrock::PatchDataFieldSpanPointer<T>>(
                [&](u64 id, PatchDataField<T> &pdf) {
                    return pdf.get_pointer_span();
                });
        }

        inline ComputeField<T> extract() { return std::move(field); }
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
