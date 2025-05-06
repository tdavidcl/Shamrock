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
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include <string>

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

    template<class T>
    class FieldSpan : public IDataEdgeNamed {
        public:
        using IDataEdgeNamed::IDataEdgeNamed;
        shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> spans;
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
