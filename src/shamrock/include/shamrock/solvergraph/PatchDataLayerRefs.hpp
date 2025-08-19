// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PatchDataLayerRefs.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the PatchDataLayerRefs class for managing distributed references to patch data
 * layers.
 *
 */

#include "shambase/DistributedData.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/FieldSpan.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include <functional>

namespace shamrock::solvergraph {

    using PatchDataLayerRef = std::reference_wrapper<patch::PatchDataLayer>;

    class IPatchDataLayerRefs : public IDataEdgeNamed {

        public:
        using IDataEdgeNamed::IDataEdgeNamed;

        virtual patch::PatchDataLayer &get(u64 id_patch)             = 0;
        virtual const patch::PatchDataLayer &get(u64 id_patch) const = 0;

        virtual const shambase::DistributedData<PatchDataLayerRef> &get_const_refs() const = 0;
        virtual shambase::DistributedData<PatchDataLayerRef> &get_refs()                   = 0;
    };

    class PatchDataLayerRefs : public IPatchDataLayerRefs {

        public:
        shambase::DistributedData<PatchDataLayerRef> patchdatas;

        using IPatchDataLayerRefs::IPatchDataLayerRefs;

        inline virtual patch::PatchDataLayer &get(u64 id_patch) { return patchdatas.get(id_patch); }

        inline virtual const patch::PatchDataLayer &get(u64 id_patch) const {
            return patchdatas.get(id_patch);
        }

        inline virtual shambase::DistributedData<PatchDataLayerRef> &get_refs() {
            return patchdatas;
        }

        inline virtual const shambase::DistributedData<PatchDataLayerRef> &get_const_refs() const {
            return patchdatas;
        }

        inline virtual void free_alloc() { patchdatas = {}; }
    };

} // namespace shamrock::solvergraph
