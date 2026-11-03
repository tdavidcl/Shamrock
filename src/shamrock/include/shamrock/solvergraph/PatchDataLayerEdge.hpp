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
 * @file PatchDataLayerEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the PatchDataLayerEdge class for managing patch data layer edges.
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/memory.hpp"
#include "shambase/narrowing.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"
#include "shamsolvergraph/edge/IEdgeNamed.hpp"
#include <memory>

namespace shamrock::solvergraph {

    class PatchDataLayerEdge : public IPatchDataLayerRefs {

        using DDPatchDataLayerRef = shambase::DistributedData<PatchDataLayerRef>;

        std::shared_ptr<patch::PatchDataLayerLayout> layout;
        shambase::DistributedData<patch::PatchDataLayer> patchdatas;
        shambase::DistributedData<PatchDataLayerRef> patchdatas_refs;

        public:
        using IPatchDataLayerRefs::IPatchDataLayerRefs;

        inline PatchDataLayerEdge(
            const std::string &name,
            const std::string &label,
            std::shared_ptr<patch::PatchDataLayerLayout> layout)
            : IPatchDataLayerRefs(name, label), layout(layout) {}

        inline void set_patchdatas(shambase::DistributedData<patch::PatchDataLayer> &&src) {
            patchdatas      = std::move(src);
            patchdatas_refs = patchdatas.map<PatchDataLayerRef>(
                [](u64 id, patch::PatchDataLayer &layer) -> PatchDataLayerRef {
                    return std::ref(layer);
                });
        }

        /**
         * @brief Make the edge hold one layer per patch in sizes, with the given object counts
         *
         * Layers already held for a patch are resized instead of being recreated, so their
         * allocations are reused between steps. Missing layers are created with the layout of the
         * edge and layers of patches absent from sizes are dropped.
         *
         * The storage of a layer is never shrunk: this edge typically holds patches to which ghost
         * zones are appended after this call, so shrinking would only lead to a reallocation when
         * the ghost zones are appended back. The content of the resized layers is unspecified and
         * must be overwritten by the caller.
         *
         * @param sizes the object count of each patch
         */
        inline void ensure_sizes(const shambase::DistributedData<u32> &sizes) {

            auto resize_layer = [](patch::PatchDataLayer &layer, u32 obj_cnt) {
                layer.for_each_field_any([&](auto &field) {
                    field.get_buf().resize_no_shrink(
                        shambase::narrow_or_throw<u32>(u64(obj_cnt) * u64(field.get_nvar())),
                        false);
                });
            };

            shambase::on_distributeddata_diff(
                patchdatas,
                sizes,
                [&](u64 id) {
                    patch::PatchDataLayer layer(layout);
                    resize_layer(layer, sizes.get(id));
                    patchdatas.add_obj(id, std::move(layer));
                },
                [&](u64 id) {
                    patch::PatchDataLayer &layer = patchdatas.get(id);
                    if (layer.get_layout_ptr().get() != layout.get()) {
                        layer = patch::PatchDataLayer(layout);
                    }
                    resize_layer(layer, sizes.get(id));
                },
                [&](u64 id) {
                    patchdatas.erase(id);
                });

            patchdatas_refs = patchdatas.map<PatchDataLayerRef>(
                [](u64 id, patch::PatchDataLayer &layer) -> PatchDataLayerRef {
                    return std::ref(layer);
                });
        }

        inline shambase::DistributedData<patch::PatchDataLayer> extract_patchdatas() {
            auto tmp = std::move(patchdatas);
            set_patchdatas({});
            return tmp;
        }

        inline const patch::PatchDataLayerLayout &pdl() const {
            return shambase::get_check_ref(layout);
        }

        inline std::shared_ptr<patch::PatchDataLayerLayout> &get_layout_ptr() { return layout; }

        inline virtual patch::PatchDataLayer &get(u64 id_patch) override {
            return patchdatas.get(id_patch);
        }

        inline virtual const patch::PatchDataLayer &get(u64 id_patch) const override {
            return patchdatas.get(id_patch);
        }

        inline virtual shambase::DistributedData<PatchDataLayerRef> &get_refs() override {
            return patchdatas_refs;
        }

        inline virtual const shambase::DistributedData<PatchDataLayerRef> &get_const_refs()
            const override {
            return patchdatas_refs;
        }

        inline virtual void free_alloc() override {
            layout          = {};
            patchdatas      = {};
            patchdatas_refs = {};
        }
    };
} // namespace shamrock::solvergraph
