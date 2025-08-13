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
#include "shamrock/solvergraph/FieldSpan.hpp"
#include "shamrock/solvergraph/IDataEdgeNamed.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include <functional>

namespace shamrock::solvergraph {

    class PatchDataLayerRefs : public IDataEdgeNamed {

        public:
        shambase::DistributedData<std::reference_wrapper<patch::PatchDataLayer>> patchdatas;

        using IDataEdgeNamed::IDataEdgeNamed;

        inline virtual patch::PatchDataLayer &get(u64 id_patch) const {
            return patchdatas.get(id_patch);
        }

        inline virtual void free_alloc() { patchdatas = {}; }
    };

    class PatchDataLayerEdge : public IDataEdgeNamed {

        public:
        std::shared_ptr<patch::PatchDataLayerLayout> layout;
        shambase::DistributedData<patch::PatchDataLayer> patchdatas;

        using IDataEdgeNamed::IDataEdgeNamed;

        inline PatchDataLayerEdge(
            std::string name,
            std::string label,
            std::shared_ptr<patch::PatchDataLayerLayout> layout)
            : IDataEdgeNamed(name, label), layout(layout) {}

        inline virtual const patch::PatchDataLayer &get(u64 id_patch) const {
            return patchdatas.get(id_patch);
        }

        inline virtual patch::PatchDataLayer &get(u64 id_patch) { return patchdatas.get(id_patch); }

        inline virtual void free_alloc() { patchdatas = {}; }
    };

    class CopyPatchDataLayerFields : public INode {

        public:
        CopyPatchDataLayerFields(
            std::shared_ptr<patch::PatchDataLayerLayout> layout_source,
            std::shared_ptr<patch::PatchDataLayerLayout> layout_target)
            : layout_source(layout_source), layout_target(layout_target) {}

        std::shared_ptr<patch::PatchDataLayerLayout> layout_source;
        std::shared_ptr<patch::PatchDataLayerLayout> layout_target;

        struct Edges {
            const PatchDataLayerRefs &original;
            PatchDataLayerEdge &target;
        };

        void set_edges(
            std::shared_ptr<PatchDataLayerRefs> original,
            std::shared_ptr<PatchDataLayerEdge> target) {
            __internal_set_ro_edges({original});
            __internal_set_rw_edges({target});
        }

        Edges get_edges() {
            return Edges{get_ro_edge<PatchDataLayerRefs>(0), get_rw_edge<PatchDataLayerEdge>(0)};
        }

        void _impl_evaluate_internal() {
            auto edges = get_edges();

            // Ensures that the layout are all matching sources and targets
            edges.original.patchdatas.for_each([&](u64 id_patch, patch::PatchDataLayer &pdat) {
                if (pdat.get_layout_ptr().get() != layout_source.get()) {
                    throw shambase::make_except_with_loc<std::invalid_argument>("layout mismatch");
                }
            });

            if (edges.target.layout.get() != layout_target.get()) {
                throw shambase::make_except_with_loc<std::invalid_argument>("layout mismatch");
            }

            // Copy the fields from the original to the target
            edges.target.patchdatas = edges.original.patchdatas.map<patch::PatchDataLayer>(
                [&](u64 id_patch, patch::PatchDataLayer &pdat) {
                    patch::PatchDataLayer pdat_new(layout_target);

                    pdat_new.for_each_field_any([&](auto &field) {
                        using T = typename std::remove_reference<decltype(field)>::type::Field_type;
                        pdat_new.get_field<T>(field.get_name())
                            .insert(pdat.get_field<T>(field.get_name()));
                    });

                    pdat_new.check_field_obj_cnt_match();
                    return pdat_new;
                });
        }

        std::string _impl_get_label() { return "CopyPatchDataLayerFields"; }

        std::string _impl_get_tex() { return "TODO"; }
    };

    class PatchDataLayerEdgeToRefs : public INode {

        public:
        PatchDataLayerEdgeToRefs() {}

        /// Utility struct to list the refs to the edges
        struct Edges {
            PatchDataLayerEdge &original;
            PatchDataLayerRefs &refs;
        };

        /**
         * @brief Set the edges of the node
         *
         * Set the edge that will be freed by this node
         *
         * @param to_free The node to free
         */
        inline void set_edges(
            std::shared_ptr<PatchDataLayerEdge> original,
            std::shared_ptr<PatchDataLayerRefs> refs) {
            __internal_set_ro_edges({});
            __internal_set_rw_edges({refs, original});
        }

        /// Get the edges of the node
        inline Edges get_edges() {
            return Edges{get_rw_edge<PatchDataLayerEdge>(0), get_rw_edge<PatchDataLayerRefs>(1)};
        }

        /// Evaluate the node
        inline void _impl_evaluate_internal() {

            auto edges = get_edges();
            edges.refs.patchdatas
                = edges.original.patchdatas.map<std::reference_wrapper<patch::PatchDataLayer>>(
                    [](u64 id_patch, patch::PatchDataLayer &pdat) {
                        return std::ref(pdat);
                    });
        }

        /// Get the label of the node
        inline virtual std::string _impl_get_label() { return "PatchDataLayerEdgeToRefs"; };

        /// Get the TeX representation of the node
        inline virtual std::string _impl_get_tex() { return "TODO"; }
    };

} // namespace shamrock::solvergraph
