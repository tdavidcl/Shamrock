// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file PatchTreeNode.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief json (de)serialization of the patch tree nodes
 *
 */

#include "shamrock/scheduler/PatchTreeNode.hpp"
#include <nlohmann/json.hpp>

namespace shamrock::scheduler {

    void to_json(nlohmann::json &j, const LinkedTreeNode &p) {

        j = nlohmann::json{
            {"level", p.level},
            {"parent_nid", p.parent_nid},
            {"childs_nid", p.childs_nid},
            {"is_leaf", p.is_leaf},
            {"child_are_all_leafs", p.child_are_all_leafs}};
    }

    void from_json(const nlohmann::json &j, LinkedTreeNode &p) {
        j.at("level").get_to(p.level);
        j.at("parent_nid").get_to(p.parent_nid);
        j.at("childs_nid").get_to(p.childs_nid);
        j.at("is_leaf").get_to(p.is_leaf);
        j.at("child_are_all_leafs").get_to(p.child_are_all_leafs);
    }

    void to_json(nlohmann::json &j, const PatchTreeNode &p) {

        j = nlohmann::json{
            {"linked_patchid", p.linked_patchid},
            {"load_value", p.load_value},
            {"tree_node", p.tree_node},
            {"patch_coord",
             {
                 {"min", p.patch_coord.coord_min},
                 {"max", p.patch_coord.coord_max},
             }},
        };
    }

    void from_json(const nlohmann::json &j, PatchTreeNode &p) {
        j.at("linked_patchid").get_to(p.linked_patchid);
        j.at("load_value").get_to(p.load_value);
        j.at("tree_node").get_to(p.tree_node);
        j.at("patch_coord").at("min").get_to(p.patch_coord.coord_min);
        j.at("patch_coord").at("max").get_to(p.patch_coord.coord_max);
    }

} // namespace shamrock::scheduler
