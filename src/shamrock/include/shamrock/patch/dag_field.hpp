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
 * @file dag_field.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shambase/DistributedData.hpp"
#include "shamdag/IDataEdge.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include <functional>
#include <memory>

namespace shamrock::dag {

    template<class T>
    class Field : public IDataEdge {

        std::string name;
        std::string texsymbol;

        public:
        bool is_data_owning                                                             = false;
        shambase::DistributedData<PatchDataField<T>> data_ownership                     = {};
        shambase::DistributedData<std::reference_wrapper<PatchDataField<T>>> field_refs = {};

        Field(std::string name, std::string texsymbol) : name(name), texsymbol(texsymbol) {}

        inline std::string _impl_get_label() { return name; };
        inline std::string _impl_get_tex_symbol() { return texsymbol; };

        inline shambase::DistributedData<PatchDataField<T>> steal_data() {
            report_data_stealing();
            field_refs.reset();
            is_data_owning = false;
            return std::move(data_ownership);
        }
    };

} // namespace shamrock::dag
