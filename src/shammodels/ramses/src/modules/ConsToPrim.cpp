// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ConsToPrim.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/riemann.hpp"
#include "shammath/riemann_dust.hpp"
#include "shammodels/ramses/modules/ConsToPrim.hpp"
#include "shammodels/ramses/modules/ConsToPrimDust.hpp"
#include "shammodels/ramses/modules/ConsToPrimGas.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/ComputeFieldEdge.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include <memory>
#include <utility>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::cons_to_prim_gas() {}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::cons_to_prim_dust() {}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::reset_gas() {}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::reset_dust() {}

template class shammodels::basegodunov::modules::ConsToPrim<f64_3, i64_3>;
