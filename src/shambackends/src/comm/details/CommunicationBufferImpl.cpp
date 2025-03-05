// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CommunicationBufferImpl.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 * \todo this file should pull queues from backends and not sys lib
 */

#include "shambackends/comm/details/CommunicationBufferImpl.hpp"
#include "shambase/exception.hpp"
#include "shambackends/USMBufferInterop.hpp"
#include <cstring>
#include <stdexcept>

namespace shamcomm::details {} // namespace shamcomm::details
