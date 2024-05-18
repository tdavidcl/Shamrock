// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file DeviceContext.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/Device.hpp"

namespace sham {

    /**
     * @brief A class that represents a SYCL context
     *
     * This class is responsible for creating and holding the SYCL context
     * object, as well as providing methods for accessing it.
     */
    class DeviceContext {
    public:
        /**
         * The device(s) associated with this context
         */
        std::shared_ptr<Device> device;

        /**
         * The SYCL context object
         */
        sycl::context ctx;

        /**
         * @brief Print information about this context
         */
        void print_info();

        /**
         * @brief Construct a new Device Context object
         *
         * @param device The device(s) to use for this context
         */
        explicit DeviceContext(std::shared_ptr<Device> device);
    };


} // namespace sham