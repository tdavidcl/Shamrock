// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file CommunicationBuffer.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Shamrock communication buffers
 *
 * \todo make a better exemple
 *
 * \code{.cpp}
 *   u32 nbytes = 1e5;
 *   sycl::buffer<u8> buf_comp = shamalgs::random::mock_buffer<u8>(0x111, nbytes);
 *   shamcomm::CommunicationBuffer cbuf {buf_comp, shamcomm::CopyToHost};
 *   sycl::buffer<u8> ret = cbuf.copy_back();
 * \endcode
 */

#include "shambase/exception.hpp"
#include "shambackends/DeviceContext.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/comm/details/CommunicationBufferImpl.hpp"

namespace shamcomm {

    /**
     * @brief Shamrock communication buffers
     * \todo try reducing compile time by type erasing impl
     */
    class CommunicationBuffer {

        private:
        using int_var_t = std::variant<std::unique_ptr<details::CommunicationBuffer<CopyToHost>>,
                                       std::unique_ptr<details::CommunicationBuffer<DirectGPU>>>;

        int_var_t _int_type;

        explicit CommunicationBuffer(int_var_t &&moved_int_var)
            : _int_type(std::move(moved_int_var)) {}

        using Protocol = CommunicationProtocol;

        public:
        inline CommunicationBuffer(u64 bytelen, sham::DeviceScheduler &queue_details) {
            sham::Device & dev = *queue_details.ctx->device;
            Protocol comm_mode = get_protocol(dev);
            if (comm_mode == CopyToHost) {
                _int_type = std::make_unique<details::CommunicationBuffer<CopyToHost>>(bytelen, queue_details.get_queue().q);
            } else if (comm_mode == DirectGPU) {
                _int_type = std::make_unique<details::CommunicationBuffer<DirectGPU>>(bytelen, queue_details.get_queue().q);
            } else {
                throw shambase::make_except_with_loc<std::invalid_argument>("unknown mode");
            }
        }

        inline CommunicationBuffer(sycl::buffer<u8> &bytebuf, sham::DeviceScheduler &queue_details) {
            sham::Device & dev = *queue_details.ctx->device;
            Protocol comm_mode = get_protocol(dev);
            if (comm_mode == CopyToHost) {
                _int_type = std::make_unique<details::CommunicationBuffer<CopyToHost>>(bytebuf, queue_details.get_queue().q);
            } else if (comm_mode == DirectGPU) {
                _int_type = std::make_unique<details::CommunicationBuffer<DirectGPU>>(bytebuf, queue_details.get_queue().q);
            } else {
                throw shambase::make_except_with_loc<std::invalid_argument>("unknown mode");
            }
        }

        inline CommunicationBuffer(sycl::buffer<u8> &&bytebuf, sham::DeviceScheduler &queue_details) {
            sham::Device & dev = *queue_details.ctx->device;
            Protocol comm_mode = get_protocol(dev);
            if (comm_mode == CopyToHost) {
                _int_type = std::make_unique<details::CommunicationBuffer<CopyToHost>>(
                    std::forward<sycl::buffer<u8>>(bytebuf), queue_details.get_queue().q);
            } else if (comm_mode == DirectGPU) {
                _int_type = std::make_unique<details::CommunicationBuffer<DirectGPU>>(
                    std::forward<sycl::buffer<u8>>(bytebuf), queue_details.get_queue().q);
            } else {
                throw shambase::make_except_with_loc<std::invalid_argument>("unknown mode");
            }
        }

        /**
         * @brief return a copy of the held object in the buffer
         *
         * @return T
         */
        inline sycl::buffer<u8> copy_back() {
            return std::visit([=](auto &&arg) { return arg->copy_back(); }, _int_type);
        }

        inline u64 get_bytesize() {
            return std::visit([=](auto &&arg) { return arg->get_bytesize(); }, _int_type);
        }

        inline u8 *get_ptr() {
            return std::visit([=](auto &&arg) { return arg->get_ptr(); }, _int_type);
        }

        /**
         * @brief duplicate the comm buffer and return a unique ptr to the copy
         *
         * @return CommBuffer
         */
        inline CommunicationBuffer duplicate() {

            int_var_t tmp = std::visit(
                [=](auto &&arg) { return int_var_t(arg->duplicate_to_ptr()); }, _int_type);

            return CommunicationBuffer(std::move(tmp));
        }

        /**
         * @brief duplicate the comm buffer and return a unique ptr to the copy
         *
         * @return CommBuffer
         */
        inline std::unique_ptr<CommunicationBuffer> duplicate_to_ptr() {
            return std::make_unique<CommunicationBuffer>(duplicate());
        }
        /**
         * @brief destroy the buffer and recover the held object
         *
         * @param buf
         * @return T
         */
        inline static sycl::buffer<u8> convert(CommunicationBuffer &&buf) {
            return std::visit(
                [=](auto &&arg) {
                    using _t = typename std::remove_reference<decltype(*arg)>::type;
                    return _t::convert(std::forward<_t>(*arg));
                },
                buf._int_type);
        }
    };

    void validate_comm(sham::DeviceScheduler & sched);

} // namespace shamcomm