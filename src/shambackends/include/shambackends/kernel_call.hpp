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
 * @file kernel_call.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include <optional>

namespace sham {

    namespace details {

        /**
         * @brief Get a pointer to the data of an optional device buffer, for reading.
         * @details If the optional is empty, a null pointer is returned. Otherwise, the read
         * access of the buffer is requested and the depends_list is updated accordingly.
         *
         * @param buffer An optional holding a reference to the device buffer.
         * @param depends_list The list of events to wait for.
         * @return A pointer to the data of the buffer, or nullptr if the optional is empty.
         */
        template<class T>
        const T *read_access_optional(
            std::optional<std::reference_wrapper<sham::DeviceBuffer<T>>> buffer,
            sham::EventList &depends_list) {
            if (!buffer.has_value()) {
                return nullptr;
            } else {
                return buffer.value().get().get_read_access(depends_list);
            }
        }

        /**
         * @brief Get a pointer to the data of an optional device buffer, for writing.
         * @details If the optional is empty, a null pointer is returned. Otherwise, the write
         * access of the buffer is requested and the depends_list is updated accordingly.
         *
         * @param buffer An optional holding a reference to the device buffer.
         * @param depends_list The list of events to wait for.
         * @return A pointer to the data of the buffer, or nullptr if the optional is empty.
         */
        template<class T>
        T *write_access_optional(
            std::optional<std::reference_wrapper<sham::DeviceBuffer<T>>> buffer,
            sham::EventList &depends_list) {
            if (!buffer.has_value()) {
                return nullptr;
            } else {
                return buffer.value().get().get_write_access(depends_list);
            }
        }

        /**
         * @brief Complete the event state of an optional device buffer.
         * @details If the optional is empty, nothing is done. Otherwise, the event state of the
         * buffer is completed with the given event.
         */
        template<class T>
        void
        complete_state_optional(sycl::event e, std::optional<std::reference_wrapper<T>> buffer) {
            if (buffer.has_value()) {
                buffer.value().get().complete_event_state(e);
            }
        }

    } // namespace details

    /**
     * @brief Converts a reference to a given object into an optional reference wrapper.
     * @tparam T Type of the object to reference.
     * @param t Reference to the object.
     * @return An std::optional containing a std::reference_wrapper of the object.
     */
    template<class T>
    std::optional<std::reference_wrapper<T>> to_opt_ref(T &t) {
        return t;
    }

    /**
     * @brief Returns an empty optional containing a reference to a sham::DeviceBuffer<T>.
     * @details This function is useful when you want to pass an optional reference to a kernel
     * argument but you don't know if the argument is going to be used or not.
     * @return An empty std::optional containing a std::reference_wrapper of a
     * sham::DeviceBuffer<T>.
     */
    template<class T>
    auto empty_buf_ref() {
        return std::optional<std::reference_wrapper<sham::DeviceBuffer<T>>>{};
    }

    /**
     * @brief A variant of MultiRef for optional buffers.
     *
     * This class is equivalent to MultiRef but it allows optional buffers. Only DeviceBuffer are
     * supported as optional buffers.
     *
     * @see MultiRef
     */
    template<class... Targ>
    struct MultiRefOpt {
        /// A tuple of optional references to the buffers.
        using storage_t = std::tuple<std::optional<std::reference_wrapper<Targ>>...>;

        /// The tuple of optional references to the buffers.
        storage_t storage;

        /// Constructor from a tuple of optional references to the buffers.
        MultiRefOpt(std::optional<std::reference_wrapper<Targ>>... arg) : storage(arg...) {}

        /**
         * @brief Get a tuple of pointers to the data of the buffers, for reading.
         * @details If a buffer is empty, a null pointer is returned. Otherwise, the read
         * access of the buffer is requested and the depends_list is updated accordingly.
         *
         * @param depends_list The list of events to wait for.
         * @return A tuple of pointers to the data of the buffers, or nullptr if the buffer is
         * empty.
         */
        auto get_read_access(sham::EventList &depends_list) {
            StackEntry stack_loc{};
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(details::read_access_optional(__a, depends_list)...);
                },
                storage);
        }
        /**
         * @brief Get a tuple of pointers to the data of the buffers, for writing.
         * @details If a buffer is empty, a null pointer is returned. Otherwise, the write
         * access of the buffer is requested and the depends_list is updated accordingly.
         *
         * @param depends_list The list of events to wait for.
         * @return A tuple of pointers to the data of the buffers, or nullptr if the buffer is
         * empty.
         */
        auto get_write_access(sham::EventList &depends_list) {
            StackEntry stack_loc{};
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(details::write_access_optional(__a, depends_list)...);
                },
                storage);
        }

        /**
         * @brief Complete the event state of the buffers.
         * @details This function completes the event state of all the buffers in the
         * MultiRefOpt by registering the event `e` in all the buffers.
         *
         * @param e The SYCL event to register in the buffers.
         */
        void complete_event_state(sycl::event e) {
            StackEntry stack_loc{};
            std::apply(
                [&](auto &...__in) {
                    ((details::complete_state_optional(e, __in)), ...);
                },
                storage);
        }
    };

    namespace details {
        /// internal_utility for MultiRef template deduction guide
        template<class T>
        struct mapper {
            /// The mapped type.
            using type = T;
        };

        /// internal_utility for MultiRef template deduction guide
        template<class T>
        struct mapper<std::optional<std::reference_wrapper<T>>> {
            /// The mapped type.
            using type = T;
        };
    } // namespace details

    /// deduction guide to allow the MutliRefOpt to be build without the use of sham::to_opt_ref
    template<class... Targ>
    MultiRefOpt(Targ... arg) -> MultiRefOpt<typename details::mapper<Targ>::type...>;

    /**
     * @brief A class that references multiple buffers or similar objects.
     *
     * This class serves as a means to pass multiple buffers or objects with similar accessor
     * patterns to a kernel. It provides methods to obtain read and write access to these
     * entities and to complete their event state.
     *
     * A version of this class is also available for optional references to the buffers or similar
     * objects, @see MultiRefOpt.
     */
    template<class... Targ>
    struct MultiRef {
        /// A tuple of references to the buffers.
        using storage_t = std::tuple<Targ &...>;

        /// A tuple of references to the buffers.
        storage_t storage;

        /// Constructor
        MultiRef(Targ &...arg) : storage(arg...) {}

        /// Get a tuple of pointers to the data of the buffers, for reading. Register also the
        /// depedancies in depends_list.
        auto get_read_access(sham::EventList &depends_list) {
            StackEntry stack_loc{};
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(__a.get_read_access(depends_list)...);
                },
                storage);
        }

        /// Get a tuple of pointers to the data of the buffers, for writing. Register also the
        /// depedancies in depends_list.
        auto get_write_access(sham::EventList &depends_list) {
            StackEntry stack_loc{};
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(__a.get_write_access(depends_list)...);
                },
                storage);
        }

        /// Complete the event state of the buffers.
        /// @param e The SYCL event to register in the buffers.
        void complete_event_state(sycl::event e) {
            StackEntry stack_loc{};
            std::apply(
                [&](auto &...__in) {
                    ((__in.complete_event_state(e)), ...);
                },
                storage);
        }
    };

    /**
     * @brief Submit a kernel to a SYCL queue.
     *
     * @todo Add code examples from the PR
     *
     * @param q The SYCL queue to submit the kernel to.
     * @param in The input buffer or MultiRef or MultiRefOpt.
     * @param in_out The input/output buffer or MultiRef or MultiRefOpt.
     * @param n The number of thread to launch.
     * @param func The functor to call for each thread launched.
     * @param args Additional arguments to pass to the functor.
     */
    template<class RefIn, class RefOut, class... Targs, class Functor>
    void kernel_call(
        sham::DeviceQueue &q, RefIn in, RefOut in_out, u32 n, Functor &&func, Targs... args) {
        StackEntry stack_loc{};
        sham::EventList depends_list;

        auto acc_in     = in.get_read_access(depends_list);
        auto acc_in_out = in_out.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>{n}, [=](sycl::item<1> item) {
                std::apply(
                    [&](auto &...__acc_in) {
                        std::apply(
                            [&](auto &...__acc_in_out) {
                                func(item.get_linear_id(), __acc_in..., __acc_in_out..., args...);
                            },
                            acc_in_out);
                    },
                    acc_in);
            });
        });

        in.complete_event_state(e);
        in_out.complete_event_state(e);
    }

} // namespace sham
