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

    template<class... Targ>
    struct MultiRefOpt {
        using storage_t = std::tuple<std::optional<std::reference_wrapper<Targ>>...>;

        storage_t storage;

        MultiRefOpt(std::optional<std::reference_wrapper<Targ>>... arg) : storage(arg...) {}

        auto get_read_access(sham::EventList &depends_list) {
            StackEntry stack_loc{};
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(details::read_access_optional(__a, depends_list)...);
                },
                storage);
        }
        auto get_write_access(sham::EventList &depends_list) {
            StackEntry stack_loc{};
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(details::write_access_optional(__a, depends_list)...);
                },
                storage);
        }

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
        template<class T>
        struct mapper {
            using type = T;
        };

        template<class T>
        struct mapper<std::optional<std::reference_wrapper<T>>> {
            using type = T;
        };
    } // namespace details

    template<class... Targ>
    MultiRefOpt(Targ... arg) -> MultiRefOpt<typename details::mapper<Targ>::type...>;

    template<class... Targ>
    struct MultiRef {
        using storage_t = std::tuple<Targ &...>;

        storage_t storage;

        MultiRef(Targ &...arg) : storage(arg...) {}

        auto get_read_access(sham::EventList &depends_list) {
            StackEntry stack_loc{};
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(__a.get_read_access(depends_list)...);
                },
                storage);
        }
        auto get_write_access(sham::EventList &depends_list) {
            StackEntry stack_loc{};
            return std::apply(
                [&](auto &...__a) {
                    return std::tuple(__a.get_write_access(depends_list)...);
                },
                storage);
        }

        void complete_event_state(sycl::event e) {
            StackEntry stack_loc{};
            std::apply(
                [&](auto &...__in) {
                    ((__in.complete_event_state(e)), ...);
                },
                storage);
        }
    };

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
