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
 * @file compute_histogram.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/kernel_call.hpp"
#include <shambackends/sycl.hpp>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace shamalgs::primitives {

    enum class histo_impl { reference, naive_gpu, gpu_team_fetching, gpu_oversubscribe };
    inline histo_impl impl = histo_impl::reference;

    template<class T, class Tbins, class... Targs, class Tfunctor>
    inline sham::DeviceBuffer<T> compute_histogram(
        sham::DeviceScheduler_ptr dev_sched,
        const sham::DeviceBuffer<Tbins> &bin_edge_inf,
        const sham::DeviceBuffer<Tbins> &bin_edge_sup,
        size_t element_count,
        Tfunctor &&functor,
        const sham::DeviceBuffer<Targs> &...input_data) {

        size_t nbins = bin_edge_inf.get_size();

        if (nbins != bin_edge_sup.get_size()) {
            shambase::make_except_with_loc<std::invalid_argument>("TODO: message");
        }

        sham::DeviceBuffer<T> result(nbins, dev_sched);

        if (impl == histo_impl::reference) {

            auto result_vec = result.copy_to_stdvec();

            auto cpu_basic_impl = [&](const std::vector<Tbins> &bin_edge_inf,
                                      const std::vector<Tbins> &bin_edge_sup,
                                      const std::vector<Targs> &...in_data,
                                      std::vector<T> &result) {
                for (size_t ibin = 0; ibin < nbins; ibin++) {
                    Tbins edge_inf = bin_edge_inf[ibin];
                    Tbins edge_sup = bin_edge_sup[ibin];

                    T accumulator = 0;

                    for (size_t i = 0; i < element_count; i++) {
                        bool has_value = false;
                        auto tmp       = functor(edge_inf, edge_sup, in_data[i]..., has_value);
                        if (has_value) {
                            accumulator += tmp;
                        }
                    }

                    result[ibin] = accumulator;
                }
            };

            cpu_basic_impl(
                bin_edge_inf.copy_to_stdvec(),
                bin_edge_sup.copy_to_stdvec(),
                input_data.copy_to_stdvec()...,
                result_vec);

            result.copy_from_stdvec(result_vec);

        } else if (impl == histo_impl::naive_gpu) {

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{bin_edge_inf, bin_edge_sup, input_data...},
                sham::MultiRef{result},
                nbins,
                [element_count, functor](
                    u32 ibin,
                    const Tbins *__restrict bin_edge_inf,
                    const Tbins *__restrict bin_edge_sup,
                    const Targs *__restrict... in_data,
                    T *__restrict result) {
                    Tbins edge_inf = bin_edge_inf[ibin];
                    Tbins edge_sup = bin_edge_sup[ibin];

                    T accumulator = 0;

                    for (size_t i = 0; i < element_count; i++) {
                        bool has_value = false;
                        T tmp          = functor(edge_inf, edge_sup, in_data[i]..., has_value);
                        if (has_value) {
                            accumulator += tmp;
                        }
                    }

                    result[ibin] = accumulator;
                });

        } else if (impl == histo_impl::gpu_team_fetching) {

            sham::kernel_call_hndl(
                dev_sched->get_queue(),
                sham::MultiRef{bin_edge_inf, bin_edge_sup, input_data...},
                sham::MultiRef{result},
                nbins,
                [element_count, functor](
                    u32 nbins,
                    const Tbins *__restrict bin_edge_inf,
                    const Tbins *__restrict bin_edge_sup,
                    const Targs *__restrict... in_data,
                    T *__restrict result) {
                    return [=, in_data = std::tuple{in_data...}](sycl::handler &cgh) {
                        u32 group_size = 128;
                        u32 group_cnt  = shambase::group_count(nbins, group_size);

                        group_cnt         = group_cnt + (group_cnt % 4);
                        u32 corrected_len = group_cnt * group_size;

                        auto locals
                            = sycl::local_accessor<std::tuple<Targs...>, 1>(group_size, cgh);

                        cgh.parallel_for(
                            sycl::nd_range<1>{corrected_len, group_size},
                            [=](sycl::nd_item<1> item) {
                                u32 local_id      = item.get_local_id(0);
                                u32 group_tile_id = item.get_group_linear_id();
                                u32 ibin          = group_tile_id * group_size + local_id;

                                bool is_valid_point = (ibin < nbins);
                                Tbins edge_inf      = is_valid_point ? bin_edge_inf[ibin] : Tbins{};
                                Tbins edge_sup      = is_valid_point ? bin_edge_sup[ibin] : Tbins{};

                                T local_sum = 0;

                                for (size_t i = 0; i < element_count; i += group_size) {

                                    item.barrier(sycl::access::fence_space::local_space);

                                    if (i + local_id < element_count) {
                                        std::apply(
                                            [&](auto &...in_data) {
                                                locals[local_id]
                                                    = std::tuple{in_data[i + local_id]...};
                                            },
                                            in_data);
                                    }

                                    item.barrier(sycl::access::fence_space::local_space);

                                    if (is_valid_point) {
                                        for (size_t lane = 0; lane < group_size; lane++) {
                                            if (i + lane >= element_count) {
                                                continue;
                                            }
                                            bool has_value = false;
                                            T tmp          = std::apply(
                                                [&](auto &...local_accs) {
                                                    return functor(
                                                        edge_inf,
                                                        edge_sup,
                                                        local_accs...,
                                                        has_value);
                                                },
                                                locals[lane]);
                                            if (has_value) {
                                                local_sum += tmp;
                                            }
                                        }
                                    }

                                    item.barrier(sycl::access::fence_space::local_space);
                                }

                                if (is_valid_point) {
                                    result[ibin] = local_sum;
                                }
                            });
                    };
                });

        } else if (impl == histo_impl::gpu_oversubscribe) {

            u32 group_size = 256;
            sham::kernel_call_hndl(
                dev_sched->get_queue(),
                sham::MultiRef{bin_edge_inf, bin_edge_sup, input_data...},
                sham::MultiRef{result},
                nbins * group_size,
                [element_count, functor, group_size, nbins](
                    u32 nbins_oversubscribed,
                    const Tbins *__restrict bin_edge_inf,
                    const Tbins *__restrict bin_edge_sup,
                    const Targs *__restrict... in_data,
                    T *__restrict result) {
                    return [=, in_data = std::tuple{in_data...}](sycl::handler &cgh) {
                        u32 group_cnt = shambase::group_count(nbins_oversubscribed, group_size);

                        group_cnt         = group_cnt + (group_cnt % 4);
                        u32 corrected_len = group_cnt * group_size;

                        cgh.parallel_for(
                            sycl::nd_range<1>{corrected_len, group_size},
                            [=](sycl::nd_item<1> item) {
                                u32 local_id = item.get_local_id(0);
                                u32 ibin     = item.get_group_linear_id();

                                bool is_valid_point = (ibin < nbins);
                                Tbins edge_inf      = is_valid_point ? bin_edge_inf[ibin] : Tbins{};
                                Tbins edge_sup      = is_valid_point ? bin_edge_sup[ibin] : Tbins{};

                                // for each thread this will the sum of all the
                                // "func(in_data[group_size*i + local_data]) for all i"
                                T local_sum = 0;

                                for (size_t i = 0; i < element_count; i += group_size) {

                                    if (i + local_id < element_count) {

                                        bool has_value = false;

                                        // coalesced read of the data and then
                                        // compute the value to accumulate
                                        T tmp = std::apply(
                                            [&](auto &...in_data) {
                                                return functor(
                                                    edge_inf,
                                                    edge_sup,
                                                    in_data[i + local_id]...,
                                                    has_value);
                                            },
                                            in_data);

                                        if (has_value) {
                                            // add it to the local sum of this thread
                                            local_sum += tmp;
                                        }
                                    }
                                }

                                // we have all the terms scattered across the threads of the group,
                                // we can just accumulate the result
                                auto group_sum = sycl::reduce_over_group(
                                    item.get_group(), local_sum, sycl::plus<T>{});

                                if (is_valid_point && local_id == 0) {
                                    result[ibin] = group_sum;
                                }
                            });
                    };
                });
        } else {
            shambase::throw_unimplemented("unknown implementation");
        }

        return result;
    }

} // namespace shamalgs::primitives
