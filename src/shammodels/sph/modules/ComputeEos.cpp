// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeEos.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "ComputeEos.hpp"
#include "shambase/exception.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamphys/eos.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/legacy/log.hpp"

template<class T>
const T *read_access(std::optional<std::reference_wrapper<sham::DeviceBuffer<T>>>buffer, sham::EventList &depends_list) {
    if (!buffer.has_value()) {
        return nullptr;
    }else {
        return buffer.value().get().get_read_access(depends_list);
    }
    
}

template<class T>
T *write_access(std::optional<std::reference_wrapper<sham::DeviceBuffer<T>>>buffer, sham::EventList &depends_list) {
    if (!buffer.has_value()) {
        return nullptr;
    }else {
        return buffer.value().get().get_write_access(depends_list);
}
}

template<class T>
void complete_state(sycl::event e, std::optional<std::reference_wrapper<sham::DeviceBuffer<T>>>buffer) {
     if (buffer.has_value()) {
                 buffer.value().get().complete_event_state(e);
    }
}

template <class T>
std::optional<std::reference_wrapper<T>> to_opt_ref(T& t) { return t; }

template <class T>
auto empty_buf_ref(){
    return std::optional<std::reference_wrapper<sham::DeviceBuffer<T>>>{};
}


template<class... Targ>
struct MultiRef {
    using storage_t = std::tuple<std::optional<std::reference_wrapper<Targ>>...>;

    storage_t storage;

    MultiRef(std::optional<std::reference_wrapper<Targ >>...arg) : storage(arg...) {}

    auto get_read_access(sham::EventList &depends_list) {
        StackEntry stack_loc{};
        return std::apply(
            [&](auto &...__a) {
                return std::tuple(read_access(__a, depends_list)...);
            },
            storage);
    }
    auto get_write_access(sham::EventList &depends_list) {
        StackEntry stack_loc{};
        return std::apply(
            [&](auto &...__a) {
                return std::tuple(write_access(__a, depends_list)...);
            },
            storage);
    }

    void complete_event_state(sycl::event e) {
        StackEntry stack_loc{};
        std::apply(
            [&](auto &...__in) {
                ((complete_state(e, __in)), ...);
            },
            storage);
    }
};

template<class T>
struct mapper{
    using type = T;   
};

template<class T>
struct mapper<std::optional<std::reference_wrapper<T>>> {
    using type = T;
};

template <class... Targ>
MultiRef(Targ... arg) -> MultiRef<typename mapper<Targ>::type...>;





template<class... Targ1, class... Targ2>
void kernel_call(
    MultiRef<Targ1...> in,
    MultiRef<Targ2...> in_out,
    u32 n,
    std::function<void(const Targ1 *..., Targ2 *..., u32 i)> func) {

    sham::EventList depends_list;

    // get pointers associated to in.storage... .get_read_access(depends_list);
    auto acc_in = std::apply(
        [&](auto &...__a) {
            return std::tuple<const Targ1 *...>(__a.get_read_access(depends_list)...);
        },
        in.storage);

    // get pointers associated to in_out.storage... .get_write_access(depends_list);
    auto acc_in_out = std::apply(
        [&](auto &...__a) {
            return std::tuple<Targ2 *...>(__a.get_write_access(depends_list)...);
        },
        in_out.storage);

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{n}, [=](sycl::item<1> item) {
            std::apply(
                [&](auto &...__acc_in) {
                    std::apply(
                        [&](auto &...__acc_in_out) {
                            func(__acc_in..., __acc_in_out..., item.get_linear_id());
                        },
                        acc_in_out);
                },
                acc_in);
        });
    });

    std::apply(
        [&](auto &...__in) {
            ((__in.complete_event_state(e)), ...);
        },
        in.storage);
    std::apply(
        [&](auto &...__in_out) {
            ((__in_out.complete_event_state(e)), ...);
        },
        in_out.storage);
}

template<class Kernel, class... Targ1, class... Targ2, class... Targs>
void kernel_call_class(MultiRef<Targ1...> in, MultiRef<Targ2...> in_out, u32 n, Targs... args) {

    sham::EventList depends_list;

    // get pointers associated to in.storage... .get_read_access(depends_list);
    auto acc_in = std::apply(
        [&](auto &...__a) {
            return std::tuple<const Targ1 *...>(__a.get_read_access(depends_list)...);
        },
        in.storage);

    // get pointers associated to in_out.storage... .get_write_access(depends_list);
    auto acc_in_out = std::apply(
        [&](auto &...__a) {
            return std::tuple<Targ2 *...>(__a.get_write_access(depends_list)...);
        },
        in_out.storage);

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        auto kern = std::apply(
            [&](auto &...__acc_in) {
                return std::apply(
                    [&](auto &...__acc_in_out) {
                        return Kernel{__acc_in..., __acc_in_out..., args...};
                    },
                    acc_in_out);
            },
            acc_in);

        cgh.parallel_for<Kernel>(sycl::range<1>{n}, kern);
    });

    std::apply(
        [&](auto &...__in) {
            ((__in.complete_event_state(e)), ...);
        },
        in.storage);
    std::apply(
        [&](auto &...__in_out) {
            ((__in_out.complete_event_state(e)), ...);
        },
        in_out.storage);
}

template<class... Mappers, class... Refs, class Functor>
void kernel_call_mapper(Refs... args, u32 n, Functor &&func) {

    std::tuple<Refs...> buf_refs = std::make_tuple(args...);

    sham::EventList depends_list;

    auto acc = std::apply(
        [&](auto &...__a) {
            return std::tuple(Mappers{__a.get_access(depends_list)}...);
        },
        buf_refs);

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{n}, [=](sycl::item<1> item) {
            std::apply(
                [&](auto &...__acc) {
                    func(item.get_linear_id(), __acc...);
                },
                acc);
        });
    });

    std::apply(
        [&](auto &...__in) {
            ((__in.complete_event_state(e)), ...);
        },
        buf_refs);
}
template<class RefIn, class RefOut, class... Targs, class Functor>
void kernel_call2(RefIn in, RefOut in_out, u32 n, Functor &&func, Targs... args) {
    StackEntry stack_loc{};
    sham::EventList depends_list;

    auto acc_in     = in.get_read_access(depends_list);
    auto acc_in_out = in_out.get_write_access(depends_list);

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        auto ker_in = acc_in;
        auto ker_in_out = acc_in_out;

        std::tuple args_f = std::make_tuple(args...);

        cgh.parallel_for(sycl::range<1>{n}, [=](sycl::item<1> item) {
            /*
            std::apply(
                [&,ker_in_out](auto &...__acc_in) {
                    std::apply(
                        [&](auto &...__acc_in_out) {
                            std::apply(
                                [&](auto &...__args) {
                                                            func(item.get_linear_id(), __acc_in..., __acc_in_out..., __args...);

                            },
                                args_f);
                        },
                        ker_in_out);
                },
                ker_in);
                */
        });
    });

    in.complete_event_state(e);
    in_out.complete_event_state(e);
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ComputeEos<Tvec, SPHKernel>::compute_eos() {

    NamedStackEntry stack_loc{"compute eos"};

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");

    shamrock::SchedulerUtility utility(scheduler());

    storage.pressure.set(utility.make_compute_field<Tscal>("pressure", 1, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    storage.soundspeed.set(utility.make_compute_field<Tscal>("soundspeed", 1, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    using SolverConfigEOS                   = typename Config::EOSConfig;
    using SolverEOS_Isothermal              = typename SolverConfigEOS::Isothermal;
    using SolverEOS_Adiabatic               = typename SolverConfigEOS::Adiabatic;
    using SolverEOS_LocallyIsothermal       = typename SolverConfigEOS::LocallyIsothermal;
    using SolverEOS_LocallyIsothermalLP07   = typename SolverConfigEOS::LocallyIsothermalLP07;
    using SolverEOS_LocallyIsothermalFA2014 = typename SolverConfigEOS::LocallyIsothermalFA2014;

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    if (SolverEOS_Isothermal *eos_config
        = std::get_if<SolverEOS_Isothermal>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_Isothermal<Tscal>;

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPatchData &mpdat) {
            sham::DeviceBuffer<Tscal> &buf_P  = storage.pressure.get().get_buf_check(id);
            sham::DeviceBuffer<Tscal> &buf_cs = storage.soundspeed.get().get_buf_check(id);
            sham::DeviceBuffer<Tscal> &buf_h  = mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf);

            sham::EventList depends_list;

            auto P  = buf_P.get_write_access(depends_list);
            auto cs = buf_cs.get_write_access(depends_list);
            auto h  = buf_h.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                Tscal pmass  = gpart_mass;
                Tscal cs_cfg = eos_config->cs;

                cgh.parallel_for(sycl::range<1>{mpdat.total_elements}, [=](sycl::item<1> item) {
                    using namespace shamrock::sph;

                    Tscal rho_a = rho_h(pmass, h[item], Kernel::hfactd);
                    Tscal P_a   = EOS::pressure(cs_cfg, rho_a);
                    P[item]     = P_a;
                    cs[item]    = cs_cfg;
                });
            });

            buf_P.complete_event_state(e);
            buf_h.complete_event_state(e);
            buf_cs.complete_event_state(e);
        });
    } else if (
        SolverEOS_Adiabatic *eos_config
        = std::get_if<SolverEOS_Adiabatic>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_Adiabatic<Tscal>;

        constexpr bool is_monofluid = false;

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPatchData &mpdat) {
            sham::DeviceBuffer<Tscal> &buf_P  = storage.pressure.get().get_buf_check(id);
            sham::DeviceBuffer<Tscal> &buf_cs = storage.soundspeed.get().get_buf_check(id);

            sham::DeviceBuffer<Tscal> &buf_h = mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf);
            sham::DeviceBuffer<Tscal> &buf_uint = mpdat.pdat.get_field_buf_ref<Tscal>(iuint_interf);



            auto get_eps = [&]()  {
                if constexpr (is_monofluid) {
                    sham::DeviceBuffer<Tscal> &buf_epsilon
                        = mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf);
                    return to_opt_ref(buf_epsilon);
                } else {
                    return empty_buf_ref<Tscal>();
                }
            };

            kernel_call2(
                MultiRef{buf_h, buf_uint, get_eps()},
                MultiRef{buf_P, buf_cs},
                mpdat.total_elements,
                [](
                    u32 i,
                    const Tscal * h,
                    const Tscal * U,
                     const Tscal * epsilon /* set to nullptr if not is_monofluid */,
                    Tscal * P,
                    Tscal * cs,Tscal pmass , Tscal gamma ) {
                    auto rho = [&]() {
                        using namespace shamrock::sph;
                        if constexpr (is_monofluid) {
                            return (1 - epsilon[i]) * rho_h(pmass, h[i], Kernel::hfactd);
                        } else {
                            return rho_h(pmass, h[i], Kernel::hfactd);
                        }
                    };

                    Tscal rho_a = rho();
                    Tscal P_a   = EOS::pressure(gamma, rho_a, U[i]);
                    Tscal cs_a  = EOS::cs_from_p(gamma, rho_a, P_a);
                    P[i]        = P_a;
                    cs[i]       = cs_a;
                }, gpart_mass, eos_config->gamma);
        });

    } else if (
        SolverEOS_LocallyIsothermal *eos_config
        = std::get_if<SolverEOS_LocallyIsothermal>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_LocallyIsothermal<Tscal>;

        u32 isoundspeed_interf = ghost_layout.get_field_idx<Tscal>("soundspeed");

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPatchData &mpdat) {
            sham::DeviceBuffer<Tscal> &buf_P    = storage.pressure.get().get_buf_check(id);
            sham::DeviceBuffer<Tscal> &buf_cs   = storage.soundspeed.get().get_buf_check(id);
            sham::DeviceBuffer<Tscal> &buf_uint = mpdat.pdat.get_field_buf_ref<Tscal>(iuint_interf);
            sham::DeviceBuffer<Tscal> &buf_h = mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf);
            sham::DeviceBuffer<Tscal> &buf_cs0
                = mpdat.pdat.get_field_buf_ref<Tscal>(isoundspeed_interf);

            sham::EventList depends_list;

            auto P   = buf_P.get_write_access(depends_list);
            auto cs  = buf_cs.get_write_access(depends_list);
            auto h   = buf_h.get_read_access(depends_list);
            auto U   = buf_uint.get_read_access(depends_list);
            auto cs0 = buf_cs0.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                Tscal pmass = gpart_mass;

                cgh.parallel_for(sycl::range<1>{mpdat.total_elements}, [=](sycl::item<1> item) {
                    using namespace shamrock::sph;

                    Tscal cs_out = cs0[item];
                    Tscal rho_a  = rho_h(pmass, h[item], Kernel::hfactd);

                    Tscal P_a = EOS::pressure_from_cs(cs_out * cs_out, rho_a);

                    P[item]  = P_a;
                    cs[item] = cs_out;
                });
            });

            buf_P.complete_event_state(e);
            buf_cs.complete_event_state(e);
            buf_h.complete_event_state(e);
            buf_uint.complete_event_state(e);
            buf_cs0.complete_event_state(e);
        });

    } else if (
        SolverEOS_LocallyIsothermalLP07 *eos_config
        = std::get_if<SolverEOS_LocallyIsothermalLP07>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_LocallyIsothermal<Tscal>;

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPatchData &mpdat) {
            auto &mfield = storage.merged_xyzh.get().get(id);

            sham::DeviceBuffer<Tvec> &buf_xyz = mfield.field_pos.get_buf();

            sham::DeviceBuffer<Tscal> &buf_P    = storage.pressure.get().get_buf_check(id);
            sham::DeviceBuffer<Tscal> &buf_cs   = storage.soundspeed.get().get_buf_check(id);
            sham::DeviceBuffer<Tscal> &buf_uint = mpdat.pdat.get_field_buf_ref<Tscal>(iuint_interf);
            sham::DeviceBuffer<Tscal> &buf_h = mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf);

            sham::EventList depends_list;

            auto P   = buf_P.get_write_access(depends_list);
            auto cs  = buf_cs.get_write_access(depends_list);
            auto h   = buf_h.get_read_access(depends_list);
            auto U   = buf_uint.get_read_access(depends_list);
            auto xyz = buf_xyz.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                Tscal cs0   = eos_config->cs0;
                Tscal mq    = -eos_config->q;
                Tscal r0sq  = eos_config->r0 * eos_config->r0;
                Tscal pmass = gpart_mass;

                cgh.parallel_for(sycl::range<1>{mpdat.total_elements}, [=](sycl::item<1> item) {
                    using namespace shamrock::sph;

                    Tvec R = xyz[item];

                    Tscal Rsq    = sycl::dot(R, R);
                    Tscal cs_sq  = EOS::soundspeed_sq(cs0 * cs0, Rsq / r0sq, mq);
                    Tscal cs_out = sycl::sqrt(cs_sq);
                    Tscal rho_a  = rho_h(pmass, h[item], Kernel::hfactd);

                    Tscal P_a = EOS::pressure_from_cs(cs_out * cs_out, rho_a);

                    P[item]  = P_a;
                    cs[item] = cs_out;
                });
            });

            buf_P.complete_event_state(e);
            buf_cs.complete_event_state(e);
            buf_h.complete_event_state(e);
            buf_uint.complete_event_state(e);
            buf_xyz.complete_event_state(e);
        });

    } else if (
        SolverEOS_LocallyIsothermalFA2014 *eos_config
        = std::get_if<SolverEOS_LocallyIsothermalFA2014>(&solver_config.eos_config.config)) {

        Tscal _G = solver_config.get_constant_G();

        using EOS = shamphys::EOS_LocallyIsothermal<Tscal>;

        auto &sink_parts = storage.sinks.get();
        std::vector<Tvec> sink_pos;
        std::vector<Tscal> sink_mass;
        u32 sink_cnt = 0;

        for (auto &s : sink_parts) {
            sink_pos.push_back(s.pos);
            sink_mass.push_back(s.mass);
            sink_cnt++;
        }

        sycl::buffer<Tvec> sink_pos_buf{sink_pos};
        sycl::buffer<Tscal> sink_mass_buf{sink_mass};

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPatchData &mpdat) {
            auto &mfield = storage.merged_xyzh.get().get(id);

            sham::DeviceBuffer<Tvec> &buf_xyz = mfield.field_pos.get_buf();

            sham::DeviceBuffer<Tscal> &buf_P    = storage.pressure.get().get_buf_check(id);
            sham::DeviceBuffer<Tscal> &buf_cs   = storage.soundspeed.get().get_buf_check(id);
            sham::DeviceBuffer<Tscal> &buf_uint = mpdat.pdat.get_field_buf_ref<Tscal>(iuint_interf);
            sham::DeviceBuffer<Tscal> &buf_h = mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf);

            sham::EventList depends_list;

            auto P   = buf_P.get_write_access(depends_list);
            auto cs  = buf_cs.get_write_access(depends_list);
            auto h   = buf_h.get_read_access(depends_list);
            auto U   = buf_uint.get_read_access(depends_list);
            auto xyz = buf_xyz.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                sycl::accessor spos{sink_pos_buf, cgh, sycl::read_only};
                sycl::accessor smass{sink_mass_buf, cgh, sycl::read_only};
                u32 scount = sink_cnt;

                Tscal pmass    = gpart_mass;
                Tscal h_over_r = eos_config->h_over_r;
                Tscal G        = _G;

                cgh.parallel_for(sycl::range<1>{mpdat.total_elements}, [=](sycl::item<1> item) {
                    using namespace shamrock::sph;

                    Tvec R    = xyz[item];
                    Tscal h_a = h[item];

                    Tscal mpotential = 0;
                    for (u32 i = 0; i < scount; i++) {
                        Tvec s_r      = spos[i] - R;
                        Tscal s_m     = smass[i];
                        Tscal s_r_abs = sycl::length(s_r);
                        mpotential += G * s_m / s_r_abs;
                    }

                    Tscal rho_a = rho_h(pmass, h_a, Kernel::hfactd);

                    Tscal cs_out = h_over_r * sycl::sqrt(mpotential);
                    Tscal P_a    = EOS::pressure_from_cs(cs_out * cs_out, rho_a);

                    P[item]  = P_a;
                    cs[item] = cs_out;
                });
            });

            buf_P.complete_event_state(e);
            buf_cs.complete_event_state(e);
            buf_h.complete_event_state(e);
            buf_uint.complete_event_state(e);
            buf_xyz.complete_event_state(e);
        });

    } else {
        shambase::throw_unimplemented();
    }
}

using namespace shammath;
template class shammodels::sph::modules::ComputeEos<f64_3, M4>;
template class shammodels::sph::modules::ComputeEos<f64_3, M6>;
template class shammodels::sph::modules::ComputeEos<f64_3, M8>;
