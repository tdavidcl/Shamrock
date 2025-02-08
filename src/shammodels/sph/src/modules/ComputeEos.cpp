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

#include "shambase/exception.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/ComputeEos.hpp"
#include "shamphys/eos.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tscal>
struct RhoGetterBase {
    sham::DeviceBuffer<Tscal> &buf_h;
    Tscal pmass;
    Tscal hfact;

    struct accessed {
        const Tscal *h;
        Tscal pmass;
        Tscal hfact;

        Tscal operator()(u32 i) const {
            using namespace shamrock::sph;
            return rho_h(pmass, h[i], hfact);
        }
    };

    accessed get_read_access(sham::EventList &depends_list) {
        auto h = buf_h.get_read_access(depends_list);
        return accessed{h, pmass, hfact};
    }

    void complete_event_state(sycl::event e) { buf_h.complete_event_state(e); }
};

template<class Tscal>
struct RhoGetterMonofluid {
    sham::DeviceBuffer<Tscal> &buf_h;
    sham::DeviceBuffer<Tscal> &buf_epsilon;
    u32 nvar_dust;
    Tscal pmass;
    Tscal hfact;

    struct accessed {
        const Tscal *h;
        const Tscal *buf_epsilon;
        u32 nvar_dust;
        Tscal pmass;
        Tscal hfact;

        Tscal operator()(u32 i) const {

            Tscal epsilon_sum = 0;
            for (u32 j = 0; j < nvar_dust; j++) {
                epsilon_sum += buf_epsilon[i * nvar_dust + j];
            }

            using namespace shamrock::sph;
            return (1 - epsilon_sum) * rho_h(pmass, h[i], hfact);
        }
    };

    accessed get_read_access(sham::EventList &depends_list) {
        auto h       = buf_h.get_read_access(depends_list);
        auto epsilon = buf_epsilon.get_read_access(depends_list);

        return accessed{h, epsilon, nvar_dust, pmass, hfact};
    }

    void complete_event_state(sycl::event e) { buf_h.complete_event_state(e); }
};

template<class Tvec, template<class> class SPHKernel>
template<class RhoGetGen>
void shammodels::sph::modules::ComputeEos<Tvec, SPHKernel>::compute_eos_internal(
    RhoGetGen &&rho_getter_gen) {

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");

    using namespace shamrock;

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
            auto rho_getter                   = rho_getter_gen(mpdat);

            sham::kernel_call(
                q,
                sham::MultiRef{rho_getter},
                sham::MultiRef{buf_P, buf_cs},
                mpdat.total_elements,
                [cs_cfg
                 = eos_config->cs](u32 i, auto rho, Tscal *__restrict P, Tscal *__restrict cs) {
                    using namespace shamrock::sph;
                    Tscal rho_a = rho(i);
                    Tscal P_a   = EOS::pressure(cs_cfg, rho_a);
                    P[i]        = P_a;
                    cs[i]       = cs_cfg;
                });
        });
    } else if (
        SolverEOS_Adiabatic *eos_config
        = std::get_if<SolverEOS_Adiabatic>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_Adiabatic<Tscal>;

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPatchData &mpdat) {
            sham::DeviceBuffer<Tscal> &buf_P    = storage.pressure.get().get_buf_check(id);
            sham::DeviceBuffer<Tscal> &buf_cs   = storage.soundspeed.get().get_buf_check(id);
            sham::DeviceBuffer<Tscal> &buf_uint = mpdat.pdat.get_field_buf_ref<Tscal>(iuint_interf);
            auto rho_getter                     = rho_getter_gen(mpdat);

            sham::kernel_call(
                q,
                sham::MultiRef{rho_getter, buf_uint},
                sham::MultiRef{buf_P, buf_cs},
                mpdat.total_elements,
                [gamma = eos_config->gamma](
                    u32 i,
                    auto rho,
                    const Tscal *__restrict U,
                    Tscal *__restrict P,
                    Tscal *__restrict cs) {
                    using namespace shamrock::sph;
                    Tscal rho_a = rho(i);
                    Tscal P_a   = EOS::pressure(gamma, rho_a, U[i]);
                    Tscal cs_a  = EOS::cs_from_p(gamma, rho_a, P_a);
                    P[i]        = P_a;
                    cs[i]       = cs_a;
                });
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
            auto rho_getter                     = rho_getter_gen(mpdat);
            sham::DeviceBuffer<Tscal> &buf_cs0
                = mpdat.pdat.get_field_buf_ref<Tscal>(isoundspeed_interf);

            sham::kernel_call(
                q,
                sham::MultiRef{rho_getter, buf_uint, buf_cs0},
                sham::MultiRef{buf_P, buf_cs},
                mpdat.total_elements,
                [](u32 i,
                   auto rho,
                   const Tscal *__restrict U,
                   const Tscal *__restrict cs0,
                   Tscal *__restrict P,
                   Tscal *__restrict cs) {
                    using namespace shamrock::sph;

                    Tscal cs_out = cs0[i];
                    Tscal rho_a  = rho(i);

                    Tscal P_a = EOS::pressure_from_cs(cs_out * cs_out, rho_a);

                    P[i]  = P_a;
                    cs[i] = cs_out;
                });
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
            auto rho_getter                     = rho_getter_gen(mpdat);

            Tscal cs0  = eos_config->cs0;
            Tscal r0sq = eos_config->r0 * eos_config->r0;
            Tscal mq   = eos_config->q;

            sham::kernel_call(
                q,
                sham::MultiRef{rho_getter, buf_uint, buf_xyz},
                sham::MultiRef{buf_P, buf_cs},
                mpdat.total_elements,
                [cs0, r0sq, mq](
                    u32 i,
                    auto rho,
                    const Tscal *__restrict U,
                    const Tvec *__restrict xyz,
                    Tscal *__restrict P,
                    Tscal *__restrict cs) {
                    using namespace shamrock::sph;

                    Tvec R      = xyz[i];
                    Tscal rho_a = rho(i);

                    Tscal Rsq    = sycl::dot(R, R);
                    Tscal cs_sq  = EOS::soundspeed_sq(cs0 * cs0, Rsq / r0sq, mq);
                    Tscal cs_out = sycl::sqrt(cs_sq);

                    Tscal P_a = EOS::pressure_from_cs(cs_out * cs_out, rho_a);

                    P[i]  = P_a;
                    cs[i] = cs_out;
                });
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
            auto rho_getter                     = rho_getter_gen(mpdat);

            // TODO: Use the complex kernel call when implemented

            sham::EventList depends_list;

            auto P   = buf_P.get_write_access(depends_list);
            auto cs  = buf_cs.get_write_access(depends_list);
            auto rho = rho_getter.get_read_access(depends_list);
            auto U   = buf_uint.get_read_access(depends_list);
            auto xyz = buf_xyz.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                sycl::accessor spos{sink_pos_buf, cgh, sycl::read_only};
                sycl::accessor smass{sink_mass_buf, cgh, sycl::read_only};
                u32 scount = sink_cnt;

                Tscal h_over_r = eos_config->h_over_r;
                Tscal G        = _G;

                cgh.parallel_for(sycl::range<1>{mpdat.total_elements}, [=](sycl::item<1> item) {
                    using namespace shamrock::sph;

                    Tvec R      = xyz[item];
                    Tscal rho_a = rho(item.get_linear_id());

                    Tscal mpotential = 0;
                    for (u32 i = 0; i < scount; i++) {
                        Tvec s_r      = spos[i] - R;
                        Tscal s_m     = smass[i];
                        Tscal s_r_abs = sycl::length(s_r);
                        mpotential += G * s_m / s_r_abs;
                    }

                    Tscal cs_out = h_over_r * sycl::sqrt(mpotential);
                    Tscal P_a    = EOS::pressure_from_cs(cs_out * cs_out, rho_a);

                    P[item]  = P_a;
                    cs[item] = cs_out;
                });
            });

            buf_P.complete_event_state(e);
            buf_cs.complete_event_state(e);
            rho_getter.complete_event_state(e);
            buf_uint.complete_event_state(e);
            buf_xyz.complete_event_state(e);
        });

    } else {
        shambase::throw_unimplemented();
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ComputeEos<Tvec, SPHKernel>::compute_eos() {

    NamedStackEntry stack_loc{"compute eos"};

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");

    shamrock::SchedulerUtility utility(scheduler());

    storage.pressure.set(utility.make_compute_field<Tscal>("pressure", 1, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    storage.soundspeed.set(utility.make_compute_field<Tscal>("soundspeed", 1, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    if (solver_config.dust_config.has_epsilon_field()) {

        u32 iepsilon_interf = ghost_layout.get_field_idx<Tscal>("epsilon");
        u32 nvar_dust       = solver_config.dust_config.get_dust_nvar();

        compute_eos_internal([&](MergedPatchData &mpdat) {
            return RhoGetterMonofluid<Tscal>{
                mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf),
                mpdat.pdat.get_field_buf_ref<Tscal>(iepsilon_interf),
                nvar_dust,
                gpart_mass,
                Kernel::hfactd};
        });
    } else {
        compute_eos_internal([&](MergedPatchData &mpdat) {
            return RhoGetterBase<Tscal>{
                mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf), gpart_mass, Kernel::hfactd};
        });
    }
}

using namespace shammath;
template class shammodels::sph::modules::ComputeEos<f64_3, M4>;
template class shammodels::sph::modules::ComputeEos<f64_3, M6>;
template class shammodels::sph::modules::ComputeEos<f64_3, M8>;
