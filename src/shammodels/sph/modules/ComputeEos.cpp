// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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

    using SolverConfigEOS     = typename Config::EOSConfig;
    using SolverEOS_Adiabatic = typename SolverConfigEOS::Adiabatic;
    using SolverEOS_LocallyIsothermal = typename SolverConfigEOS::LocallyIsothermal;
    using SolverEOS_LocallyIsothermalLP07 = typename SolverConfigEOS::LocallyIsothermalLP07;

    if (SolverEOS_Adiabatic *eos_config = std::get_if<SolverEOS_Adiabatic>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_Adiabatic<Tscal>;

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPatchData &mpdat) {

            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                sycl::accessor P{storage.pressure.get().get_buf_check(id), cgh, sycl::write_only, sycl::no_init};
                sycl::accessor cs{storage.soundspeed.get().get_buf_check(id), cgh, sycl::write_only, sycl::no_init};
                sycl::accessor U{mpdat.pdat.get_field_buf_ref<Tscal>(iuint_interf) , cgh, sycl::read_only};
                sycl::accessor h{mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf), cgh, sycl::read_only};

                Tscal pmass = gpart_mass;
                Tscal gamma = eos_config->gamma;

                cgh.parallel_for(sycl::range<1>{mpdat.total_elements}, [=](sycl::item<1> item) {
                    using namespace shamrock::sph;

                    Tscal rho_a = rho_h(pmass, h[item], Kernel::hfactd);
                    Tscal P_a   = EOS::pressure(gamma ,  rho_a , U[item]);
                    Tscal cs_a  = EOS::cs_from_p(gamma, rho_a, P_a);
                    P[item]     = P_a;
                    cs[item]    = cs_a;
                });
            });

        });

    }else if (SolverEOS_LocallyIsothermal *eos_config = std::get_if<SolverEOS_LocallyIsothermal>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_LocallyIsothermal<Tscal>;
        
        u32 isoundspeed_interf                               = ghost_layout.get_field_idx<Tscal>("soundspeed");

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPatchData &mpdat) {
            
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                sycl::accessor P{storage.pressure.get().get_buf_check(id), cgh, sycl::write_only, sycl::no_init};
                sycl::accessor cs{storage.soundspeed.get().get_buf_check(id), cgh, sycl::write_only, sycl::no_init};
                sycl::accessor U{mpdat.pdat.get_field_buf_ref<Tscal>(iuint_interf) , cgh, sycl::read_only};
                sycl::accessor h{mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf), cgh, sycl::read_only};
                sycl::accessor cs0{mpdat.pdat.get_field_buf_ref<Tscal>(isoundspeed_interf), cgh, sycl::read_only};

                Tscal pmass = gpart_mass;

                cgh.parallel_for(sycl::range<1>{mpdat.total_elements}, [=](sycl::item<1> item) {
                    using namespace shamrock::sph;

                    Tscal cs_out = cs0[item];
                    Tscal rho_a = rho_h(pmass, h[item], Kernel::hfactd);

                    Tscal P_a   = EOS::pressure_from_cs(cs_out*cs_out , rho_a) ;
                    
                    P[item]     = P_a;
                    cs[item]    = cs_out;
                });
            });
            

        });

    }else if (SolverEOS_LocallyIsothermalLP07 *eos_config = std::get_if<SolverEOS_LocallyIsothermalLP07>(&solver_config.eos_config.config)) {

        using EOS = shamphys::EOS_LocallyIsothermal<Tscal>;

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPatchData &mpdat) {
            
            auto &mfield = storage.merged_xyzh.get().get(id);

            sycl::buffer<Tvec> &buf_xyz    = shambase::get_check_ref(mfield.field_pos.get_buf());

            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

                sycl::accessor xyz{buf_xyz, cgh, sycl::read_only};
                sycl::accessor P{storage.pressure.get().get_buf_check(id), cgh, sycl::write_only, sycl::no_init};
                sycl::accessor cs{storage.soundspeed.get().get_buf_check(id), cgh, sycl::write_only, sycl::no_init};
                sycl::accessor U{mpdat.pdat.get_field_buf_ref<Tscal>(iuint_interf) , cgh, sycl::read_only};
                sycl::accessor h{mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf), cgh, sycl::read_only};
                
                Tscal cs0   = eos_config->cs0;
                Tscal q     = eos_config->q;
                Tscal r0sq    = eos_config->r0 * eos_config->r0;
                Tscal pmass = gpart_mass;

                cgh.parallel_for(sycl::range<1>{mpdat.total_elements}, [=](sycl::item<1> item) {
                    using namespace shamrock::sph;

                    Tvec R = xyz[item];

                    Tscal Rsq = sycl::dot(R, R);
                    Tscal cs_sq = EOS::soundspeed_sq(cs0*cs0, Rsq/r0sq, q);
                    Tscal cs_out = sycl::sqrt(cs_sq);
                    Tscal rho_a = rho_h(pmass, h[item], Kernel::hfactd);

                    Tscal P_a   = EOS::pressure_from_cs(cs_out*cs_out , rho_a) ;
                    
                    P[item]     = P_a;
                    cs[item]    = cs_out;
                });
            });
            

        });

    } else {
        shambase::throw_unimplemented();
    }
}

using namespace shammath;
template class shammodels::sph::modules::ComputeEos<f64_3, M4>;
template class shammodels::sph::modules::ComputeEos<f64_3, M6>;
template class shammodels::sph::modules::ComputeEos<f64_3, M8>;