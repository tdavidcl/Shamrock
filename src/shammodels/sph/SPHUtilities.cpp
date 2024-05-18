// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SPHUtilities.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "SPHUtilities.hpp"
#include "shamrock/tree/TreeTraversal.hpp"
#include "shammath/sphkernels.hpp"

using namespace shamrock::sph;

namespace shammodels::sph {

    template<class vec, class SPHKernel>
    void SPHUtilities<vec, SPHKernel>::iterate_smoothing_length_cache(

        sycl::buffer<vec> &merged_r,
        sycl::buffer<flt> &hnew,
        sycl::buffer<flt> &hold,
        sycl::buffer<flt> &eps_h,
        sycl::range<1> update_range,
        shamrock::tree::ObjectCache &neigh_cache,

        flt gpart_mass,
        flt h_evol_max,
        flt h_evol_iter_max

    ) {
        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            // tree::ObjectIterator particle_looper(tree,cgh);

            shamrock::tree::ObjectCacheIterator particle_looper(neigh_cache, cgh);

            sycl::accessor eps{eps_h, cgh, sycl::read_write};
            sycl::accessor r{merged_r, cgh, sycl::read_only};
            sycl::accessor h_new{hnew, cgh, sycl::read_write};
            sycl::accessor h_old{hold, cgh, sycl::read_only};
            // sycl::accessor omega {omega_h, cgh, sycl::write_only, sycl::no_init};

            const flt part_mass          = gpart_mass;
            const flt h_max_tot_max_evol = h_evol_max;
            const flt h_max_evol_p       = h_evol_iter_max;
            const flt h_max_evol_m       = 1 / h_evol_iter_max;

            shambase::parralel_for(cgh, update_range.size(),"iter h", [=](u32 id_a) {

                if (eps[id_a] > 1e-6) {

                    vec xyz_a = r[id_a]; // could be recovered from lambda

                    flt h_a  = h_new[id_a];
                    flt dint = h_a * h_a * Rkern * Rkern;

                    vec inter_box_a_min = xyz_a - h_a * Rkern;
                    vec inter_box_a_max = xyz_a + h_a * Rkern;

                    flt rho_sum = 0;
                    flt sumdWdh = 0;

                    // particle_looper.rtree_for([&](u32, vec bmin,vec bmax) -> bool {
                    //     return
                    //     shammath::domain_are_connected(bmin,bmax,inter_box_a_min,inter_box_a_max);
                    // },[&](u32 id_b){
                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        vec dr   = xyz_a - r[id_b];
                        flt rab2 = sycl::dot(dr, dr);

                        if (rab2 > dint) {
                            return;
                        }

                        flt rab = sycl::sqrt(rab2);

                        rho_sum += part_mass * SPHKernel::W_3d(rab, h_a);
                        sumdWdh += part_mass * SPHKernel::dhW_3d(rab, h_a);
                    });

                    using namespace shamrock::sph;

                    flt rho_ha = rho_h(part_mass, h_a,SPHKernel::hfactd);
                    flt new_h  = newtown_iterate_new_h(rho_ha, rho_sum, sumdWdh, h_a);

                    if (new_h < h_a * h_max_evol_m)
                        new_h = h_max_evol_m * h_a;
                    if (new_h > h_a * h_max_evol_p)
                        new_h = h_max_evol_p * h_a;

                    flt ha_0 = h_old[id_a];

                    if (new_h < ha_0 * h_max_tot_max_evol) {
                        h_new[id_a] = new_h;
                        eps[id_a]   = sycl::fabs(new_h - h_a) / ha_0;
                    } else {
                        h_new[id_a] = ha_0 * h_max_tot_max_evol;
                        eps[id_a]   = -1;
                    }
                }
            });
        }).wait();
    }

    template<class vec, class SPHKernel, class u_morton>
    void SPHTreeUtilities<vec, SPHKernel, u_morton>::iterate_smoothing_length_tree(
        sycl::buffer<vec> &merged_r,
        sycl::buffer<flt> &hnew,
        sycl::buffer<flt> &hold,
        sycl::buffer<flt> &eps_h,
        sycl::range<1> update_range,
        RadixTree<u_morton, vec> &tree,

        flt gpart_mass,
        flt h_evol_max,
        flt h_evol_iter_max) {

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            shamrock::tree::ObjectIterator particle_looper(tree, cgh);
            // shamrock::tree::ObjectCacheIterator particle_looper(neigh_cache, cgh);

            sycl::accessor eps{eps_h, cgh, sycl::read_write};
            sycl::accessor r{merged_r, cgh, sycl::read_only};
            sycl::accessor h_new{hnew, cgh, sycl::read_write};
            sycl::accessor h_old{hold, cgh, sycl::read_only};
            // sycl::accessor omega {omega_h, cgh, sycl::write_only, sycl::no_init};

            const flt part_mass          = gpart_mass;
            const flt h_max_tot_max_evol = h_evol_max;
            const flt h_max_evol_p       = h_evol_iter_max;
            const flt h_max_evol_m       = 1 / h_evol_iter_max;

            shambase::parralel_for(cgh, update_range.size(),"iter h", [=](u32 id_a) {

                if (eps[id_a] > 1e-6) {

                    vec xyz_a = r[id_a]; // could be recovered from lambda

                    flt h_a  = h_new[id_a];
                    flt dint = h_a * h_a * Rkern * Rkern;

                    vec inter_box_a_min = xyz_a - h_a * Rkern;
                    vec inter_box_a_max = xyz_a + h_a * Rkern;

                    flt rho_sum = 0;
                    flt sumdWdh = 0;

                    particle_looper.rtree_for(
                        [&](u32, vec bmin, vec bmax) -> bool {
                            return shammath::domain_are_connected(
                                bmin, bmax, inter_box_a_min, inter_box_a_max);
                        },
                        [&](u32 id_b) {
                            // particle_looper.for_each_object(id_a, [&](u32 id_b) {
                            vec dr   = xyz_a - r[id_b];
                            flt rab2 = sycl::dot(dr, dr);

                            if (rab2 > dint) {
                                return;
                            }

                            flt rab = sycl::sqrt(rab2);

                            rho_sum += part_mass * SPHKernel::W_3d(rab, h_a);
                            sumdWdh += part_mass * SPHKernel::dhW_3d(rab, h_a);
                        });

                    using namespace shamrock::sph;

                    flt rho_ha = rho_h(part_mass, h_a,SPHKernel::hfactd);
                    flt new_h  = newtown_iterate_new_h(rho_ha, rho_sum, sumdWdh, h_a);

                    if (new_h < h_a * h_max_evol_m)
                        new_h = h_max_evol_m * h_a;
                    if (new_h > h_a * h_max_evol_p)
                        new_h = h_max_evol_p * h_a;

                    flt ha_0 = h_old[id_a];

                    if (new_h < ha_0 * h_max_tot_max_evol) {
                        h_new[id_a] = new_h;
                        eps[id_a]   = sycl::fabs(new_h - h_a) / ha_0;
                    } else {
                        h_new[id_a] = ha_0 * h_max_tot_max_evol;
                        eps[id_a]   = -1;
                    }
                }
            });
        }).wait();
    }

    template<class vec, class SPHKernel>
    void SPHUtilities<vec, SPHKernel>::compute_omega(sycl::buffer<vec> &merged_r,
                                                     sycl::buffer<flt> &h_part,
                                                     sycl::buffer<flt> &omega_h,
                                                     sycl::range<1> part_range,
                                                     shamrock::tree::ObjectCache &neigh_cache,
                                                     flt gpart_mass) {
        using namespace shamrock::tree;

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            // tree::ObjectIterator particle_looper(tree,cgh);

            ObjectCacheIterator particle_looper(neigh_cache, cgh);

            sycl::accessor r{merged_r, cgh, sycl::read_only};
            sycl::accessor hpart{h_part, cgh, sycl::read_only};
            sycl::accessor omega{omega_h, cgh, sycl::write_only, sycl::no_init};

            const flt part_mass = gpart_mass;

            shambase::parralel_for(cgh, part_range.size(),"compute omega", [=](u32 id_a) {

                vec xyz_a = r[id_a]; // could be recovered from lambda

                flt h_a  = hpart[id_a];
                flt dint = h_a * h_a * Rkern * Rkern;

                //vec inter_box_a_min = xyz_a - h_a * Rkern;
                //vec inter_box_a_max = xyz_a + h_a * Rkern;

                flt rho_sum        = 0;
                flt part_omega_sum = 0;

                // particle_looper.rtree_for([&](u32, vec bmin,vec bmax) -> bool {
                //     return
                //     shammath::domain_are_connected(bmin,bmax,inter_box_a_min,inter_box_a_max);
                // },[&](u32 id_b){
                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    vec dr   = xyz_a - r[id_b];
                    flt rab2 = sycl::dot(dr, dr);

                    if (rab2 > dint) {
                        return;
                    }

                    flt rab = sycl::sqrt(rab2);

                    rho_sum += part_mass * SPHKernel::W_3d(rab, h_a);
                    part_omega_sum += part_mass * SPHKernel::dhW_3d(rab, h_a);
                });

                using namespace shamrock::sph;

                flt rho_ha  = rho_h(part_mass, h_a,SPHKernel::hfactd);
                flt omega_a = 1 + (h_a / (3 * rho_ha)) * part_omega_sum;
                omega[id_a] = omega_a;

                // logger::raw(shambase::format("pmass {}, rho_a {}, omega_a {}\n",
                // part_mass,rho_ha, omega_a));
            });
        }).wait();
    }

    template class SPHUtilities<f64_3, shammath::M4<f64>>;
    template class SPHUtilities<f64_3, shammath::M6<f64>>;
    template class SPHUtilities<f64_3, shammath::M8<f64>>;

    template class SPHTreeUtilities<f64_3, shammath::M4<f64>, u32>;
    template class SPHTreeUtilities<f64_3, shammath::M6<f64>, u64>;
    template class SPHTreeUtilities<f64_3, shammath::M8<f64>, u64>;

} // namespace shammodels::sph