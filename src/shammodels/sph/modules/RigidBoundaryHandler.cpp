// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file RigidBoundaryHandler.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "RigidBoundaryHandler.hpp"
#include "shammath/sphkernels.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::RigidBoundaryHandler<Tvec, SPHKernel>::
    advect_particle_apply_boundaries(Tscal dt) {

    struct Plane {
        Tvec u, v, n;
        Tvec center;

        Tscal u_range, v_range;

        bool does_intersect(Tvec r, Tvec v, Tscal &t_intersect) {
            Tvec delt = r - center;

            Tscal delt_n = sycl::dot(delt, n);
            Tscal v_n    = sycl::dot(v, n);
            t_intersect  = -(delt_n / v_n);

            Tvec rt_int = r + (t_intersect*(1 - 10*shambase::get_epsilon<Tscal>())) * v;
            Tscal u_int = sham::dot(rt_int, u);
            Tscal v_int = sham::dot(rt_int, v);

            return (
                sham::abs(u_int) < u_range && sham::abs(v_int) < v_range && t_intersect > 0
                && v_n != 0);
        }
    };

    auto advect_alg = [&](Tvec &pos, Tvec &vel, Plane &p, Tscal dt) {
        Tscal t_loc = 0;
        Tscal next_dt;
        while (t_loc < dt) {

            Tscal dt_intersect;
            bool intersect = p.does_intersect(pos, vel, dt_intersect);

            if (dt_intersect + t_loc < dt && intersect) {
                next_dt = dt_intersect;
            } else {
                next_dt   = dt - t_loc;
                intersect = false;
            }

            pos = pos + next_dt * vel;

            if (intersect) {
                vel = sham::reflect(vel, p.n);
            }

            t_loc += next_dt;
        }
    };


    if constexpr (false) {
        Tvec r = Tvec{0, 0, 0};
        Tvec v = Tvec{0.7, 0.7001, 0};
        Plane p{
                    Tvec(0.0, 0.0, 1.0),
                    Tvec(0.0, 1.0, 0.0),
                    Tvec(1.0, 0.0, 0.0),
                    Tvec(0.5, 0.0, 0.0),
                    0.5,
                    0.5};

        Tscal dt = 0.0001;

        std::ofstream out("out.py");
        out << "table = [";
        for (Tscal t = 0; t < 1; t += dt) {
            out << "["<< t << ", ";
            out << r[0] << ", ";
            out << r[1] << ", ";
            out << r[2] << ", ";
            out << v[0] << ", ";
            out << v[1] << ", ";
            out << v[2] << ", ";
            out << "],\n";
            advect_alg(r, v, p, dt);
        }
        out << "]\n";
        out.close();
        std::abort();
    }



    StackEntry stack_loc{};

    using namespace shamrock::patch;
    PatchDataLayout &pdl = scheduler().pdl;
    const u32 ixyz       = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz      = pdl.get_field_idx<Tvec>("vxyz");

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        sycl::buffer<Tvec> &buf_xyz  = pdat.get_field_buf_ref<Tvec>(ixyz);
        sycl::buffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor acc_r{buf_xyz, cgh, sycl::read_write};
            sycl::accessor acc_v{buf_vxyz, cgh, sycl::read_write};

            cgh.parallel_for(pdat.get_obj_cnt(), [=](sycl::item<1> item) {
                u32 gid = (u32) item.get_id();

                Plane p{
                    Tvec(0.0, 0.0, 1.0),
                    Tvec(0.0, 1.0, 0.0),
                    Tvec(1.0, 0.0, 0.0),
                    Tvec(0.3, 0.0, 0.0),
                    1,
                    1};

                Tvec pos = acc_r[item];
                Tvec vel = acc_v[item];

                advect_alg(pos, vel, p, dt);

                acc_r[item] = pos;
                acc_v[item] = vel;
            });
        });
    });
}

using namespace shammath;
template class shammodels::sph::modules::RigidBoundaryHandler<f64_3, M4>;
template class shammodels::sph::modules::RigidBoundaryHandler<f64_3, M6>;
template class shammodels::sph::modules::RigidBoundaryHandler<f64_3, M8>;
