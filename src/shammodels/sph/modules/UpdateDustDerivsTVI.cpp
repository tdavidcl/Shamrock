// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file UpdateViscosity.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "UpdateDustDerivsTVI.hpp"
#include "shammath/sphkernels.hpp"


template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::UpdateDustDerivsTVI<Tvec, SPHKernel>::update_dust_derivs_tvi() {

}




using namespace shammath;
template class shammodels::sph::modules::UpdateDustDerivsTVI<f64_3, M4>;
template class shammodels::sph::modules::UpdateDustDerivsTVI<f64_3, M6>;