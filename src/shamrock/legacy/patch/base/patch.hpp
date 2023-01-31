// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file patch.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Header file for the patch struct and related function 
 * @version 1.0
 * @date 2022-02-28
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once

#include "shamrock/patch/Patch.hpp"


#include "shamsys/legacy/mpi_handler.hpp"





/**
 * @brief patch related functions
 */
namespace patch {

    /**
     * @brief split patch \p p0 in p0 -> p7
     * 
     * @param p0 
     * @param p1 
     * @param p2 
     * @param p3 
     * @param p4 
     * @param p5 
     * @param p6 
     * @param p7 
     */
    void split_patch_obj(shamrock::patch::Patch & p0, shamrock::patch::Patch & p1,shamrock::patch::Patch & p2,shamrock::patch::Patch & p3,shamrock::patch::Patch & p4,shamrock::patch::Patch & p5,shamrock::patch::Patch & p6,shamrock::patch::Patch & p7);

    /**
     * @brief merge patch \p p0 -> p7 into p0
     * 
     * @param p0 
     * @param p1 
     * @param p2 
     * @param p3 
     * @param p4 
     * @param p5 
     * @param p6 
     * @param p7 
     */
    void merge_patch_obj(shamrock::patch::Patch & p0, shamrock::patch::Patch & p1,shamrock::patch::Patch & p2,shamrock::patch::Patch & p3,shamrock::patch::Patch & p4,shamrock::patch::Patch & p5,shamrock::patch::Patch & p6,shamrock::patch::Patch & p7);





}
