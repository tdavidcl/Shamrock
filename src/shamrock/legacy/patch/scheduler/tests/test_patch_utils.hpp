// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include <vector>
#include <random>

#include "shamrock/scheduler/scheduler_mpi.hpp"


inline void make_global_local_check_vec(std::vector<shamrock::patch::Patch> & global, std::vector<shamrock::patch::Patch> & local){

    using namespace shamrock::patch;
    
    global.resize(shamsys::instance::world_size*6);
    {
        //fill the check vector with a pseudo random int generator (seed:0x1111)
        std::mt19937 eng(0x1111);        
        std::uniform_int_distribution<u32> distu32(u32_min,u32_max);                  
        std::uniform_int_distribution<u64> distu64(u64_min,u64_max);

        std::uniform_int_distribution<u32> distdtcnt(0,10000); 

        u64 id_patch = 0;
        for (Patch & element : global) {
            element.id_patch      = id_patch;
            element.pack_node_index     = u64_max;
            element.load_value    = distdtcnt(eng);
            element.x_min         = distu64(eng);
            element.y_min         = distu64(eng);
            element.z_min         = distu64(eng);
            element.x_max         = distu64(eng);
            element.y_max         = distu64(eng);
            element.z_max         = distu64(eng);
            element.data_count    = element.load_value;
            element.node_owner_id = distu32(eng);


            //if(id_patch > 7) element.pack_node_index = 10;

            id_patch++;
        }
    }



    {
        std::vector<u32> pointer_start_node(shamsys::instance::world_size);
        pointer_start_node[0] = 0;
        for(i32 i = 1; i < shamsys::instance::world_size; i ++){
            pointer_start_node[i] = pointer_start_node[i-1] + ((i-1)%5)*((i-1)%5);
        }
        pointer_start_node.push_back(shamsys::instance::world_size*6);


        for(i32 irank = 0; irank < shamsys::instance::world_size; irank ++){
            for(u32 id = pointer_start_node[irank]; id < pointer_start_node[irank+1]; id ++){
                global[id].node_owner_id = irank;
            }
        }

        for(u32 id = pointer_start_node[shamsys::instance::world_rank]; id < pointer_start_node[shamsys::instance::world_rank+1]; id ++){
            local.push_back(global[id]);
        }
    }


}
