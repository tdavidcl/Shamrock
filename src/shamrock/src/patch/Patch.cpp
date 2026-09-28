// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Patch.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamrock/patch/Patch.hpp"
#include "shamsys/MpiDataTypeHandler.hpp"

namespace shamrock::patch {

    MPI_Datatype patch_3d_MPI_type;

    template<>
    MPI_Datatype get_patch_mpi_type<3>() {
        return patch_3d_MPI_type;
    }

    // Defined out of line: PatchCoord::merge is costly to instantiate (sycl::min/max) and this
    // function would otherwise instantiate it in every file including Patch.hpp.
    Patch Patch::merge_patch(std::array<Patch, splts_count> patches) {

        PatchCoord merged_c = PatchCoord<dim>::merge(
            {patches[0].get_coords(),
             patches[1].get_coords(),
             patches[2].get_coords(),
             patches[3].get_coords(),
             patches[4].get_coords(),
             patches[5].get_coords(),
             patches[6].get_coords(),
             patches[7].get_coords()});

        Patch ret{};
        ret = patches[0];

        ret.coord_min[0] = merged_c.coord_min[0];
        ret.coord_min[1] = merged_c.coord_min[1];
        ret.coord_min[2] = merged_c.coord_min[2];
        ret.coord_max[0] = merged_c.coord_max[0];
        ret.coord_max[1] = merged_c.coord_max[1];
        ret.coord_max[2] = merged_c.coord_max[2];

        ret.pack_node_index = u64_max;

        ret.load_value += patches[1].load_value;
        ret.load_value += patches[2].load_value;
        ret.load_value += patches[3].load_value;
        ret.load_value += patches[4].load_value;
        ret.load_value += patches[5].load_value;
        ret.load_value += patches[6].load_value;
        ret.load_value += patches[7].load_value;

        return ret;
    }
} // namespace shamrock::patch

/////////////////////////////////////////////
// MPI init related to patches
/////////////////////////////////////////////

MPI_Datatype patch_3d_MPI_types_list[2];
int patch_3d_MPI_block_lens[2];
MPI_Aint patch_3d_MPI_offset[2];

Register_MPIDtypeInit(init_patch_type, "mpi patch type") {
    using namespace shamrock::patch;

    patch_3d_MPI_block_lens[0] = 9; // 9 u64
    patch_3d_MPI_block_lens[1] = 1; // 2 u32

    patch_3d_MPI_types_list[0] = MPI_LONG;
    patch_3d_MPI_types_list[1] = MPI_INT;

    patch_3d_MPI_offset[0] = offsetof(shamrock::patch::Patch, id_patch);
    patch_3d_MPI_offset[1] = offsetof(shamrock::patch::Patch, node_owner_id);

    mpi::type_create_struct(
        2,
        patch_3d_MPI_block_lens,
        patch_3d_MPI_offset,
        patch_3d_MPI_types_list,
        &patch_3d_MPI_type);
    mpi::type_commit(&patch_3d_MPI_type);
}

Register_MPIDtypeFree(free_patch_type, "mpi patch type") {
    using namespace shamrock::patch;

    mpi::type_free(&patch_3d_MPI_type);
}
