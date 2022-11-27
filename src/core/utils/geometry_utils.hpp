// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file geometry_utils.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * @version 0.1
 * @date 2022-03-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include "aliases.hpp"
#include <tuple>



template<class flt>
class ALignedAxisBoundingBox{
    //TODO replace std::tuple<vec,vec> by this class
    using vec = sycl::vec<flt,3>;

    public:
    vec min_coord;
    vec max_coord;

    ALignedAxisBoundingBox(const vec & min, const vec & max) : min_coord(min), max_coord(max){}

    inline vec get_size() const {
        return max_coord - min_coord;
    }

    inline flt get_max_side_lenght() const {
        const vec sz = get_size();
        return sycl::fmax(sycl::fmax(sz.x(),sz.y()),sz.z());
    }
};



namespace BBAA {

    

    template<class VecType> bool is_particle_in_patch(VecType part_pos,VecType pos_min_patch,VecType pos_max_patch);

    template<> inline bool is_particle_in_patch<f32_3>(f32_3 part_pos,f32_3 pos_min_patch,f32_3 pos_max_patch) {
        return (
                (pos_min_patch.x() <= part_pos.x()) && (part_pos.x() < pos_max_patch.x()) &&
                (pos_min_patch.y() <= part_pos.y()) && (part_pos.y() < pos_max_patch.y()) &&
                (pos_min_patch.z() <= part_pos.z()) && (part_pos.z() < pos_max_patch.z()) 
            );
    }

    template<> inline bool is_particle_in_patch<f64_3>(f64_3 part_pos,f64_3 pos_min_patch,f64_3 pos_max_patch) {
        return (
                (pos_min_patch.x() <= part_pos.x()) && (part_pos.x() < pos_max_patch.x()) &&
                (pos_min_patch.y() <= part_pos.y()) && (part_pos.y() < pos_max_patch.y()) &&
                (pos_min_patch.z() <= part_pos.z()) && (part_pos.z() < pos_max_patch.z()) 
            );
    }



    template<class VecType> bool iscellb_inside_a(VecType pos_min_cella,VecType pos_max_cella,VecType pos_min_cellb,VecType pos_max_cellb);
    
    template<> inline bool iscellb_inside_a<u32_3>(u32_3 pos_min_cella,u32_3 pos_max_cella,u32_3 pos_min_cellb,u32_3 pos_max_cellb) {
        return (
                (pos_min_cella.x() <= pos_min_cellb.x()) && (pos_min_cellb.x() < pos_max_cellb.x()) && (pos_max_cellb.x() <= pos_max_cella.x()) &&
                (pos_min_cella.y() <= pos_min_cellb.y()) && (pos_min_cellb.y() < pos_max_cellb.y()) && (pos_max_cellb.y() <= pos_max_cella.y()) &&
                (pos_min_cella.z() <= pos_min_cellb.z()) && (pos_min_cellb.z() < pos_max_cellb.z()) && (pos_max_cellb.z() <= pos_max_cella.z()) 
            );
    }

    template<> inline bool iscellb_inside_a<f32_3>(f32_3 pos_min_cella,f32_3 pos_max_cella,f32_3 pos_min_cellb,f32_3 pos_max_cellb) {
        return (
                (pos_min_cella.x() <= pos_min_cellb.x()) && (pos_min_cellb.x() < pos_max_cellb.x()) && (pos_max_cellb.x() <= pos_max_cella.x()) &&
                (pos_min_cella.y() <= pos_min_cellb.y()) && (pos_min_cellb.y() < pos_max_cellb.y()) && (pos_max_cellb.y() <= pos_max_cella.y()) &&
                (pos_min_cella.z() <= pos_min_cellb.z()) && (pos_min_cellb.z() < pos_max_cellb.z()) && (pos_max_cellb.z() <= pos_max_cella.z()) 
            );
    }






    template<class VecType> bool cella_neigh_b(VecType pos_min_cella,VecType pos_max_cella,VecType pos_min_cellb,VecType pos_max_cellb);

    template<> inline bool cella_neigh_b<f32_3>(f32_3 pos_min_cella,f32_3 pos_max_cella,f32_3 pos_min_cellb,f32_3 pos_max_cellb){
        return (
                (sycl::fmax( pos_min_cella.x(), pos_min_cellb.x()) <= sycl::fmin(pos_max_cella.x(),pos_max_cellb.x())) &&
                (sycl::fmax( pos_min_cella.y(), pos_min_cellb.y()) <= sycl::fmin(pos_max_cella.y(),pos_max_cellb.y())) &&
                (sycl::fmax( pos_min_cella.z(), pos_min_cellb.z()) <= sycl::fmin(pos_max_cella.z(),pos_max_cellb.z())) 
            );
    }

    template<> inline bool cella_neigh_b<f64_3>(f64_3 pos_min_cella,f64_3 pos_max_cella,f64_3 pos_min_cellb,f64_3 pos_max_cellb){
        return (
                (sycl::fmax( pos_min_cella.x(), pos_min_cellb.x()) <= sycl::fmin(pos_max_cella.x(),pos_max_cellb.x())) &&
                (sycl::fmax( pos_min_cella.y(), pos_min_cellb.y()) <= sycl::fmin(pos_max_cella.y(),pos_max_cellb.y())) &&
                (sycl::fmax( pos_min_cella.z(), pos_min_cellb.z()) <= sycl::fmin(pos_max_cella.z(),pos_max_cellb.z())) 
            );
    }




    template<class VecType> bool intersect_not_null_cella_b(VecType pos_min_cella,VecType pos_max_cella,VecType pos_min_cellb,VecType pos_max_cellb);

    template<> inline bool intersect_not_null_cella_b<f64_3>(f64_3 pos_min_cella,f64_3 pos_max_cella,f64_3 pos_min_cellb,f64_3 pos_max_cellb){
        return (
                (sycl::fmax( pos_min_cella.x(), pos_min_cellb.x()) < sycl::fmin(pos_max_cella.x(),pos_max_cellb.x())) &&
                (sycl::fmax( pos_min_cella.y(), pos_min_cellb.y()) < sycl::fmin(pos_max_cella.y(),pos_max_cellb.y())) &&
                (sycl::fmax( pos_min_cella.z(), pos_min_cellb.z()) < sycl::fmin(pos_max_cella.z(),pos_max_cellb.z())) 
            );
    }

    template<> inline bool intersect_not_null_cella_b<f32_3>(f32_3 pos_min_cella,f32_3 pos_max_cella,f32_3 pos_min_cellb,f32_3 pos_max_cellb){
        return (
                (sycl::fmax( pos_min_cella.x(), pos_min_cellb.x()) < sycl::fmin(pos_max_cella.x(),pos_max_cellb.x())) &&
                (sycl::fmax( pos_min_cella.y(), pos_min_cellb.y()) < sycl::fmin(pos_max_cella.y(),pos_max_cellb.y())) &&
                (sycl::fmax( pos_min_cella.z(), pos_min_cellb.z()) < sycl::fmin(pos_max_cella.z(),pos_max_cellb.z())) 
            );
    }

    template<> inline bool intersect_not_null_cella_b<u64_3>(u64_3 pos_min_cella,u64_3 pos_max_cella,u64_3 pos_min_cellb,u64_3 pos_max_cellb){
        return (
                (sycl::max( pos_min_cella.x(), pos_min_cellb.x()) < sycl::min(pos_max_cella.x(),pos_max_cellb.x())) &&
                (sycl::max( pos_min_cella.y(), pos_min_cellb.y()) < sycl::min(pos_max_cella.y(),pos_max_cellb.y())) &&
                (sycl::max( pos_min_cella.z(), pos_min_cellb.z()) < sycl::min(pos_max_cella.z(),pos_max_cellb.z())) 
            );
    }

    template<> inline bool intersect_not_null_cella_b<u32_3>(u32_3 pos_min_cella,u32_3 pos_max_cella,u32_3 pos_min_cellb,u32_3 pos_max_cellb){
        return (
                (sycl::max( pos_min_cella.x(), pos_min_cellb.x()) < sycl::min(pos_max_cella.x(),pos_max_cellb.x())) &&
                (sycl::max( pos_min_cella.y(), pos_min_cellb.y()) < sycl::min(pos_max_cella.y(),pos_max_cellb.y())) &&
                (sycl::max( pos_min_cella.z(), pos_min_cellb.z()) < sycl::min(pos_max_cella.z(),pos_max_cellb.z())) 
            );
    }


    template<class VecType> std::tuple<VecType,VecType> get_intersect_cella_b(VecType pos_min_cella,VecType pos_max_cella,VecType pos_min_cellb,VecType pos_max_cellb);

    template<> inline std::tuple<f64_3,f64_3> get_intersect_cella_b<f64_3>(f64_3 pos_min_cella,f64_3 pos_max_cella,f64_3 pos_min_cellb,f64_3 pos_max_cellb){
        return {sycl::fmax(pos_min_cella,pos_min_cellb), sycl::fmin(pos_max_cella,pos_max_cellb)};
    }

    template<> inline std::tuple<f32_3,f32_3> get_intersect_cella_b<f32_3>(f32_3 pos_min_cella,f32_3 pos_max_cella,f32_3 pos_min_cellb,f32_3 pos_max_cellb){

        return {sycl::fmax(pos_min_cella,pos_min_cellb), sycl::fmin(pos_max_cella,pos_max_cellb)};
    }


    template<class VecType> typename VecType::element_type get_sq_distance_to_BBAAsurface(VecType pos,VecType pos_min_cell,VecType pos_max_cell);

    template<> inline f32 get_sq_distance_to_BBAAsurface<f32_3>(f32_3 pos,f32_3 pos_min_cell,f32_3 pos_max_cell){
        f32_3 clamped;

        clamped.x() = sycl::clamp(pos.x(),pos_min_cell.x(),pos_max_cell.x());
        clamped.y() = sycl::clamp(pos.y(),pos_min_cell.y(),pos_max_cell.y());
        clamped.z() = sycl::clamp(pos.z(),pos_min_cell.z(),pos_max_cell.z());

        clamped -= pos;

        return sycl::dot(clamped,clamped);
    }

    template<> inline f64 get_sq_distance_to_BBAAsurface<f64_3>(f64_3 pos,f64_3 pos_min_cell,f64_3 pos_max_cell){
        f64_3 clamped;

        clamped.x() = sycl::clamp(pos.x(),pos_min_cell.x(),pos_max_cell.x());
        clamped.y() = sycl::clamp(pos.y(),pos_min_cell.y(),pos_max_cell.y());
        clamped.z() = sycl::clamp(pos.z(),pos_min_cell.z(),pos_max_cell.z());

        clamped -= pos;

        return sycl::dot(clamped,clamped);
    }

}