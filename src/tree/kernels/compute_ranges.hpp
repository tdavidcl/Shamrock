#pragma once
#include "CL/sycl/queue.hpp"
#include "aliases.hpp"
#include "sfc/morton.hpp"
#include <memory>


template<class u_morton>
void compute_cell_ranges(

    sycl::queue & queue,

    u32 leaf_cnt , 
    u32 internal_cnt ,
    std::unique_ptr<sycl::buffer<u_morton>> & buf_morton,
    std::unique_ptr<sycl::buffer<u32>> & buf_lchild_id,
    std::unique_ptr<sycl::buffer<u32>> & buf_rchild_id,
    std::unique_ptr<sycl::buffer<u8>>  & buf_lchild_flag,
    std::unique_ptr<sycl::buffer<u8>>  & buf_rchild_flag,
    std::unique_ptr<sycl::buffer<u32>> & buf_endrange,
    
    std::unique_ptr<sycl::buffer<typename morton_3d::morton_types<u_morton>::int_vec_repr>> & buf_pos_min_cell,
    std::unique_ptr<sycl::buffer<typename morton_3d::morton_types<u_morton>::int_vec_repr>> & buf_pos_max_cell);

