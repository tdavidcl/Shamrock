// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "karras_alg.hpp"
#include "aliases.hpp"
#include <stdexcept>

#define SGN(x) (x == 0) ? 0 : ((x > 0) ? 1 : -1)

#ifdef SYCL_COMP_DPCPP
#define DELTA(x, y) ((y > morton_lenght - 1 || y < 0) ? -1 : int(sycl::clz(m[x] ^ m[y])))
#endif

#ifdef SYCL_COMP_HIPSYCL


template<class T>
int internal_clz(T a);

template<>
inline int internal_clz(u32 a){
    return __builtin_clz(a);
}

template<>
inline int internal_clz(u64 a){
    return __builtin_clzl(a);
}



#define DELTA_host(x, y) __hipsycl_if_target_host(((y > morton_lenght - 1 || y < 0) ? -1 : int(internal_clz(m[x] ^ m[y]))))
#define DELTA_cuda(x, y) __hipsycl_if_target_cuda(((y > morton_lenght - 1 || y < 0) ? -1 : int(__clz(m[x] ^ m[y]))))
#define DELTA_hip(x, y) __hipsycl_if_target_hip(((y > morton_lenght - 1 || y < 0) ? -1 : int(__clz(m[x] ^ m[y]))))
#define DELTA_spirv(x, y) __hipsycl_if_target_spirv(((y > morton_lenght - 1 || y < 0) ? -1 : int(__clz(m[x] ^ m[y]))))
#define DELTA(x, y) DELTA_host(x, y) DELTA_cuda(x, y)
#endif

template <class u_morton, class kername>
void __sycl_karras_alg(sycl::queue &queue, u32 internal_cell_count, std::unique_ptr<sycl::buffer<u_morton>> &in_morton,
                       std::unique_ptr<sycl::buffer<u32>> &out_buf_lchild_id,
                       std::unique_ptr<sycl::buffer<u32>> &out_buf_rchild_id,
                       std::unique_ptr<sycl::buffer<u8>> &out_buf_lchild_flag,
                       std::unique_ptr<sycl::buffer<u8>> &out_buf_rchild_flag,
                       std::unique_ptr<sycl::buffer<u32>> &out_buf_endrange) {

    sycl::range<1> range_radix_tree{internal_cell_count};

    if (in_morton == NULL)
        throw shamrock_exc("in_morton isn't allocated");
    if (out_buf_lchild_id == NULL)
        throw shamrock_exc("out_buf_lchild_id isn't allocated");
    if (out_buf_rchild_id == NULL)
        throw shamrock_exc("out_buf_rchild_id isn't allocated");
    if (out_buf_lchild_flag == NULL)
        throw shamrock_exc("out_buf_lchild_flag isn't allocated");
    if (out_buf_rchild_flag == NULL)
        throw shamrock_exc("out_buf_rchild_flag isn't allocated");
    if (out_buf_endrange == NULL)
        throw shamrock_exc("out_buf_endrange isn't allocated");

    queue.submit([&](sycl::handler &cgh) {
        //@TODO add check if split count above 2G
        i32 morton_lenght = (i32)internal_cell_count + 1;

        auto m = in_morton->template get_access<sycl::access::mode::read>(cgh);

        auto lchild_id      = out_buf_lchild_id->get_access<sycl::access::mode::discard_write>(cgh);
        auto rchild_id      = out_buf_rchild_id->get_access<sycl::access::mode::discard_write>(cgh);
        auto lchild_flag    = out_buf_lchild_flag->get_access<sycl::access::mode::discard_write>(cgh);
        auto rchild_flag    = out_buf_rchild_flag->get_access<sycl::access::mode::discard_write>(cgh);
        auto end_range_cell = out_buf_endrange->get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<kername>(range_radix_tree, [=](sycl::item<1> item) {
            int i = (int)item.get_id(0);

            int ddelta = DELTA(i, i + 1) - DELTA(i, i - 1);

            int d = SGN(ddelta);

            // Compute upper bound for the lenght of the range
            int delta_min = DELTA(i, i - d);
            int lmax      = 2;
            while (DELTA(i, i + lmax * d) > delta_min) {
                lmax *= 2;
            }

            // Find the other end using
            int l = 0;
            int t = lmax / 2;
            while (t > 0) {
                if (DELTA(i, i + (l + t) * d) > delta_min) {
                    l = l + t;
                }
                t = t / 2;
            }
            int j = i + l * d;

            end_range_cell[i] = j;

            // Find the split position using binary search
            int delta_node = DELTA(i, j);
            int s          = 0;

            //@todo why float
            float div = 2;
            t         = sycl::ceil(l / div);
            while (true) {
                int tmp_ = i + (s + t) * d;
                if (DELTA(i, tmp_) > delta_node) {
                    s = s + t;
                }
                if (t <= 1)
                    break;
                div *= 2;
                t = sycl::ceil(l / div);
            }
            int gamma = i + s * d + sycl::min(d, 0);

            if (sycl::min(i, j) == gamma) {
                lchild_id[i]   = gamma;
                lchild_flag[i] = 1; // leaf
            } else {
                lchild_id[i]   = gamma;
                lchild_flag[i] = 0; // leaf
            }

            if (sycl::max(i, j) == gamma + 1) {
                rchild_id[i]   = gamma + 1;
                rchild_flag[i] = 1; // leaf
            } else {
                rchild_id[i]   = gamma + 1;
                rchild_flag[i] = 0; // leaf
            }
        });
    }

    );
}

class Kernel_Karras_alg_morton32;
class Kernel_Karras_alg_morton64;

template <>
void sycl_karras_alg<u32>(sycl::queue &queue, u32 internal_cell_count, std::unique_ptr<sycl::buffer<u32>> &in_morton,
                          std::unique_ptr<sycl::buffer<u32>> &out_buf_lchild_id,
                          std::unique_ptr<sycl::buffer<u32>> &out_buf_rchild_id,
                          std::unique_ptr<sycl::buffer<u8>> &out_buf_lchild_flag,
                          std::unique_ptr<sycl::buffer<u8>> &out_buf_rchild_flag,
                          std::unique_ptr<sycl::buffer<u32>> &out_buf_endrange) {
    __sycl_karras_alg<u32, Kernel_Karras_alg_morton32>(queue, internal_cell_count, in_morton, out_buf_lchild_id,
                                                       out_buf_rchild_id, out_buf_lchild_flag, out_buf_rchild_flag,
                                                       out_buf_endrange);
}

template <>
void sycl_karras_alg<u64>(sycl::queue &queue, u32 internal_cell_count, std::unique_ptr<sycl::buffer<u64>> &in_morton,
                          std::unique_ptr<sycl::buffer<u32>> &out_buf_lchild_id,
                          std::unique_ptr<sycl::buffer<u32>> &out_buf_rchild_id,
                          std::unique_ptr<sycl::buffer<u8>> &out_buf_lchild_flag,
                          std::unique_ptr<sycl::buffer<u8>> &out_buf_rchild_flag,
                          std::unique_ptr<sycl::buffer<u32>> &out_buf_endrange) {
    __sycl_karras_alg<u64, Kernel_Karras_alg_morton64>(queue, internal_cell_count, in_morton, out_buf_lchild_id,
                                                       out_buf_rchild_id, out_buf_lchild_flag, out_buf_rchild_flag,
                                                       out_buf_endrange);
}