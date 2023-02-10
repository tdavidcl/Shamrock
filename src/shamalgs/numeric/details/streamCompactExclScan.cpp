// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "streamCompactExclScan.hpp"
#include "shamalgs/memory/memory.hpp"
#include "shamalgs/numeric/numeric.hpp"
#include "shamsys/legacy/log.hpp"

class StreamCompactionAlg;

namespace shamalgs::numeric::details {

    std::tuple<sycl::buffer<u32>, u32>
    stream_compact_excl_scan(sycl::queue &q, sycl::buffer<u32> &buf_flags, u32 len) {

        // perform the exclusive sum of the buf flag
        sycl::buffer<u32> excl_sum = exclusive_sum(q, buf_flags, len);

        // recover the end value of the sum to know the new size
        u32 new_len = memory::extract_element(q, excl_sum, len-1);

        u32 end_flag = memory::extract_element(q, buf_flags, len-1);

        if(end_flag){
            new_len ++;
        }

        // create the index buffer that we will return
        sycl::buffer<u32> index_map{new_len};

        q.submit([&](sycl::handler & cgh){

            sycl::accessor sum_vals {excl_sum,cgh,sycl::read_only};
            sycl::accessor new_idx {index_map,cgh,sycl::write_only, sycl::no_init};

            u32 last_idx = len-1;
            u32 last_flag = end_flag;

            cgh.parallel_for<StreamCompactionAlg>(sycl::range<1>{len},[=](sycl::item<1> i){

                const u32 idx = i.get_linear_id();

                u32 current_val = sum_vals[idx];
                
                if(idx < last_idx){
                    if(current_val < sum_vals[idx+1]){
                        new_idx[current_val] = idx;
                    }
                }else if(last_flag){
                    new_idx[current_val] = idx;
                }

            });

        });

        return {std::move(index_map),new_len};
    };

} // namespace shamalgs::numeric::details