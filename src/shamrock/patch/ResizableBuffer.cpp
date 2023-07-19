// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "ResizableBuffer.hpp"
#include "shamalgs/random/random.hpp"
#include "shamalgs/reduction/reduction.hpp"
#include "shambase/sycl_utils/vec_equals.hpp"
#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////
// memory manipulation
////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void ResizableBuffer<T>::alloc() {
    buf = std::make_unique<sycl::buffer<T>>(capacity);

    logger::debug_alloc_ln("PatchDataField", "allocate field :", "len =", capacity);
}

template<class T>
void ResizableBuffer<T>::free() {

    if (buf) {
        logger::debug_alloc_ln("PatchDataField", "free field :", "len =", capacity);

        buf.reset();
    }
}

template<class T>
void ResizableBuffer<T>::change_capacity(u32 new_capa) {

    logger::debug_alloc_ln(
        "ResizableBuffer", "change capacity from : ", capacity, "to :", new_capa);

    if (capacity == 0) {

        if (new_capa > 0) {
            capacity = new_capa;
            alloc();
        }

    } else {

        if (new_capa > 0) {

            if (new_capa != capacity) {
                capacity = new_capa;

                sycl::buffer<T> *old_buf = buf.release();

                alloc();

                if (val_cnt > 0) {
                    shamalgs::memory::copybuf_discard(*old_buf, *buf, std::min(val_cnt,capacity));
                }

                logger::debug_alloc_ln("PatchDataField", "delete old buf : ");
                delete old_buf;
            }

        } else {
            capacity = 0;
            free();
        }
    }
}


template<class T>
void ResizableBuffer<T>::reserve(u32 add_size) {
    StackEntry stack_loc{false};

    u32 wanted_sz = val_cnt + add_size;

    if(wanted_sz > capacity){
        change_capacity(wanted_sz*safe_fact);
    }
}

template<class T>
void ResizableBuffer<T>::resize(u32 new_size) {
    StackEntry stack_loc{false};

    logger::debug_alloc_ln("ResizableBuffer", "resize from : ", val_cnt, "to :", new_size);

    if (new_size > capacity) {
        change_capacity(new_size*safe_fact);
    }

    val_cnt = new_size;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// value manipulation
////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void ResizableBuffer<T>::overwrite(ResizableBuffer<T> &f2, u32 cnt) {
    if (val_cnt < cnt) {
        throw shambase::throw_with_loc<std::invalid_argument>(
            "to overwrite you need more element in the field");
    }

    {
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};
        sycl::host_accessor acc_f2{*f2.get_buf(), sycl::read_only};

        for (u32 i = 0; i < cnt; i++) {
            // field_data[idx_st + i] = f2.field_data[i];
            acc[i] = acc_f2[i];
        }
    }
}

template<class T>
void ResizableBuffer<T>::override(sycl::buffer<T> &data, u32 cnt) {

    if (cnt != val_cnt)
        throw shambase::throw_with_loc<std::invalid_argument>(
            "buffer size doesn't match patchdata field size"); // TODO remove ref to size

    if (val_cnt > 0) {

        {
            sycl::host_accessor acc_cur{*buf, sycl::write_only, sycl::no_init};
            sycl::host_accessor acc{data, sycl::read_only};

            for (u32 i = 0; i < val_cnt; i++) {
                // field_data[i] = acc[i];
                acc_cur[i] = acc[i];
            }
        }
    }
}

template<class T>
void ResizableBuffer<T>::override(const T val) {

    if (val_cnt > 0) {

        {
            sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};
            for (u32 i = 0; i < val_cnt; i++) {
                // field_data[i] = val;
                acc[i] = val;
            }
        }
    }
}

template<class T>
void ResizableBuffer<T>::index_remap_resize(sycl::buffer<u32> &index_map, u32 len, u32 nvar) {
    if (get_buf()) {

        auto get_new_buf = [&]() {
            if (nvar == 1) {
                return shamalgs::algorithm::index_remap(
                    shamsys::instance::get_compute_queue(), *get_buf(), index_map, len);
            } else {
                return shamalgs::algorithm::index_remap_nvar(
                    shamsys::instance::get_compute_queue(), *get_buf(), index_map, len, nvar);
            }
        };

        sycl::buffer<T> new_buf = get_new_buf();

        capacity = new_buf.size();
        val_cnt  = len * nvar;
        buf      = std::make_unique<sycl::buffer<T>>(std::move(new_buf));
    }
}

template<class T>
void ResizableBuffer<T>::serialize_buf(shamalgs::SerializeHelper &serializer) {
    if (buf) {
        serializer.write_buf(*buf, val_cnt);
    }
}

template<class T>
ResizableBuffer<T>
ResizableBuffer<T>::deserialize_buf(shamalgs::SerializeHelper &serializer, u32 val_cnt) {
    if (val_cnt == 0) {
        return ResizableBuffer();
    } else {
        ResizableBuffer rbuf(val_cnt);
        serializer.load_buf(*(rbuf.buf), val_cnt);
        return std::move(rbuf);
    }
}

template<class T>
bool ResizableBuffer<T>::check_buf_match(const ResizableBuffer<T> &f2) const {
    bool match = true;

    match = match && (val_cnt == f2.val_cnt);

    {

        using buf_t = std::unique_ptr<sycl::buffer<T>>;

        const buf_t &buf    = get_buf();
        const buf_t &buf_f2 = f2.get_buf();

        sycl::buffer<u8> res_buf(val_cnt);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor acc1{*buf, cgh, sycl::read_only};
            sycl::accessor acc2{*buf_f2, cgh, sycl::read_only};

            sycl::accessor acc_res{res_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{val_cnt}, [=](sycl::item<1> i) {
                acc_res[i] = shambase::vec_equals(acc1[i], acc2[i]);
            });
        });

        match = match && shamalgs::reduction::is_all_true(res_buf, f2.size());
    }

    return match;
}

template<class T>
u64 ResizableBuffer<T>::serialize_buf_byte_size() {
    using H = shamalgs::SerializeHelper;
    return H::serialize_byte_size<T>(val_cnt);
}

template<class T>
ResizableBuffer<T>
ResizableBuffer<T>::mock_buffer(u64 seed, u32 val_cnt, T min_bound, T max_bound) {
    sycl::buffer<T> buf_mocked = shamalgs::random::mock_buffer(seed, val_cnt, min_bound, max_bound);
    return ResizableBuffer<T>(std::move(buf_mocked), val_cnt);
}

//////////////////////////////////////////////////////////////////////////
// Define the patchdata field for all classes in XMAC_LIST_ENABLED_FIELD
//////////////////////////////////////////////////////////////////////////

#define X(a) template class ResizableBuffer<a>;
XMAC_LIST_ENABLED_FIELD
#undef X

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////