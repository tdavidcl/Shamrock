// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ResizableBuffer.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/container/ResizableBuffer.hpp"
#include "shamalgs/random.hpp"
#include "shamalgs/reduction.hpp"
#include "shamcomm/logs.hpp"
namespace shamalgs {
    u64 _ibuf = 0;
    u64 next_ibuf(SourceLocation loc) {
        _ibuf++;
        // if(_ibuf == 19){
        //     shambase::throw_with_loc<std::runtime_error>("i just wanna");
        // }
        return _ibuf;
    }
} // namespace shamalgs
////////////////////////////////////////////////////////////////////////////////////////////////
// memory manipulation
////////////////////////////////////////////////////////////////////////////////////////////////

u64 alloc_counter = 0;
u64 buf_cnt       = 0;
std::unordered_map<u64, i64> map_cnt;

void print_state(std::string info, u64 bytesize, u64 ibuf) {
    // get the sum of the map
    u64 sum = 0;
    for (auto &it : map_cnt) {
        sum += it.first * it.second;
    }

    logger::info_ln(
        "ResizableBuffer",
        info,
        bytesize,
        "ibuf",
        ibuf,
        "alloc size :",
        shambase::readable_sizeof(alloc_counter),
        "count :",
        buf_cnt,
        shambase::format("{}", map_cnt),
        "sum",
        sum,
        alloc_counter);
}

void register_new(u64 bytesize, u64 ibuf) {

    if (map_cnt.count(bytesize) == 0) {
        map_cnt[bytesize] = 0;
    }
    map_cnt[bytesize]++;

    alloc_counter += bytesize;
    buf_cnt++;

    print_state("new", bytesize, ibuf);
}
void register_del(u64 bytesize, u64 ibuf) {

    if (map_cnt.count(bytesize) == 0) {
        print_state("del throw", bytesize, ibuf);
        throw "";
    }
    map_cnt[bytesize]--;

    if (map_cnt[bytesize] == 0) {
        map_cnt.erase(bytesize);
    }

    alloc_counter -= bytesize;
    buf_cnt--;

    print_state("del", bytesize, ibuf);
}

template<class T>
sycl::buffer<T> get_new_buf(u64 len, u64 ibuf) {

    register_new(len * sizeof(T), ibuf);

    return sycl::buffer<T>(len);
}

template<class T>
void free_sycl_buf(sycl::buffer<T> &&buf, u64 ibuf) {

    register_del(buf.size() * sizeof(T), ibuf);
}

template<class T>
void shamalgs::ResizableBuffer<T>::alloc() {
    buf = std::make_unique<sycl::buffer<T>>(get_new_buf<T>(capacity, ibuf));
    logger::debug_alloc_ln("PatchDataField", "allocate field :", "len =", capacity);
}

template<class T>
void shamalgs::ResizableBuffer<T>::free() {

    if (buf) {
        logger::debug_alloc_ln("PatchDataField", "free field :", "len =", capacity);

        free_sycl_buf(shambase::extract_pointer(buf), ibuf);
    }
}

template<class T>
void shamalgs::ResizableBuffer<T>::change_capacity(u32 new_capa) {

    logger::debug_alloc_ln(
        "ResizableBuffer", "ibuf", ibuf, "change capacity from : ", capacity, "to :", new_capa);

    if (capacity == 0) {

        if (new_capa > 0) {
            capacity = new_capa;
            alloc();
        }

    } else {

        if (new_capa > 0) {

            if (new_capa != capacity) {
                capacity = new_capa;

                // print_state("tmp",buf->size() * sizeof(T), ibuf);
                auto old_buf = shambase::extract_pointer(buf);
                // print_state("tmp",old_buf.size() * sizeof(T), ibuf);

                alloc();

                if (val_cnt > 0) {
                    shamalgs::memory::copybuf_discard(
                        dev_sched->get_queue().q, old_buf, *buf, std::min(val_cnt, capacity));
                }

                logger::debug_alloc_ln("PatchDataField", "delete old buf : ");
                free_sycl_buf(std::move(old_buf), ibuf);
            }

        } else {
            capacity = 0;
            free();
        }
    }
}

template<class T>
void shamalgs::ResizableBuffer<T>::reserve(u32 add_size) {
    StackEntry stack_loc{false};

    // logger::info_ln("ResizableBuffer","ibuf",ibuf, "reserve from : ", val_cnt, "to :", val_cnt +
    // add_size); print_state("reserve",add_size,ibuf);

    u32 wanted_sz = val_cnt + add_size;

    if (wanted_sz > capacity) {
        change_capacity(wanted_sz * safe_fact);
    }
}

template<class T>
void shamalgs::ResizableBuffer<T>::resize(u32 new_size) {
    StackEntry stack_loc{false};

    logger::debug_alloc_ln("ResizableBuffer", "resize from : ", val_cnt, "to :", new_size);

    if (new_size > capacity) {
        change_capacity(new_size * safe_fact);
    }

    val_cnt = new_size;
}

template<class T>
shamalgs::ResizableBuffer<T>::ResizableBuffer(
    std::shared_ptr<sham::DeviceScheduler> _dev_sched, sycl::buffer<T> &&moved_buf, u32 val_cnt)
    : dev_sched(std::move(_dev_sched)),
      buf(std::make_unique<sycl::buffer<T>>(std::forward<sycl::buffer<T>>(moved_buf))),
      val_cnt(val_cnt), capacity(moved_buf.size()) {
    register_new(buf->size() * sizeof(T), ibuf);
}

////////////////////////////////////////////////////////////////////////////////////////////////
// value manipulation
////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void shamalgs::ResizableBuffer<T>::overwrite(ResizableBuffer<T> &f2, u32 cnt) {
    if (val_cnt < cnt) {
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "to overwrite you need more element in the field");
    }

    if (val_cnt > 0) {
        shamalgs::memory::copybuf_discard(dev_sched->get_queue().q, *f2.get_buf(), *buf, cnt);
    }
}

template<class T>
void shamalgs::ResizableBuffer<T>::override(sycl::buffer<T> &data, u32 cnt) {

    if (cnt != val_cnt)
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "buffer size doesn't match patchdata field size"); // TODO remove ref to size

    if (val_cnt > 0) {
        shamalgs::memory::copybuf_discard(dev_sched->get_queue().q, data, *buf, val_cnt);
    }
}

template<class T>
void shamalgs::ResizableBuffer<T>::override(std::vector<T> &data, u32 cnt) {

    if (cnt != val_cnt)
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "buffer size doesn't match patchdata field size"); // TODO remove ref to size

    if (data.size() < val_cnt) {
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "The input vector is too small");
    }

    if (val_cnt > 0) {

        {
            sycl::host_accessor acc_cur{*buf, sycl::write_only, sycl::no_init};

            for (u32 i = 0; i < val_cnt; i++) {
                // field_data[i] = acc[i];
                acc_cur[i] = data[i];
            }
        }
    }
}

template<class T>
void shamalgs::ResizableBuffer<T>::override(const T val) {

    if (val_cnt > 0) {

        shamalgs::memory::buf_fill_discard(
            dev_sched->get_queue().q, shambase::get_check_ref(buf), val);
    }
}

template<class T>
void shamalgs::ResizableBuffer<T>::index_remap_resize(
    sycl::buffer<u32> &index_map, u32 len, u32 nvar) {
    if (get_buf()) {

        auto get_new_buf = [&]() {
            if (nvar == 1) {
                return shamalgs::algorithm::index_remap(
                    dev_sched->get_queue().q, *get_buf(), index_map, len);
            } else {
                return shamalgs::algorithm::index_remap_nvar(
                    dev_sched->get_queue().q, *get_buf(), index_map, len, nvar);
            }
        };

        sycl::buffer<T> new_buf = get_new_buf();

        register_new(new_buf.size() * sizeof(T), ibuf);
        register_del(buf->size() * sizeof(T), ibuf);

        capacity = new_buf.size();
        val_cnt  = len * nvar;
        buf      = std::make_unique<sycl::buffer<T>>(std::move(new_buf));
    }
}

template<class T>
void shamalgs::ResizableBuffer<T>::serialize_buf(shamalgs::SerializeHelper &serializer) {
    if (buf) {
        serializer.write_buf(*buf, val_cnt);
    }
}

template<class T>
shamalgs::ResizableBuffer<T>
shamalgs::ResizableBuffer<T>::deserialize_buf(shamalgs::SerializeHelper &serializer, u32 val_cnt) {
    if (val_cnt == 0) {
        return ResizableBuffer(serializer.get_device_scheduler());
    } else {
        ResizableBuffer rbuf(serializer.get_device_scheduler(), val_cnt);
        serializer.load_buf(*(rbuf.buf), val_cnt);
        return std::move(rbuf);
    }
}

template<class T>
bool shamalgs::ResizableBuffer<T>::check_buf_match(const ResizableBuffer<T> &f2) const {
    bool match = true;

    match = match && (val_cnt == f2.val_cnt);

    {

        using buf_t = std::unique_ptr<sycl::buffer<T>>;

        const buf_t &buf    = get_buf();
        const buf_t &buf_f2 = f2.get_buf();

        sycl::buffer<u8> res_buf(val_cnt);

        dev_sched->get_queue().q.submit([&](sycl::handler &cgh) {
            sycl::accessor acc1{*buf, cgh, sycl::read_only};
            sycl::accessor acc2{*buf_f2, cgh, sycl::read_only};

            sycl::accessor acc_res{res_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{val_cnt}, [=](sycl::item<1> i) {
                acc_res[i] = sham::equals(acc1[i], acc2[i]);
            });
        });

        match = match && shamalgs::reduction::is_all_true(res_buf, f2.size());
    }

    return match;
}

template<class T>
shamalgs::SerializeSize shamalgs::ResizableBuffer<T>::serialize_buf_byte_size() {
    using H = shamalgs::SerializeHelper;
    return H::serialize_byte_size<T>(val_cnt);
}

template<class T>
shamalgs::ResizableBuffer<T> shamalgs::ResizableBuffer<T>::mock_buffer(
    std::shared_ptr<sham::DeviceScheduler> _dev_sched,
    u64 seed,
    u32 val_cnt,
    T min_bound,
    T max_bound) {
    sycl::buffer<T> buf_mocked = shamalgs::random::mock_buffer(seed, val_cnt, min_bound, max_bound);
    return ResizableBuffer<T>(_dev_sched, std::move(buf_mocked), val_cnt);
}

//////////////////////////////////////////////////////////////////////////
// Define the patchdata field for all classes in XMAC_LIST_ENABLED_ResizableUSMBuffer
//////////////////////////////////////////////////////////////////////////

#define X(a) template class shamalgs::ResizableBuffer<a>;
XMAC_LIST_ENABLED_ResizableUSMBuffer
#undef X

    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
