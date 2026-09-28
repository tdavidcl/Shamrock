// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file internal_alloc.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file contains the methods to actually allocate memory
 */

#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shambase/profiling/profiling.hpp"
#include "shambase/string.hpp"
#include "shambackends/details/internal_alloc.hpp"
#include "shambackends/details/memory_pool.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include <unordered_map>
#include <bit>
#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <vector>

namespace {

    sham::MemPerfInfos mem_perf_infos;

    void register_alloc_device(size_t size, f64 timed) {

        mem_perf_infos.allocated_byte_device += size;
        mem_perf_infos.time_alloc_device += timed;
        mem_perf_infos.max_allocated_byte_device = std::max(
            mem_perf_infos.max_allocated_byte_device, mem_perf_infos.allocated_byte_device);

        shambase::profiling::register_counter_val(
            "Device Memory", shambase::details::get_wtime(), mem_perf_infos.allocated_byte_device);
        shambase::profiling::register_counter_val(
            "Device alloc time", shambase::details::get_wtime(), mem_perf_infos.time_alloc_device);
    }

    void register_alloc_shared(size_t size, f64 timed) {

        mem_perf_infos.allocated_byte_shared += size;
        mem_perf_infos.time_alloc_shared += timed;
        mem_perf_infos.max_allocated_byte_shared = std::max(
            mem_perf_infos.max_allocated_byte_shared, mem_perf_infos.allocated_byte_shared);

        shambase::profiling::register_counter_val(

            "Shared Memory", shambase::details::get_wtime(), mem_perf_infos.allocated_byte_shared);
        shambase::profiling::register_counter_val(
            "Shared alloc time", shambase::details::get_wtime(), mem_perf_infos.time_alloc_shared);
    }

    void register_alloc_host(size_t size, f64 timed) {

        mem_perf_infos.allocated_byte_host += size;
        mem_perf_infos.time_alloc_host += timed;
        mem_perf_infos.max_allocated_byte_host
            = std::max(mem_perf_infos.max_allocated_byte_host, mem_perf_infos.allocated_byte_host);

        shambase::profiling::register_counter_val(
            "Host Memory", shambase::details::get_wtime(), mem_perf_infos.allocated_byte_host);
        shambase::profiling::register_counter_val(
            "Host alloc time", shambase::details::get_wtime(), mem_perf_infos.time_alloc_host);
    }

    void register_free_device(size_t size, f64 timed) {

        mem_perf_infos.allocated_byte_device -= size;
        mem_perf_infos.time_free_device += timed;

        shambase::profiling::register_counter_val(
            "Device Memory", shambase::details::get_wtime(), mem_perf_infos.allocated_byte_device);
        shambase::profiling::register_counter_val(
            "Device free time", shambase::details::get_wtime(), mem_perf_infos.time_free_device);
    }

    void register_free_shared(size_t size, f64 timed) {

        mem_perf_infos.allocated_byte_shared -= size;
        mem_perf_infos.time_free_shared += timed;

        shambase::profiling::register_counter_val(
            "Shared Memory", shambase::details::get_wtime(), mem_perf_infos.allocated_byte_shared);
        shambase::profiling::register_counter_val(
            "Shared free time", shambase::details::get_wtime(), mem_perf_infos.time_free_shared);
    }

    void register_free_host(size_t size, f64 timed) {

        mem_perf_infos.allocated_byte_host -= size;
        mem_perf_infos.time_free_host += timed;

        shambase::profiling::register_counter_val(
            "Host Memory", shambase::details::get_wtime(), mem_perf_infos.allocated_byte_host);
        shambase::profiling::register_counter_val(
            "Host free time", shambase::details::get_wtime(), mem_perf_infos.time_free_host);
    }

    template<sham::USMKindTarget target>
    std::string get_mode_name();

    template<>
    std::string get_mode_name<sham::device>() {
        return "device";
    }

    template<>
    std::string get_mode_name<sham::shared>() {
        return "shared";
    }

    template<>
    std::string get_mode_name<sham::host>() {
        return "host";
    }

    /**
     * @brief Pool of freed USM blocks kept for reuse by later allocations.
     *
     * Allocations are rounded up to a size class (at most 12.5% larger than requested), and
     * freed blocks are kept in the pool, keyed by (context, USM kind, size class, alignment),
     * instead of being returned to the SYCL runtime. A later allocation of the same class then
     * reuses the block, which avoids the cost of the device allocator and, on CPU backends, of
     * the page faults & zeroing of freshly mapped memory. The memory held by the pool is bounded
     * (see max_cached_fraction and max_total_fraction), and is released on allocation failure.
     *
     * With the LRU eviction (default), when the cached blocks exceed their limit the least
     * recently freed ones are released, so that the pool follows the current allocation pattern
     * instead of keeping blocks of sizes that are no longer used (e.g. from the setup phase).
     * Otherwise the newly freed block is released and the largest cached blocks are evicted
     * first.
     */
    struct UsmBlockPool {
        struct Key {
            const void *ctx_id;
            int target;
            size_t class_size;
            size_t alignment; // 0 if unspecified

            auto operator<=>(const Key &) const = default;
        };

        struct CachedBlock {
            void *ptr;
            std::shared_ptr<sham::DeviceScheduler> dev_sched; // keeps the context alive
            u64 stamp;                                        // order in which it was freed
        };

        std::mutex mtx;
        bool enabled = true;

        /// maximum fraction of the device memory that can be held by cached (free) blocks
        f64 max_cached_fraction = 0.6;
        /// maximum fraction of the device memory held by pooled blocks (live + cached) above
        /// which cached blocks are released before allocating new ones
        f64 max_total_fraction = 0.9;

        /// evict the least recently freed blocks first (otherwise the largest ones)
        bool lru_eviction = true;

        /// counter used to stamp the freed blocks
        u64 clock = 0;

        /// cached blocks by key, oldest first (the most recently freed is reused first)
        std::map<Key, std::deque<CachedBlock>> free_blocks;
        std::unordered_map<void *, Key> live_blocks;
        size_t cached_bytes = 0;
        size_t live_bytes   = 0;

        static size_t class_size(size_t sz, size_t alignment) {
            size_t gran = 256;
            if (sz > 4096) {
                gran = std::bit_floor(sz) / 8;
            }
            gran = std::max(gran, alignment); // both are powers of two
            return ((sz + gran - 1) / gran) * gran;
        }

        static size_t device_mem(const sham::DeviceScheduler &ds) {
            return ds.ctx->device->prop.global_mem_size;
        }

        /// release a cached block to the SYCL runtime (mutex must be held)
        void release(const Key &key, CachedBlock &blk) {
            sycl::free(blk.ptr, blk.dev_sched->ctx->ctx);
            cached_bytes -= key.class_size;
        }

        /// release cached blocks, largest first, until `bytes_needed` bytes were released or the
        /// pool is empty (mutex must be held)
        void evict(size_t bytes_needed) {
            size_t released = 0;
            while (released < bytes_needed && !free_blocks.empty()) {
                auto it     = std::prev(free_blocks.end());
                auto &stack = it->second;
                while (!stack.empty() && released < bytes_needed) {
                    release(it->first, stack.back());
                    released += it->first.class_size;
                    stack.pop_back();
                }
                if (stack.empty()) {
                    free_blocks.erase(it);
                }
            }
        }

        /// release the least recently freed cached blocks until `bytes_needed` bytes were
        /// released or the pool is empty (mutex must be held)
        void evict_lru(size_t bytes_needed) {
            size_t released = 0;
            while (released < bytes_needed && !free_blocks.empty()) {
                auto oldest = free_blocks.begin();
                for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
                    if (it->second.front().stamp < oldest->second.front().stamp) {
                        oldest = it;
                    }
                }
                release(oldest->first, oldest->second.front());
                released += oldest->first.class_size;
                oldest->second.pop_front();
                if (oldest->second.empty()) {
                    free_blocks.erase(oldest);
                }
            }
        }

        /// release cached blocks following the eviction policy (mutex must be held)
        void make_room(size_t bytes_needed) {
            if (lru_eviction) {
                evict_lru(bytes_needed);
            } else {
                evict(bytes_needed);
            }
        }

        void purge() {
            std::lock_guard<std::mutex> lock(mtx);
            evict(cached_bytes);
        }
    };

    UsmBlockPool &get_pool() {
        // intentionally leaked, blocks are released in finalize (see release_memory_pool)
        static UsmBlockPool *pool = new UsmBlockPool();
        return *pool;
    }

} // namespace

namespace sham::details {

    MemPerfInfos get_mem_perf_info() { return mem_perf_infos; }

    void reset_mem_info_max() {
        mem_perf_infos.max_allocated_byte_host   = mem_perf_infos.allocated_byte_host;
        mem_perf_infos.max_allocated_byte_device = mem_perf_infos.allocated_byte_device;
        mem_perf_infos.max_allocated_byte_shared = mem_perf_infos.allocated_byte_shared;
    }

    void set_memory_pool_enabled(bool enable) {
        UsmBlockPool &pool = get_pool();
        if (!enable) {
            pool.purge();
        }
        std::lock_guard<std::mutex> lock(pool.mtx);
        pool.enabled = enable;
    }

    bool is_memory_pool_enabled() {
        UsmBlockPool &pool = get_pool();
        std::lock_guard<std::mutex> lock(pool.mtx);
        return pool.enabled;
    }

    void release_memory_pool() { get_pool().purge(); }

    void set_memory_pool_lru_eviction(bool enable) {
        UsmBlockPool &pool = get_pool();
        std::lock_guard<std::mutex> lock(pool.mtx);
        pool.lru_eviction = enable;
    }

    void set_memory_pool_limits(f64 max_cached_fraction, f64 max_total_fraction) {
        UsmBlockPool &pool = get_pool();
        {
            std::lock_guard<std::mutex> lock(pool.mtx);
            pool.max_cached_fraction = max_cached_fraction;
            pool.max_total_fraction  = max_total_fraction;
        }
    }

    size_t get_memory_pool_cached_bytes() {
        UsmBlockPool &pool = get_pool();
        std::lock_guard<std::mutex> lock(pool.mtx);
        return pool.cached_bytes;
    }

    std::string log_mem_perf_info(const std::shared_ptr<DeviceScheduler> &dev_sched) {

        return sham::format(
            R"log(
    World infos :
        World size = {}
        World rank = {}
    Device infos :
        Device name = {}
    Allocs :
        max_allocated_byte_host = {}
        max_allocated_byte_device = {}
        max_allocated_byte_shared = {}
        allocated_byte_host = {}
        allocated_byte_device = {}
        allocated_byte_shared = {}
        )log",
            shamcomm::world_size(),
            shamcomm::world_rank(),
            dev_sched->ctx->device->dev.get_info<sycl::info::device::name>(),
            shambase::readable_sizeof(mem_perf_infos.max_allocated_byte_host),
            shambase::readable_sizeof(mem_perf_infos.max_allocated_byte_device),
            shambase::readable_sizeof(mem_perf_infos.max_allocated_byte_shared),
            shambase::readable_sizeof(mem_perf_infos.allocated_byte_host),
            shambase::readable_sizeof(mem_perf_infos.allocated_byte_device),
            shambase::readable_sizeof(mem_perf_infos.allocated_byte_shared));
    }

    template<USMKindTarget target>
    void internal_free(
        void *usm_ptr, size_t sz, const std::shared_ptr<DeviceScheduler> &dev_sched) {

        StackEntry __st{};

        f64 start_time = shambase::details::get_wtime();

        shamcomm::logs::debug_alloc_ln(
            "memoryHandle",
            "free usm pointer size :",
            sz,
            " | ptr =",
            usm_ptr,
            " | mode =",
            get_mode_name<target>());

        bool pooled = false;
        {
            UsmBlockPool &pool = get_pool();
            std::lock_guard<std::mutex> lock(pool.mtx);
            auto it = pool.live_blocks.find(usm_ptr);
            if (it != pool.live_blocks.end()) {
                UsmBlockPool::Key key = it->second;
                pool.live_blocks.erase(it);
                pool.live_bytes -= key.class_size;

                size_t max_cached
                    = size_t(pool.max_cached_fraction * f64(UsmBlockPool::device_mem(*dev_sched)));

                if (pool.enabled && pool.lru_eviction && key.class_size <= max_cached
                    && pool.cached_bytes + key.class_size > max_cached) {
                    pool.evict_lru(pool.cached_bytes + key.class_size - max_cached);
                }

                if (pool.enabled && pool.cached_bytes + key.class_size <= max_cached) {
                    pool.free_blocks[key].push_back({usm_ptr, dev_sched, pool.clock++});
                    pool.cached_bytes += key.class_size;
                    pooled = true;
                }
            }
        }

        if (!pooled) {
            sycl::context &sycl_ctx = dev_sched->ctx->ctx;
            sycl::free(usm_ptr, sycl_ctx);
        }

        f64 end_time = shambase::details::get_wtime();

        if constexpr (target == device) {
            register_free_device(sz, end_time - start_time);
        } else if constexpr (target == shared) {
            register_free_shared(sz, end_time - start_time);
        } else if constexpr (target == host) {
            register_free_host(sz, end_time - start_time);
        }
    }

    template<USMKindTarget target>
    void *internal_alloc(
        size_t sz,
        const std::shared_ptr<DeviceScheduler> &dev_sched,
        std::optional<size_t> alignment) {

        StackEntry __st{};
        f64 start_time = shambase::details::get_wtime();

        shamcomm::logs::debug_alloc_ln(
            "memoryHandle", "alloc usm pointer size :", sz, " | mode =", get_mode_name<target>());

        auto &ds                = shambase::get_check_ref(dev_sched);
        sycl::context &sycl_ctx = ds.ctx->ctx;
        sycl::device &dev       = ds.ctx->device->dev;

        void *usm_ptr = nullptr;

        // memory pool : try to reuse a cached block of the same class
        UsmBlockPool &pool = get_pool();
        std::optional<UsmBlockPool::Key> pool_key;
        {
            std::lock_guard<std::mutex> lock(pool.mtx);
            if (pool.enabled && sz > 0) {
                size_t align = (alignment) ? *alignment : 0;
                pool_key     = UsmBlockPool::Key{
                    ds.ctx.get(), int(target), UsmBlockPool::class_size(sz, align), align};

                auto it = pool.free_blocks.find(*pool_key);
                if (it != pool.free_blocks.end() && !it->second.empty()) {
                    usm_ptr = it->second.back().ptr;
                    it->second.pop_back();
                    if (it->second.empty()) {
                        pool.free_blocks.erase(it);
                    }
                    pool.cached_bytes -= pool_key->class_size;
                    pool.live_blocks[usm_ptr] = *pool_key;
                    pool.live_bytes += pool_key->class_size;
                } else {
                    // make room if the pool holds too much memory
                    size_t max_total
                        = size_t(pool.max_total_fraction * f64(UsmBlockPool::device_mem(ds)));
                    size_t total = pool.live_bytes + pool.cached_bytes + pool_key->class_size;
                    if (total > max_total) {
                        pool.make_room(total - max_total);
                    }
                }
            }
        }

        if (usm_ptr != nullptr) {
            f64 end_time = shambase::details::get_wtime();
            if constexpr (target == device) {
                register_alloc_device(sz, end_time - start_time);
            } else if constexpr (target == shared) {
                register_alloc_shared(sz, end_time - start_time);
            } else if constexpr (target == host) {
                register_alloc_host(sz, end_time - start_time);
            }
            return usm_ptr;
        }

        // size actually allocated (the size class if the block will be pooled)
        size_t alloc_sz = (pool_key) ? pool_key->class_size : sz;

        auto catch_alloc_except = [&](auto alloc_lambda) {
            try {
                usm_ptr = alloc_lambda();
                if (usm_ptr == nullptr && pool.cached_bytes > 0) {
                    pool.purge();
                    usm_ptr = alloc_lambda();
                }
            } catch (std::exception &ex) {
                std::string log = sham::format(
                    "Alloc failed with exception : {}\nShamrock mem infos : {}",
                    ex.what(),
                    log_mem_perf_info(dev_sched));
                shambase::throw_with_loc<std::runtime_error>(log);
            }
        };

        // check max alloc sizes
        if constexpr (target == device) {
            if (sz > ds.get_queue().get_device_prop().max_mem_alloc_size_dev) {
                std::string err_log = sham::format(
                    "You are trying to allocate more than the maximum allocation size allowed by "
                    "the "
                    "device\n"
                    "  size = {} | max_alloc_size = {}",
                    sz,
                    ds.get_queue().get_device_prop().max_mem_alloc_size_dev);
                shambase::throw_with_loc<std::runtime_error>(err_log);
            }
        } else if constexpr (target == shared) {
            size_t max_alloc_size_dev  = ds.get_queue().get_device_prop().max_mem_alloc_size_dev;
            size_t max_alloc_size_host = ds.get_queue().get_device_prop().max_mem_alloc_size_host;
            if (sz > sycl::min(max_alloc_size_dev, max_alloc_size_host)) {
                std::string err_log = sham::format(
                    "You are trying to allocate more than the maximum allocation size allowed by "
                    "the "
                    "device\n"
                    "  size = {} | max_alloc_size = {}",
                    sz,
                    sycl::min(max_alloc_size_dev, max_alloc_size_host));
                shambase::throw_with_loc<std::runtime_error>(err_log);
            }
        } else if constexpr (target == host) {
            if (sz > ds.get_queue().get_device_prop().max_mem_alloc_size_host) {
                std::string err_log = sham::format(
                    "You are trying to allocate more than the maximum allocation size allowed by "
                    "the "
                    "host\n"
                    "  size = {} | max_alloc_size = {}",
                    sz,
                    ds.get_queue().get_device_prop().max_mem_alloc_size_host);
                shambase::throw_with_loc<std::runtime_error>(err_log);
            }
        } else {
            shambase::throw_unimplemented();
        }

        if (alignment) {

            if (*alignment % ds.get_queue().get_device_prop().mem_base_addr_align != 0) {
                shambase::throw_with_loc<std::runtime_error>(sham::format(
                    "The alignment of the USM pointer is not aligned with minimum device "
                    "alignment\n"
                    "  alignment = {} | device alignment = {} | alignment % device alignment = {}",
                    *alignment,
                    ds.get_queue().get_device_prop().mem_base_addr_align,
                    *alignment % ds.get_queue().get_device_prop().mem_base_addr_align));
            }

            if (sz % *alignment != 0) {
                shambase::throw_with_loc<std::runtime_error>(sham::format(
                    "The size of the USM pointer is not aligned with the given alignment\n"
                    "  size = {} | alignment = {} | size % alignment = {}",
                    sz,
                    *alignment,
                    sz % *alignment));
            }

            // TODO upgrade alignment to 256-bit for CUDA ?

            if constexpr (target == device) {
                catch_alloc_except([&] {
                    return sycl::aligned_alloc_device(*alignment, alloc_sz, dev, sycl_ctx);
                });
            } else if constexpr (target == shared) {
                catch_alloc_except([&] {
                    return sycl::aligned_alloc_shared(*alignment, alloc_sz, dev, sycl_ctx);
                });
            } else if constexpr (target == host) {
                catch_alloc_except([&] {
                    return sycl::aligned_alloc_host(*alignment, alloc_sz, sycl_ctx);
                });
            } else {
                shambase::throw_unimplemented();
            }
        } else {
            if constexpr (target == device) {
                catch_alloc_except([&] {
                    return sycl::malloc_device(alloc_sz, dev, sycl_ctx);
                });
            } else if constexpr (target == shared) {
                catch_alloc_except([&] {
                    return sycl::malloc_shared(alloc_sz, dev, sycl_ctx);
                });
            } else if constexpr (target == host) {
                catch_alloc_except([&] {
                    return sycl::malloc_host(alloc_sz, sycl_ctx);
                });
            } else {
                shambase::throw_unimplemented();
            }
        }

        if (usm_ptr == nullptr) {
            std::string err_msg = "";
            if (alignment) {
                err_msg = sham::format(
                    "USM allocation failed, details : sz={}, target={}, alignment={}, alloc "
                    "result = {}",
                    sz,
                    get_mode_name<target>(),
                    *alignment,
                    usm_ptr);
            } else {
                err_msg = sham::format(
                    "USM allocation failed, details : sz={}, target={}, alloc result = {}",
                    sz,
                    get_mode_name<target>(),
                    usm_ptr);
            }
            shambase::throw_with_loc<std::runtime_error>(err_msg + log_mem_perf_info(dev_sched));
        }

        if (alignment) {

            shamcomm::logs::debug_alloc_ln(
                "memoryHandle", "pointer created : ptr =", usm_ptr, "alignment =", *alignment);

            if (!shambase::is_aligned(usm_ptr, *alignment)) {
                shambase::throw_with_loc<std::runtime_error>(
                    "The pointer is not aligned with the given alignment");
            }

        } else {

            shamcomm::logs::debug_alloc_ln(
                "memoryHandle", "pointer created : ptr =", usm_ptr, "alignment = None");
        }

        if (pool_key) {
            std::lock_guard<std::mutex> lock(pool.mtx);
            pool.live_blocks[usm_ptr] = *pool_key;
            pool.live_bytes += pool_key->class_size;
        }

        f64 end_time = shambase::details::get_wtime();

        if constexpr (target == device) {
            register_alloc_device(sz, end_time - start_time);
        } else if constexpr (target == shared) {
            register_alloc_shared(sz, end_time - start_time);
        } else if constexpr (target == host) {
            register_alloc_host(sz, end_time - start_time);
        }

        return usm_ptr;
    }

#ifndef DOXYGEN
    template void internal_free<host>(
        void *usm_ptr, size_t sz, const std::shared_ptr<DeviceScheduler> &dev_sched);
    template void *internal_alloc<host>(
        size_t sz,
        const std::shared_ptr<DeviceScheduler> &dev_sched,
        std::optional<size_t> alignment);
    template void internal_free<device>(
        void *usm_ptr, size_t sz, const std::shared_ptr<DeviceScheduler> &dev_sched);
    template void *internal_alloc<device>(
        size_t sz,
        const std::shared_ptr<DeviceScheduler> &dev_sched,
        std::optional<size_t> alignment);
    template void internal_free<shared>(
        void *usm_ptr, size_t sz, const std::shared_ptr<DeviceScheduler> &dev_sched);
    template void *internal_alloc<shared>(
        size_t sz,
        const std::shared_ptr<DeviceScheduler> &dev_sched,
        std::optional<size_t> alignment);
#endif

} // namespace sham::details
