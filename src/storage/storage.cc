/*!
 * Copyright (c) 2015 by Contributors
 */
#include <cuda_profiler_api.h>
#include <mxnet/storage.h>
#include <mshadow/tensor.h>
#include <dmlc/logging.h>
#include <array>
#include "./storage_manager.h"
#include "./naive_storage_manager.h"
#include "./pooled_storage_manager.h"
#include "./cpu_device_storage.h"
#include "./gpu_device_storage.h"
#include "./pinned_memory_storage.h"
#include "../common/cuda_utils.h"
#include "../common/lazy_alloc_array.h"

namespace mxnet {

// consider change storage as a pure abstract class
class StorageImpl : public Storage {
 public:
  Handle Alloc(size_t size, Context ctx, bool direct = false) override;
  void Free(Handle handle) override;
  void DirectFree(Handle handle) override;
  StorageImpl() {}
  virtual ~StorageImpl() = default;

 private:
  static constexpr size_t kMaxNumberOfDevices = Context::kMaxDevType + 1;
  static constexpr size_t kMaxNumberOfDeviceIDs = Context::kMaxDevID + 1;

  static void ActivateDevice(Context ctx) {
    switch (ctx.dev_type) {
      case Context::kCPU: break;
      case Context::kGPU:
      case Context::kCPUPinned:
#if MXNET_USE_CUDA
        CUDA_CALL(cudaSetDevice(ctx.dev_id));
#else  // MXNET_USE_CUDA
        LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_CUDA
        break;
      default:
        LOG(FATAL) << "Unimplemented device";
    }
  }
  // internal storage managers
  std::array<common::LazyAllocArray<storage::StorageManager>,
             kMaxNumberOfDevices> storage_managers_;
};  // struct Storage::Impl

Storage::Handle StorageImpl::Alloc(size_t size, Context ctx, bool direct) {
  // space already recycled, ignore request
  Handle hd;
  hd.ctx = ctx;
  hd.size = size;
  auto&& device = storage_managers_.at(ctx.dev_type);
  storage::StorageManager *manager = device.Get(
      ctx.dev_id, [ctx]() {
        storage::StorageManager *ptr = nullptr;
        switch (ctx.dev_type) {
          case Context::kCPU: {
            ptr = new storage::NaiveStorageManager<storage::CPUDeviceStorage>();
            break;
          }
          case Context::kCPUPinned: {
#if MXNET_USE_CUDA
            ptr = new storage::NaiveStorageManager<storage::PinnedMemoryStorage>();
#else
            LOG(FATAL) << "Compile with USE_CUDA=1 to enable GPU usage";
#endif  // MXNET_USE_CUDA
            break;
          }
          case Context::kGPU: {
#if MXNET_USE_CUDA
            ptr = new storage::GPUPooledStorageManager();
#else
            LOG(FATAL) << "Compile with USE_CUDA=1 to enable GPU usage";
#endif  // MXNET_USE_CUDA
            break;
          }
          default: LOG(FATAL) <<  "Unimplemented device " << ctx.dev_type;
        }
        return ptr;
      });
  this->ActivateDevice(ctx);
  int dev_id = ctx.dev_id;
  if (ctx.dev_type == Context::kCPU) {
      dev_id = -1;
  }
  if (direct) {
    hd.SetDptr(manager->DirectAlloc(size), dev_id);
  } else {
    hd.SetDptr(manager->Alloc(size), dev_id);
  }
  return hd;
}

void StorageImpl::Free(Storage::Handle handle) {
  const Context &ctx = handle.ctx;
  auto&& device = storage_managers_.at(ctx.dev_type);
  storage::StorageManager *manager = device.Get(
      ctx.dev_id, []() {
        LOG(FATAL) <<  "Cannot Free space to a device you have not allocated";
        return nullptr;
      });
  this->ActivateDevice(ctx);
  manager->Free(handle.GetDptr(), handle.size);
  handle.Free(true);
}

void StorageImpl::DirectFree(Storage::Handle handle) {
  const Context &ctx = handle.ctx;
  auto&& device = storage_managers_.at(ctx.dev_type);
  storage::StorageManager *manager = device.Get(
      ctx.dev_id, []() {
        LOG(FATAL) <<  "Cannot Free space to a device you have not allocated";
        return nullptr;
      });
  this->ActivateDevice(ctx);
  // directly free ths data.
  manager->DirectFree(handle.GetDptr(), handle.size);
  handle.Free(false);
}

std::shared_ptr<Storage> Storage::_GetSharedRef() {
#ifdef __MXNET_JS__
  // dummy code needed for emscripten code to pass
  // do not know why, the new will be NULLPTR
  static int *q = new int();
#endif
  static std::shared_ptr<Storage> inst(new StorageImpl());
  return inst;
}

Storage* Storage::Get() {
  static Storage *ptr = _GetSharedRef().get();
  return ptr;
}

/**
 *
 * Swap, Cache and MemHistory Implemenetation.
 *
 */

std::atomic<handle_id_t> Storage::Handle::base_id_(0);

std::shared_ptr<MemHistory> MemHistory::_GetSharedRef() {
    static std::shared_ptr<MemHistory> inst(new MemHistory());
    return inst;
}

MemHistory* MemHistory::Get() {
    static MemHistory *s = _GetSharedRef().get();
    return s;
}

MemHistory::MemHistory() {
    iteration_started_ = false;
    iteration_idx_ = 1;
    do_record_ = false;
    num_device_ = 1;
}

MemHistory::~MemHistory() {
    std::cout << "Destroy MemHistory" << std::endl;
}

void MemHistory::PutRecord(handle_id_t handle_id, int device, record_t type,
                           size_t size) {
    if (!iteration_started_) {
        return;
    }
    if (do_record_) {
        //std::cout << "His " << handle_id << " " << size << std::endl;
        std::lock_guard<std::mutex> lock(mutex_[device]);
        if (device >= num_device_) {
            num_device_ = device + 1;
        }
        timestamp_t t =
            (duration_cast<microseconds>(high_resolution_clock::now() -
                                         begin_time_)).count();
        MemRecord r = {handle_id, type, t, size};
        history[device].push_back(r);
        //std::cout << "his : "
                  //<< history[curr_idx[device]].handle_id << " "
                  //<< history[curr_idx[device]].type << " "
                  //<< history[curr_idx[device]].size << " "
                  //<< history[curr_idx[device]].time << " "
                  //<< history.size() << std::endl;
                  //
    } else {
        //std::cout << "Now " << handle_id << " " << size << std::endl;
        //if (history.size() > 0) {
            //std::cout << "his : "
                      //<< history[curr_idx[device]].handle_id << " "
                      //<< history[curr_idx[device]].type << " "
                      //<< history[curr_idx[device]].size << " "
                      //<< history[curr_idx[device]].time << " "
                      //<< history.size() << std::endl;
            //std::cout << "now : "
                      //<< id << " " << type << " " << size
                      //<< " " << curr_idx[device] << std::endl;
        //}
    }
    curr_idx[device] += 1;
}

void MemHistory::StartIteration() {
    unsigned start_iteration = dmlc::GetEnv("MXNET_MEM_RECORD_ITERATION", 4);
    iteration_started_ = true;
    for (int i = 0; i < 8; i++) {
        curr_idx[i] = -1;
    }
    do_record_ = (iteration_idx_ == start_iteration);
    begin_time_ = high_resolution_clock::now();
}

void MemHistory::MakeRecordSet(std::unordered_map<handle_id_t, MemRecord>& members,
                               MemRecordSet *record_set,
                               int device) {
    unsigned long time = members.begin()->second.time;
    for (auto m : members) {
        record_set->unique_records.push_back(
            MemRecord{m.second.handle_id, m.second.type,
                      time, m.second.size});
    }
    std::sort(record_set->unique_records.begin(),
              record_set->unique_records.end(),
              [](const MemRecord& a, const MemRecord& b) -> bool {
                return a.size > b.size;
              });
    set_history[device].push_back(record_set);
    if (set_accu_nrecords[device].size() == 0) {
        set_accu_nrecords[device].push_back(record_set->all_records.size());
    } else {
        set_accu_nrecords[device].push_back(
            set_accu_nrecords[device][set_accu_nrecords[device].size() - 1] +
            record_set->all_records.size());
    }
}

void MemHistory::Analyze() {
    unsigned threshold = dmlc::GetEnv("MXNET_SET_MEM_ORDER_THRESHOLD", 500);
    for (int device = 0; device < num_device_; device++) {
        std::unordered_map<handle_id_t, MemRecord> members;
        MemRecordSet *record_set = new MemRecordSet();
        unsigned previous_time = 0;
        for (auto& h : history[device]) {
            access_stats[h.handle_id] += 1;
            if (h.time - previous_time > threshold && members.size() > 0) {
                MakeRecordSet(members, record_set, device);
                record_set = new MemRecordSet();
                members.clear();
            }
            record_set->all_records.push_back(h);
            if (members.find(h.handle_id) == members.end()) {
                members[h.handle_id] = h;
            }
            previous_time = h.time;
        }
        if (members.size() > 0) {
            MakeRecordSet(members, record_set, device);
        }
    }
#if 0
    std::cout << "Set history:" << std::endl;
    size_t all = 0;
    size_t idx = 0;
    for (auto set : set_history[0]) {
        all += set->all_records.size();
        std::cout << "Count: " << all << std::endl;
        std::cout << "Index: " << idx++ << std::endl;
        for (auto& h : set->unique_records) {
            std::cout << h.handle_id << " " << h.size << std::endl;
        }
        std::cout << "----------" << std::endl;
    }

#endif
#if 0
    std::unordered_set<handle_id_t> handles;
    size_t size = 0;
    unsigned long previous = 0;
    std::cout << "differences : " << std::endl;
    for (auto& h : history) {
        auto it = handles.find(h.handle_id);
        std::cout << h.time - previous << std::endl;
        previous = h.time;
        if (it == handles.end()) {
            size += h.size;
            handles.insert(h.handle_id);
        }
    }
    std::cout << "Size = " << size * 1.0 / 1024 / 1024 << std::endl;
    size_t free, total;
    CUDA_CALL(cudaMemGetInfo(&free, &total));
    std::cout << "Total = " << total * 1.0 / 1024 / 1024 << std::endl;
#endif
}

void MemHistory::StopIteration() {
    iteration_started_ = false;
    iteration_idx_ += 1;
    if (do_record_) {
        Analyze();
    }
    do_record_ = false;
}

std::shared_ptr<Cache> Cache::_GetSharedRef() {
    static std::shared_ptr<Cache> inst(new Cache());
    return inst;
}

Cache* Cache::Get() {
    static Cache *c = _GetSharedRef().get();
    return c;
}

Cache::Cache() {
    std::cout << "Initialize Cache" << std::endl;
    mhistory_ = MemHistory::_GetSharedRef();
    enabled_ = dmlc::GetEnv("MXNET_DO_CACHE", 0);
    for (int i = 0; i < 8; i++) {
        locks_[i] = PTHREAD_RWLOCK_INITIALIZER;
    }
}

Cache::~Cache() {
    std::cout << "Destroy Cache" << std::endl;
}

void Cache::Register(int tensor_id, TShape offset, void* output_dptr) {
    if (mhistory_->IsRecording()) {
        int device;
        CUDA_CALL(cudaGetDevice(&device));
        handle_id_t handle_id = dptr_to_handle_[device].at(output_dptr);
        // Check if the cache_item exists.
        unsigned long long key = tensor_id;
        std::hash<index_t> dim_hash;
        for (auto dim : offset) {
            key ^= dim_hash(dim) + 0x9e3779b9 + (key << 6) +
                   (key >> 2);
        }
        auto cache_it = key_to_cache_[device].find(key);
        CacheItem *cache_item;
        if (cache_it == key_to_cache_[device].end()) {
            cache_item = new CacheItem{nullptr, false, 0};
        } else {
            cache_item = cache_it->second;
        }
        cache_item->all_handles.insert(handle_id);
    }
}

bool Cache::Cached(void* dptr) {
    int device;
    CUDA_CALL(cudaGetDevice(&device));
    pthread_rwlock_wrlock(&locks_[device]);
    handle_id_t handle_id = dptr_to_handle_[device].at(dptr);
    pthread_rwlock_unlock(&locks_[device]);
    auto it = cache_[device].find(handle_id);
    return (it != cache_[device].end() && it->second->cached);
}

void Cache::Release(void* dptr)
{
    if (!enabled_) {return;}

    int device;
    CUDA_CALL(cudaGetDevice(&device));
    pthread_rwlock_wrlock(&locks_[device]);
    auto handle_it = dptr_to_handle_[device].find(dptr);
    if (handle_it == dptr_to_handle_[device].end()) {
        pthread_rwlock_unlock(&locks_[device]);
        return;
    }
    auto cache_it = cache_[device].find(handle_it->second);
    auto temporary_it = temporary_items_[device].find(handle_it->second);
    if (cache_it != cache_[device].end()) {
        if (!cache_it->second->cached) {
            CUDA_CALL(cudaFree(cache_it->second->dptr));
        }
    } else if (temporary_it != temporary_items_[device].end()) {
        CUDA_CALL(cudaFree(temporary_it->second->dptr));
        temporary_it->second->dptr = nullptr;
        temporary_it->second->swap_in = false;
    } else {
        // Do nothing for all other entries
    }
    dptr_to_handle_[device].erase(dptr);
    pthread_rwlock_unlock(&locks_[device]);
}

void Cache::SetAddr(SwapInfo *info, bool record) {
    pthread_rwlock_rdlock(&locks_[info->device]);
    auto temporary_it = temporary_items_[info->device].find(info->handle_id);
    if (temporary_it != temporary_items_[info->device].end()) {
        CUDA_CALL(cudaFree(temporary_it->second->dptr));
        temporary_it->second->dptr = nullptr;
        temporary_it->second->swap_in = false;
    }
    pthread_rwlock_unlock(&locks_[info->device]);
}

bool Cache::CacheIt(SwapInfo *info) {
    return true;
}

void Cache::GetAddr(SwapInfo *info, bool record) {
    // FIXME(fegin):
    // 1. If the matrix is not temporary nor cached, simply return the address.
    // 2. If the matrix is temporary, allocate it.
    // 3. If the matrix is cached, simply return the address. (done)
    // 4. If the matrix is a cache candidate, decide whether to cache it.
    pthread_rwlock_wrlock(&locks_[info->device]);
    auto cache_it = cache_[info->device].find(info->handle_id);
    if (cache_it != cache_[info->device].end()) {
        if (cache_it->second->cached) {
            CHECK_EQ(info->size, cache_it->second->size);
            info->dptr = cache_it->second->dptr;
        } else {
            cudaError_t e = cudaMalloc(&(info->dptr), info->size);
            if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
                LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
            }
            info->swap_in = true;
            cache_it->second->size = info->size;
            cache_it->second->dptr = info->dptr;
        }
        CacheIt(info);
    } else if (!info->swap_in) { // Temporary.
        cudaError_t e = cudaMalloc(&(info->dptr), info->size);
        if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
            LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
        }
        info->swap_in = true;
    } else {
        // Do nothing for all other data entries.
        return;
    }
    dptr_to_handle_[info->device][info->dptr] = info->handle_id;
    pthread_rwlock_unlock(&locks_[info->device]);
}

void Cache::DelAddr(SwapInfo *info, bool record) {
    // FIXME(fegin): Do we need to do anything with DelAddr?
    return;
}

std::shared_ptr<Swap> Swap::_GetSharedRef() {
    static std::shared_ptr<Swap> inst(new Swap());
    return inst;
}

Swap* Swap::Get() {
    static Swap *s = _GetSharedRef().get();
    return s;
}

Swap::Swap() {
    std::cout << "Initialize Swap" << std::endl;
    mhistory_ = MemHistory::_GetSharedRef();
    cache_ = Cache::_GetSharedRef();
    do_swap_ = dmlc::GetEnv("MXNET_DO_SWAP", 0);
    do_cache_ = dmlc::GetEnv("MXNET_DO_CACHE", 0);
    CHECK(!(do_swap_ && do_cache_));
    look_ahead_ = dmlc::GetEnv("MXNET_SWAPPER_LOOK_AHEAD", 100);
    free_cpu_ = dmlc::GetEnv("MXNET_FREE_CPU_MEMORY", false);
    unsigned multiplier = dmlc::GetEnv("MXNET_SWAP_THRESHOLD_MULTIPLIER", 32);
    std::cout << "MXNET_DO_SWAP = " << do_swap_ << std::endl;
    std::cout << "MXNET_DO_CACHE = " << do_cache_ << std::endl;
    std::cout << "MXNET_SWAPPER_LOOK_AHEAD = " << look_ahead_<< std::endl;
    std::cout << "MXNET_FREE_CPU_MEMORY = " << free_cpu_ << std::endl;
    std::cout << "MXNET_SWAP_THRESHOLD_MULTIPLIER= " << multiplier << std::endl;
    swap_lock_ = PTHREAD_RWLOCK_INITIALIZER;
    swapper_began_ = false;
    lru_ = std::vector<std::list<SwapInfo*>>(8);
    reserved_mem_ = std::vector<std::unordered_map<void*, size_t>>(8);
    for (int i = 0; i < 8; i++) {
        locks_[i] = PTHREAD_RWLOCK_INITIALIZER;
        streams_init_[i] = false;
        free_memory_.push_back(0);
    }
    num_device_ = 1;
    size_t fifo_size, heap_size;
    cudaDeviceGetLimit(&fifo_size, cudaLimitPrintfFifoSize);
    cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
    swap_threshold_ = (fifo_size + heap_size + 1024 * 1024) * multiplier;
};

Swap::~Swap() {
    std::cout << "Destroy Swap" << std::endl;
}

int Swap::UpdateFree(int device) {
    size_t free = 10000000000, total = 0;
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMemGetInfo(&free, &total));
    free_memory_[device] = free;
    //std::cout << "free_memory_ " << free_memory_[device] << std::endl;
    return device;
}

bool Swap::FreeReserved(void *ptr, size_t size) {
    int device = -1;
    CUDA_CALL(cudaGetDevice(&device));
    if (device < 0) {
        for (device = 0; device < mhistory_->GetNumDevice(); device++) {
            if (reserved_mem_[device].find(ptr) !=
                    reserved_mem_[device].end()) {
                break;
            }
        }
        if (device == mhistory_->GetNumDevice()) {
            std::cout << "OOOps" << std::endl;
            return false;
        }
    }
    //std::cout << "device " << device << std::endl;
    pthread_rwlock_wrlock(&locks_[device]);
    auto it = reserved_mem_[device].find(ptr);
    bool ret = true;
    if (it != reserved_mem_[device].end()) {
        ret = (it->second & (1L << 63));
        CHECK_EQ(it->second & (~(1L << 63)), size);
        reserved_mem_[device].erase(it);
    } else {
        ret = false;
    }
    pthread_rwlock_unlock(&locks_[device]);
    return ret;
}

bool Swap::CheckReservedAndFree(void *ptr, size_t size) {
    int device;
    CUDA_CALL(cudaGetDevice(&device));
    pthread_rwlock_wrlock(&locks_[device]);
    auto it = reserved_mem_[device].find(ptr);
    CHECK_EQ(it->second & (~(1L << 63)), size);
    if (!(it->second & (1L << 63))) {
        reserved_mem_[device].erase(it);
        pthread_rwlock_unlock(&locks_[device]);
        return true;
    } else {
        pthread_rwlock_unlock(&locks_[device]);
        return false;
    }
}

void Swap::AllocateReserved(size_t required, int device) {
    std::cout << "Unfortunately, we need to free reserved memory."
              << "This implementation is very tricky and dangerous "
              << "since we don't have a persistent id. The correctness "
              << "is not guaranteed."
              << std::endl;

    auto it = reserved_mem_[device].begin();
    while (required > 0) {
        while (!(it->second & (1L << 63))) {
            it++;
        }
        CHECK(reserved_mem_[device].find(it->first) !=
              reserved_mem_[device].end());
        CHECK(it != reserved_mem_[device].end());
        CUDA_CALL(cudaFree(it->first));
        it->second &= (~(1L << 63));
        if (it->second > required) {
            required = 0;
        } else {
            required -= it->second;
        }
    }
}

void Swap::DoSwap(SwapInfo* info, bool swap_out, bool async) {
    if (!streams_init_[info->device]) {
        streams_init_[info->device] = true;
        cudaStreamCreate(&streams_[info->device]);
    }
    if (swap_out) {
        if (info->cpu_address == nullptr) {
            info->cpu_address = (char*)malloc(info->size);
        }
        if (access_stats_.size() == 0 ||
                (access_stats_[info->handle_id] <
                    mhistory_->access_stats[info->handle_id])) {
            //std::cout << "Swap out " << info->handle_id << " " << info->size << std::endl;
            if (async) {
                CUDA_CALL(cudaMemcpyAsync(info->cpu_address, info->dptr,
                                          info->size,
                                          cudaMemcpyDeviceToHost,
                                          streams_[info->device]));
                CUDA_CALL(cudaStreamSynchronize(streams_[info->device]));
            } else {
                //std::cout << " info->cpu_address " << (size_t)(info->cpu_address)
                          //<< " info->dptr " << (size_t)(info->dptr) << std::endl;
                CUDA_CALL(cudaMemcpy(info->cpu_address, info->dptr,
                                     info->size, cudaMemcpyDeviceToHost));
            }
        }
        CUDA_CALL(cudaFree(info->dptr));
        info->dptr = nullptr;
    } else {
        cudaError_t e = cudaMalloc(&(info->dptr), info->size);
        if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
            LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
        }
        CHECK(info->cpu_address != nullptr);
        if (access_stats_.size() == 0 || access_stats_[info->handle_id] > 1) {
            if (async) {
                CUDA_CALL(cudaMemcpyAsync(info->dptr, info->cpu_address,
                                          info->size, cudaMemcpyHostToDevice,
                                          streams_[info->device]));
                CUDA_CALL(cudaStreamSynchronize(streams_[info->device]));
            } else {
                CUDA_CALL(cudaMemcpy(info->dptr, info->cpu_address, info->size,
                                     cudaMemcpyHostToDevice));
            }
        }
        if (free_cpu_) {
            free(info->cpu_address);
            info->cpu_address = nullptr;
        }
    }
}

void Swap::SwapOut(unsigned required_memory, int device,
                   bool acquire_lock=false, bool async=false) {
    if (!do_swap_) {
        return;
    }
    if (device == -1) {
        CUDA_CALL(cudaGetDevice(&device));
    }
    if (acquire_lock) {
        pthread_rwlock_wrlock(&locks_[device]);
    }
    UpdateFree(device);
    if (free_memory_[device] > required_memory + swap_threshold_) {
        if (acquire_lock) {
            pthread_rwlock_unlock(&locks_[device]);
        }
        return;
    }
    //std::cout << "Required " << required_memory << " " << swap_threshold_ << std::endl;
    unsigned ignored_lru = 0;
    CUDA_CALL(cudaSetDevice(device));
    while (free_memory_[device] < required_memory + swap_threshold_) {
        if (lru_[device].size() - ignored_lru > 0) {
            auto target = lru_[device].back();
            lru_[device].pop_back();
            if (target->dptr == nullptr || !target->swap_in) {
                std::cout << "(info->dptr == nullptr || !target->swap_in)."
                          << "This should happen when all iterations are done."
                          << "Current lru size is " << lru_[device].size()
                          << std::endl;
                ignored_lru += 1;
                lru_[device].push_front(target);
                continue;
            }
            CHECK(!target->is_swapping.test_and_set(std::memory_order_acquire));
            target->swap_in = false;
            target->it = lru_[device].end();
            pthread_rwlock_unlock(&locks_[device]);
            DoSwap(target, true, async);
            pthread_rwlock_wrlock(&locks_[device]);
            target->is_swapping.clear(std::memory_order_release);
        } else {
            size_t required = required_memory + swap_threshold_ -
                              free_memory_[device];
            AllocateReserved(required, device);
        }
        UpdateFree(device);
    }
    if (acquire_lock) {
        pthread_rwlock_unlock(&locks_[device]);
    }
}

void Swap::SwapIn(SwapInfo *info, bool async=false) {
    //std::cout << "SwapIn " << info->handle_id << std::endl;
    bool waiting = false;
    while (info->is_swapping.test_and_set(std::memory_order_acquire)) {
        waiting = true;
        pthread_rwlock_unlock(&locks_[info->device]);
        usleep(50);
        pthread_rwlock_wrlock(&locks_[info->device]);
    }
    if (waiting) {
        waiting_swapping_ += 1;
    }
    if (!info->swap_in) {
        CHECK(info->dptr == nullptr);
        CHECK(info->cpu_address != nullptr);
        int old_device = 0;
        CUDA_CALL(cudaGetDevice(&old_device));
        SwapOut(info->size, info->device, false, async);
        CHECK(free_memory_[info->device] > info->size);
        pthread_rwlock_unlock(&locks_[info->device]);
        DoSwap(info, false, async);
        pthread_rwlock_wrlock(&locks_[info->device]);
        CUDA_CALL(cudaSetDevice(old_device));
        info->swap_count += 1;
        info->swap_in = true;
    }
    info->is_swapping.clear(std::memory_order_release);
}

void Swap::SetAddr_Swap(SwapInfo *info, bool record) {
    lru_[info->device].push_front(info);
    info->it = lru_[info->device].begin();
}

void Swap::SetAddr_Cache(SwapInfo *info, bool record) {
    cache_->SetAddr(info, record);
}

void Swap::SetAddr(handle_id_t handle_id, void* dptr, size_t size, int dev_id,
                   bool record) {
    //std::cout << "SetAddr " << handle_id << " " << dptr << std::endl;
    if (dev_id != -1 && record) {
        mhistory_->PutRecord(handle_id, dev_id, MemHistory::SET_ADDR, size);
    }
    if (dptr == nullptr) {
        return;
    }
    pthread_rwlock_wrlock(&swap_lock_);
    auto iter = swap_info_.find(handle_id);
    if (iter == swap_info_.end()) {
        CHECK(dptr != nullptr);
        SwapInfo* info = new SwapInfo{handle_id, true, ATOMIC_FLAG_INIT, 0,
                                           dev_id, dptr, nullptr, size};
        swap_info_[handle_id] = info;
        if (dev_id != -1) {
            pthread_rwlock_wrlock(&locks_[dev_id]);
            if (do_cache_) {
                SetAddr_Cache(info, record);
            } else {
                SetAddr_Swap(info, record);
            }
            pthread_rwlock_unlock(&locks_[dev_id]);
        } else {
            info->it = lru_[0].end();
        }
    } else {
        std::cout << "SetAddr duplicated id " << handle_id << std::endl;
        std::cout << "SetAddr " << iter->second->size << " " << size << std::endl;
        CHECK(iter->second->swap_in);
        CHECK_EQ(iter->second->handle_id, handle_id);
        CHECK_EQ(iter->second->dptr, dptr);
        CHECK_EQ(iter->second->size, size);
    }
    pthread_rwlock_unlock(&swap_lock_);
};

void Swap::DelAddr_Swap(SwapInfo *info, bool preserve, bool record) {
    if (info->cpu_address != nullptr) {
        free(info->cpu_address);
        info->cpu_address = nullptr;
    }
    auto& reserved_mem = reserved_mem_[info->device];
    if (info->swap_in) {
        lru_[info->device].erase(info->it);
        info->it = lru_[info->device].end();
        if (preserve) {
            CHECK(reserved_mem.find(info->dptr) == reserved_mem.end());
            reserved_mem[info->dptr] = info->size | (1L << 63);
        }
    } else if (preserve) {
        CHECK(reserved_mem.find(info->dptr) == reserved_mem.end());
        reserved_mem[info->dptr] = info->size | (0L << 63);
    }
}

void Swap::DelAddr_Cache(SwapInfo *info, bool preserve, bool record) {
    cache_->DelAddr(info, record);
}

void Swap::DelAddr(handle_id_t handle_id, size_t size, bool preserve,
                   bool record) {
    //std::cout << "DelAddr " << handle_id << std::endl;
    pthread_rwlock_wrlock(&swap_lock_);
    auto info = swap_info_.at(handle_id);
    if (info->device != -1) {
        if (record) {
            mhistory_->PutRecord(handle_id, info->device, MemHistory::DEL_ADDR,
                                 size);
        }
        pthread_rwlock_wrlock(&locks_[info->device]);
        pthread_rwlock_unlock(&locks_[info->device]);
    }
    delete info;
    swap_info_.erase(handle_id);
    pthread_rwlock_unlock(&swap_lock_);
};

void Swap::GetAddr_Swap(SwapInfo *info, bool record) {
    if (info->it != lru_[info->device].end()) {
        lru_[info->device].erase(info->it);
    }
    lru_[info->device].push_front(info);
    info->it = lru_[info->device].begin();
    if (!info->swap_in && do_swap_) {
        if (record) {
            //std::cout << "Cache miss " << handle_id << " " << size << std::endl;
            cache_miss_ += 1;
        }
        SwapIn(info, !record);
    }
}

void Swap::GetAddr_Cache(SwapInfo *info, bool record) {
    cache_->GetAddr(info, record);
}

void* Swap::GetAddr(handle_id_t handle_id, size_t size, bool record) {
    //std::cout << "GetAddr " << handle_id << std::endl;
    pthread_rwlock_rdlock(&swap_lock_);
    auto info = swap_info_.at(handle_id);
    if (info->device != -1) {
        if (record) {
            mhistory_->PutRecord(handle_id, info->device, MemHistory::GET_ADDR,
                                 size);
        }
        pthread_rwlock_wrlock(&locks_[info->device]);
        if (access_stats_.size() > 0) {
            access_stats_[handle_id] += 1;
        }
        CHECK_EQ(info->size, size);
        CHECK_EQ(info->handle_id, handle_id);
        if (do_cache_) {
            GetAddr_Cache(info, record);
        } else {
            GetAddr_Swap(info, record);
        }
        pthread_rwlock_unlock(&locks_[info->device]);
    }
    pthread_rwlock_unlock(&swap_lock_);
    return info->dptr;
};

void Swap::SwapperLookahead(int device, int& lookahead_pos) {
    pthread_rwlock_rdlock(&swap_lock_);
    bool do_it = false;
    while ((int)mhistory_->history[device].size() > lookahead_pos + 1 &&
            ((lookahead_pos - mhistory_->curr_idx[device]) < look_ahead_ || do_it)) {
        do_it = true;
        lookahead_pos += 1;
        auto &h = mhistory_->history[device][lookahead_pos];
        auto info = swap_info_.at(h.handle_id);
        do_it &= (info->swap_in); // A heuristic.
        if (h.type == MemHistory::GET_ADDR) {
            Swap::Get()->GetAddr(h.handle_id, h.size, false);
        } else {
            std::cout << "The history item contains not only read item : "
                      << h.type << std::endl;
            CHECK(false);
        }
    }
    pthread_rwlock_unlock(&swap_lock_);
}

void Swap::SwapperSetLookahead(int device, int& lookahead_pos) {
    int curr_set = -1;
    for (auto accu : mhistory_->set_accu_nrecords[device]) {
        if (accu - 1 <= mhistory_->curr_idx[device]) {
            curr_set += 1;
        } else {
            break;
        }
    }
    pthread_rwlock_rdlock(&swap_lock_);
    bool do_it = false;
    while ((int)mhistory_->set_history[device].size() > lookahead_pos + 1 &&
            (lookahead_pos - curr_set < look_ahead_ || do_it)) {
        lookahead_pos += 1;
        auto &set = mhistory_->set_history[device][lookahead_pos];
        do_it = true;
        for (auto& h : set->unique_records) {
            auto info = swap_info_.at(h.handle_id);
            if (h.type == MemHistory::GET_ADDR) {
                //std::cout << "Lookahead " << h.handle_id << " " << h.size << " "
                          //<< lookahead_pos << " "
                          //<< curr_set << " " << mhistory_->curr_idx[device]
                          //<< std::endl;
                do_it &= (info->swap_in); // A heuristic.
                Swap::Get()->GetAddr(h.handle_id, h.size, false);
            } else {
                std::cout << "The history item contains not only read item : "
                          << h.type << std::endl;
                CHECK(false);
            }
        }
        while (mhistory_->set_accu_nrecords[device][curr_set] -1 <
            mhistory_->curr_idx[device]) {
            curr_set += 1;
        }
    }
    pthread_rwlock_unlock(&swap_lock_);
}

void Swap::Swapper(int device) {
    int lookahead_pos = -1;
    std::cout << "Execute Swapper()" << std::endl;
    while (!should_stop_) {
        //if (true) {
            //SwapperLookahead(device, lookahead_pos);
        //}
        if (true) {
            SwapperSetLookahead(device, lookahead_pos);
        }
        swapper_began_ = true;
        usleep(5);
    }
}

void Swap::StartIteration() {
    num_device_ = mhistory_->GetNumDevice();
    //cudaProfilerStart();
    mhistory_->StartIteration();
#if 1
    if (mhistory_->HistoryRecorded() && do_swap_) {
        for (auto& it : mhistory_->access_stats) {
            access_stats_[it.first] = 0;
        }
        should_stop_ = false;
        swapper_began_ = false;
        cache_miss_ = 0;
        waiting_swapping_ = 0;
        std::cout << "Prepare to execute Swapper()" << std::endl;
        for (int device = 0; device < num_device_; device++) {
            swapper_[device] = std::thread(&Swap::Swapper, this, device);
        }
        while (!swapper_began_) {
            usleep(5);
        }
    }
#endif
}

void Swap::StopIteration() {
    //std::cout << "Swap address " << this << std::endl;
    mhistory_->StopIteration();
    should_stop_ = true;
    if (swapper_began_) {
        size_t size = 0;
        size_t set_size = 0;
        for (int device = 0; device < num_device_; device++) {
            swapper_[device].join();
            size += mhistory_->history[device].size();
            set_size += mhistory_->set_history[device].size();
        }
        std::cout << "Total dptr access " << size << std::endl;
        std::cout << "Total set " << set_size << std::endl;
        std::cout << "We have " << cache_miss_ << " cache miss." << std::endl;
        std::cout << "We have " << waiting_swapping_ << " waiting swapping." << std::endl;
    }
    //cudaProfilerStop();
}

}  // namespace mxnet
