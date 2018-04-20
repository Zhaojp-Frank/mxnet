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
  Handle Alloc(size_t size, Context ctx) override;
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

Storage::Handle StorageImpl::Alloc(size_t size, Context ctx) {
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
  hd.SetDptr(manager->Alloc(size), dev_id);
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
                  //<< history[record_idx].handle_id << " "
                  //<< history[record_idx].type << " "
                  //<< history[record_idx].size << " "
                  //<< history[record_idx].time << " "
                  //<< history.size() << std::endl;
                  //
    } else {
        //if (history.size() > 0) {
            //std::cout << "his : "
                      //<< history[record_idx].handle_id << " "
                      //<< history[record_idx].type << " "
                      //<< history[record_idx].size << " "
                      //<< history[record_idx].time << " "
                      //<< history.size() << std::endl;
            //std::cout << "now : "
                      //<< id << " " << type << " " << size
                      //<< " " << record_idx << std::endl;
        //}
    }
    record_idx += 1;
}

void MemHistory::StartIteration() {
    unsigned start_iteration = dmlc::GetEnv("MXNET_MEM_RECORD_ITERATION", 4);
    iteration_started_ = true;
    record_idx = 0;
    do_record_ = (iteration_idx_ == start_iteration);
    begin_time_ = high_resolution_clock::now();
}

void MemHistory::Analyze() {
    unsigned threshold = dmlc::GetEnv("MXNET_MACRO_MEM_ORDER_THRESHOLD", 500);
    for (int device = 0; device < num_device_; device++) {
        unsigned previous_time = 0;
        std::unordered_map<handle_id_t, MemRecord> members;
        for (auto& h : history[device]) {
            access_stats[h.handle_id] += 1;
            if (h.time - previous_time > threshold) {
                std::vector<MemRecord> temp;
                unsigned long time = members.begin()->second.time;
                for (auto m : members) {
                    temp.push_back(MemRecord{m.second.handle_id, m.second.type, 
                                             time, m.second.size});
                }
                std::sort(temp.begin(), temp.end(),
                          [](const MemRecord& a, const MemRecord& b) -> bool {
                            return a.size > b.size;
                          });
                macro_history[device].insert(macro_history[device].end(),
                                             temp.begin(), temp.end());
                members.clear();
            }
            if (members.find(h.handle_id) == members.end()) {
                members[h.handle_id] = h;
            }
            previous_time = h.time;
        }
    }
#if 0
    std::cout << "Macro history:" << std::endl;
    for (auto& h : macro_history) {
        std::cout << h.time << " " << h.handle_id << " " << h.size << std::endl;
    }

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
    do_swap_ = dmlc::GetEnv("MXNET_DO_SWAP", 0);
    look_ahead_ = dmlc::GetEnv("MXNET_SWAPPER_LOOK_AHEAD", 100);
    free_cpu_ = dmlc::GetEnv("MXNET_FREE_CPU_MEMORY", false);
    unsigned multiplier = dmlc::GetEnv("MXNET_SWAP_THRESHOLD_MULTIPLIER", 32);
    std::cout << "MXNET_DO_SWAP = " << do_swap_ << std::endl;
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
    int device;
    CUDA_CALL(cudaGetDevice(&device));
    pthread_rwlock_wrlock(&locks_[device]);
    auto it = reserved_mem_[device].find(ptr);
    bool ret = true;
    if (it != reserved_mem_[device].end()) {
        ret = (it->second & (1L << 63));
        CHECK_EQ(it->second & (~(1L << 63)), size);
        reserved_mem_[device].erase(it);
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
                    MemHistory::Get()->access_stats[info->handle_id])) {

            if (async) {
                CUDA_CALL(cudaMemcpyAsync(info->cpu_address, info->dptr,
                                          info->size,
                                          cudaMemcpyDeviceToHost,
                                          streams_[info->device]));
                CUDA_CALL(cudaStreamSynchronize(streams_[info->device]));
            } else {
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
                std::cout << "(swap_info->dptr == nullptr || !target->swap_in)."
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

void Swap::SetAddr(handle_id_t handle_id, void* dptr, size_t size, int dev_id,
                   bool record) {
    //std::cout << "SetAddr " << handle_id << " " << dptr << std::endl;
    if (dev_id != -1 && record) {
        MemHistory::Get()->PutRecord(handle_id, dev_id, MemHistory::DEL_ADDR,
                                     size);
    }
    if (dptr == nullptr) {
        return;
    }
    pthread_rwlock_wrlock(&swap_lock_);
    auto iter = swap_info_.find(handle_id);
    if (iter == swap_info_.end()) {
        CHECK(dptr != nullptr);
        SwapInfo* swap_info = new SwapInfo{handle_id, true, ATOMIC_FLAG_INIT, 0,
                                           dev_id, dptr, nullptr, size};
        swap_info_[handle_id] = swap_info;
        if (dev_id != -1) {
            pthread_rwlock_wrlock(&locks_[dev_id]);
            UpdateFree(dev_id);
            lru_[dev_id].push_front(swap_info);
            swap_info->it = lru_[dev_id].begin();
            pthread_rwlock_unlock(&locks_[dev_id]);
        } else {
            swap_info->it = lru_[0].end();
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

void Swap::DelAddr(handle_id_t handle_id, size_t size, bool preserve,
                   bool record) {
    //std::cout << "DelAddr " << handle_id << std::endl;
    pthread_rwlock_wrlock(&swap_lock_);
    auto info = swap_info_.at(handle_id);
    if (info->device != -1) {
        if (record) {
            MemHistory::Get()->PutRecord(handle_id, info->device,
                                         MemHistory::DEL_ADDR, size);
        }
        pthread_rwlock_wrlock(&locks_[info->device]);
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
        UpdateFree(info->device);
        pthread_rwlock_unlock(&locks_[info->device]);
    }
    delete info;
    swap_info_.erase(handle_id);
    pthread_rwlock_unlock(&swap_lock_);
};

void* Swap::GetAddr(handle_id_t handle_id, size_t size, bool record) {
    //std::cout << "GetAddr " << handle_id << std::endl;
    pthread_rwlock_rdlock(&swap_lock_);
    auto info = swap_info_.at(handle_id);
    if (info->device != -1) {
        if (record) {
            MemHistory::Get()->PutRecord(handle_id, info->device,
                                         MemHistory::GET_ADDR, size);
        }
        pthread_rwlock_wrlock(&locks_[info->device]);
        if (access_stats_.size() > 0) {
            access_stats_[handle_id] += 1;
        }
        CHECK_EQ(info->size, size);
        CHECK_EQ(info->handle_id, handle_id);

        if (info->it != lru_[info->device].end()) {
            lru_[info->device].erase(info->it);
        }
        lru_[info->device].push_front(info);
        info->it = lru_[info->device].begin();
        if (!info->swap_in && do_swap_) {
            if (record) {
                cache_miss_ += 1;
            }
            SwapIn(info, !record);
        }
        pthread_rwlock_unlock(&locks_[info->device]);
    }
    pthread_rwlock_unlock(&swap_lock_);
    return info->dptr;
};

void Swap::Swapper(int device) {
    cache_miss_ = 0;
    waiting_swapping_ = 0;
    size_t curr_pos = 0;
    std::cout << "Execute Swapper()" << std::endl;
    auto mhistory = MemHistory::Get();
    for (auto& it : mhistory->access_stats) {
        access_stats_[it.first] = 0;
    }
    while (!should_stop_) {
        while (mhistory->history[device].size() > curr_pos &&
               (mhistory->record_idx >= curr_pos ||
                (curr_pos - mhistory->record_idx) < look_ahead_)) {
            auto &h = mhistory->history[device][curr_pos];
            auto info = swap_info_.at(h.handle_id);
            if (h.type == MemHistory::GET_ADDR) {
                Swap::Get()->GetAddr(h.handle_id, h.size, false);
            } else {
                std::cout << "The history item contains not only read item : "
                          << h.type << std::endl;
                CHECK(false);
            }
            curr_pos += 1;
        }
        swapper_began_ = true;
        usleep(50);
    }
}

void Swap::StartIteration() {
    num_device_ = MemHistory::Get()->GetNumDevice();
    //cudaProfilerStart();
    MemHistory::Get()->StartIteration();
    if (MemHistory::Get()->HistoryRecorded() && do_swap_) {
        should_stop_ = false;
        swapper_began_ = false;
        std::cout << "Prepare to execute Swapper()" << std::endl;
        for (int device = 0; device < num_device_; device++) {
            swapper_[device] = std::thread(&Swap::Swapper, this, device);
        }
        while (!swapper_began_) {
            usleep(5);
        }
    }
}

void Swap::StopIteration() {
    MemHistory::Get()->StopIteration();
    should_stop_ = true;
    if (swapper_began_) {
        size_t size = 0;
        for (int device = 0; device < num_device_; device++) {
            swapper_[device].join();
            size += MemHistory::Get()->history[device].size();
        }
        std::cout << "Total dptr access " << size << std::endl;
        std::cout << "We have " << cache_miss_ << " cache miss." << std::endl;
        std::cout << "We have " << waiting_swapping_ << " waiting swapping." << std::endl;
    }
    //cudaProfilerStop();
}

}  // namespace mxnet
