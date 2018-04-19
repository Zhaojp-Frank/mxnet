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
  hd.SetDptr(manager->Alloc(size));
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

std::random_device Storage::Handle::handle_rd;
std::mt19937_64 Storage::Handle::handle_gen;
std::uniform_int_distribution<handle_id_t> Storage::Handle::handle_dis;
handle_id_t Storage::Handle::base_id_ = 0;
bool Storage::Handle::handle_random = false;

Storage* Storage::Get() {
  static Storage *ptr = _GetSharedRef().get();
  return ptr;
}

std::shared_ptr<MemHistory> MemHistory::_GetSharedRef() {
    static std::shared_ptr<MemHistory> inst(new MemHistory());
    return inst;
}

MemHistory* MemHistory::Get() {
    static MemHistory *s = _GetSharedRef().get();
    return s;
}

std::shared_ptr<Swap> Swap::_GetSharedRef() {
    static std::shared_ptr<Swap> inst(new Swap());
    return inst;
}

MemHistory::MemHistory() {
    iteration_started_ = false;
    iteration_idx_ = 0;
    do_record_ = false;
}

MemHistory::~MemHistory() {
    std::cout << "Destroy MemHistory" << std::endl;
}

void MemHistory::PutRecord(handle_id_t id, record_t type, size_t size) {
    if (!iteration_started_) {
        return;
    }
    if (do_record_) {
        std::lock_guard<std::mutex> lock(mutex_);
        timestamp_t t =
            (duration_cast<microseconds>(high_resolution_clock::now() -
                                         begin_time_)).count();
        MemRecord r = {id, type, t, size};
        history.push_back(r);
        //std::cout << "his : "
                  //<< history[record_idx].id << " "
                  //<< history[record_idx].type << " "
                  //<< history[record_idx].size << " "
                  //<< history[record_idx].time << " "
                  //<< history.size() << std::endl;
                  //
    } else {
        //if (history.size() > 0) {
            //std::cout << "his : "
                      //<< history[record_idx].id << " "
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
    iteration_started_ = true;
    record_idx = 0;
    if (iteration_idx_ == 2) {
        do_record_ = true ;
    }
    begin_time_ = high_resolution_clock::now();
}

void MemHistory::CreateMacroOrder() {
    unsigned threshold = dmlc::GetEnv("MXNET_MACRO_MEM_ORDER_THRESHOLD", 500);
    unsigned previous = 0;
    std::unordered_map<handle_id_t, MemRecord> members;
    for (auto& h : history) {
        if (h.time - previous > threshold) {
            std::vector<MemRecord> temp;
            unsigned long time = 0;
            for (auto m : members) {
                if (time == 0) {
                    time = m.second.time;
                }
                temp.push_back(MemRecord{m.second.id, m.second.type, time,
                                         m.second.size});
            }
            std::sort(temp.begin(), temp.end(),
                      [](const MemRecord& a, const MemRecord& b) -> bool {
                        return a.size > b.size;
                      });
            macro_history.insert(macro_history.end(), temp.begin(),
                                 temp.end());
            members.clear();
        }
        if (members.find(h.id) == members.end()) {
            members[h.id] = h;
        }
        previous = h.time;
    }
#if 0
    std::cout << "Macro history:" << std::endl;
    for (auto& h : macro_history) {
        std::cout << h.time << " " << h.id << " " << h.size << std::endl;
    }
#endif
}

void MemHistory::StopIteration() {
    iteration_started_ = false;
    iteration_idx_ += 1;
    if (do_record_) {
        CreateMacroOrder();
#if 0
        std::unordered_set<handle_id_t> handles;
        size_t size = 0;
        unsigned long previous = 0;
        std::cout << "differences : " << std::endl;
        for (auto& h : history) {
            auto it = handles.find(h.id);
            std::cout << h.time - previous << std::endl;
            previous = h.time;
            if (it == handles.end()) {
                size += h.size;
                handles.insert(h.id);
            }
        }
        std::cout << "Size = " << size * 1.0 / 1024 / 1024 << std::endl;
        size_t free, total;
        CUDA_CALL(cudaMemGetInfo(&free, &total));
        std::cout << "Total = " << total * 1.0 / 1024 / 1024 << std::endl;
#endif
    }
    do_record_ = false;
}

Swap* Swap::Get() {
    static Swap *s = _GetSharedRef().get();
    return s;
}

Swap::Swap() {
    std::cout << "Initialize Swap" << std::endl;
    do_swap_ = dmlc::GetEnv("MXNET_DO_SWAP", 0);
    swap_lock_ = PTHREAD_RWLOCK_INITIALIZER;
    swapper_began_ = false;
    look_ahead_ = dmlc::GetEnv("MXNET_SWAPPER_LOOK_AHEAD", 100);
    lru_ = std::vector<std::list<SwapInfo*>>(8);
    reserved_mem_ = std::vector<std::unordered_map<void*, size_t>>(8);
    for (int i = 0; i < 8; i++) {
        locks_[i] = PTHREAD_RWLOCK_INITIALIZER;
        streams_init_[i] = false;
        free_memory_.push_back(0);
    }
#if MXNET_USE_CUDA
    size_t fifo_size, heap_size;
    cudaDeviceGetLimit(&fifo_size, cudaLimitPrintfFifoSize);
    cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
    unsigned swap_threshold_multiplier =
        dmlc::GetEnv("MXNET_SWAP_THRESHOLD_MULTIPLIER", 32);
    swap_threshold_ = (fifo_size + heap_size + 1024 * 1024) *
                      swap_threshold_multiplier;
#endif  // MXNET_USE_CUDA
};

Swap::~Swap() {
    std::cout << "Destroy Swap" << std::endl;
}

int Swap::UpdateFree(int device) {
    size_t free = 10000000000, total = 0;
#if MXNET_USE_CUDA
    CUDA_CALL(cudaMemGetInfo(&free, &total));
    free_memory_[device] = free;
    //std::cout << "free_memory_ " << free_memory_[device] << std::endl;
#endif  // MXNET_USE_CUDA
    return device;
}

void Swap::SwapOut(unsigned required_memory, int device,
                   bool acquire_lock=false, bool async=false) {
    if (!do_swap_) {
        return;
    }
#if MXNET_USE_CUDA
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
            //std::cout << "Size " << lru_[device].size() << std::endl;
            //CHECK(target->swap_in);
            //CHECK(target->dptr != nullptr);
            if (target->dptr == nullptr || !target->swap_in) {
                std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl
                          << "swap_info->dptr is nullptr or "
                          << "target->swap_in is not set." << std::endl
                          << "This should happen only when all iterations are "
                          << "done." << std::endl
                          << "Current lru size is " << lru_[device].size()
                          << std::endl;
                ignored_lru += 1;
                lru_[device].push_front(target);
                continue;
            }
            CHECK(!target->is_swapping.test_and_set(std::memory_order_acquire));
            //std::cout << "Swapping out " << target->handle_id << std::endl;
            target->swap_in = false;
            target->it = lru_[device].end();
            pthread_rwlock_unlock(&locks_[device]);
            if (target->cpu_address == nullptr) {
                target->cpu_address = new char[int(target->size)];
            }
            if (!streams_init_[device]) {
                streams_init_[device] = true;
                cudaStreamCreate(&streams_[device]);
            }
            if (async) {
                CUDA_CALL(cudaMemcpyAsync(target->cpu_address, target->dptr,
                                          target->size,
                                          cudaMemcpyDeviceToHost,
                                          streams_[device]));
                CUDA_CALL(cudaStreamSynchronize(streams_[device]));
            } else {
                CUDA_CALL(cudaMemcpy(target->cpu_address, target->dptr,
                                     target->size, cudaMemcpyDeviceToHost));
            }
            CUDA_CALL(cudaFree(target->dptr));
            target->dptr = nullptr;
            pthread_rwlock_wrlock(&locks_[device]);
            target->is_swapping.clear(std::memory_order_release);
        } else {
            std::cout << "Unfortunately, we need to free reserved memory."
                      << "This implementation is very tricky and dangerous "
                      << "since we don't have a persistent id. The correctness "
                      << "is not guaranteed."
                      << std::endl;
            auto it = reserved_mem_[device].begin();
            size_t needed = (required_memory + swap_threshold_) -
                            free_memory_[device];
            while (needed > 0) {
                while (!(it->second & (1L << 63))) {
                    it++;
                }
                CHECK(reserved_mem_[device].find(it->first) != reserved_mem_[device].end());
                CHECK(it != reserved_mem_[device].end());
                CUDA_CALL(cudaSetDevice(device));
                CUDA_CALL(cudaFree(it->first));
                it->second &= (~(1L << 63));
                if (it->second > needed) {
                    needed = 0;
                } else {
                    needed -= it->second;
                }
            }
        }
        UpdateFree(device);
    }
#else  // MXNET_USE_CUDA
    LOG(FATAL) << "No swap out required without CUDA.";
#endif  // MXNET_USE_CUDA
    if (acquire_lock) {
        pthread_rwlock_unlock(&locks_[device]);
    }
}

void Swap::SwapIn(SwapInfo *info, bool async=false) {
    //std::cout << "SwapIn " << info->handle_id << std::endl;
#if MXNET_USE_CUDA
    bool waiting = false;
    while (info->is_swapping.test_and_set(std::memory_order_acquire)) {
        waiting = true;
        pthread_rwlock_unlock(&locks_[info->device]);
        usleep(100);
        pthread_rwlock_wrlock(&locks_[info->device]);
    }
    if (waiting) {
        waiting_swapping_ += 1;
    }
    //std::cout << "SwapIn " << info->handle_id << std::endl;
    if (!info->swap_in) {
        CHECK(info->dptr == nullptr);
        CHECK(info->cpu_address != nullptr);
        int old_device = 0;
        CUDA_CALL(cudaGetDevice(&old_device));
        SwapOut(info->size, info->device, false, async);
        CHECK(free_memory_[info->device] > info->size);
        CUDA_CALL(cudaSetDevice(info->device));
        pthread_rwlock_unlock(&locks_[info->device]);
        cudaError_t e = cudaMalloc(&(info->dptr), info->size);
        if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
            LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
        }
        CHECK(info->cpu_address != nullptr);
        if (async) {
            CUDA_CALL(cudaMemcpyAsync(info->dptr, info->cpu_address, info->size,
                                      cudaMemcpyHostToDevice,
                                      streams_[info->device]));
            CUDA_CALL(cudaStreamSynchronize(streams_[info->device]));
        } else {
            CUDA_CALL(cudaMemcpy(info->dptr, info->cpu_address, info->size,
                                 cudaMemcpyHostToDevice));
        }
        pthread_rwlock_wrlock(&locks_[info->device]);
        CUDA_CALL(cudaSetDevice(old_device));
        info->swap_count += 1;
        info->swap_in = true;
        //std::cout << "Do swap in" << std::endl;
#else  // MXNET_USE_CUDA
        LOG(FATAL) << "No swap in required without CUDA.";
#endif  // MXNET_USE_CUDA
    } else {
        //std::cout << "Doesn't do swap in" << std::endl;
    }
    info->is_swapping.clear(std::memory_order_release);
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

void Swap::SetAddr(handle_id_t handle_id, void* dptr, size_t size,
                   bool record) {
    //std::cout << "SetAddr " << handle_id << std::endl;
    if (record) {
        MemHistory::Get()->PutRecord(handle_id, MemHistory::DEL_ADDR, size);
    }
    if (dptr == nullptr) {
        return ;
    }
    auto key = handle_id ^ size;
    pthread_rwlock_wrlock(&swap_lock_);
    auto iter = swap_info_.find(key);
    if (iter == swap_info_.end()) {
        int device = 0;
        CUDA_CALL(cudaGetDevice(&device));
        UpdateFree(device);
        CHECK(dptr != nullptr);
        auto swap_info = new SwapInfo{handle_id, true, ATOMIC_FLAG_INIT, 0,
                                      device, dptr, nullptr, size};
        swap_info_[key] = swap_info;
        pthread_rwlock_wrlock(&locks_[device]);
        lru_[device].push_front(swap_info);
        swap_info->it = lru_[device].begin();
        pthread_rwlock_unlock(&locks_[device]);
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
    if (record) {
        MemHistory::Get()->PutRecord(handle_id, MemHistory::DEL_ADDR, size);
    }
    //std::cout << "DelAddr " << handle_id << std::endl;
    MemHistory::Get()->PutRecord(handle_id, MemHistory::DEL_ADDR, size);
    auto key = handle_id ^ size;
    pthread_rwlock_wrlock(&swap_lock_);
    auto iter = swap_info_.find(key);
    pthread_rwlock_wrlock(&locks_[iter->second->device]);
    CHECK(iter != swap_info_.end());
    if (iter->second->cpu_address != nullptr) {
        delete iter->second->cpu_address;
    }
    if (iter->second->swap_in) {
        //std::cout << "DelAddr" << std::endl;
        lru_[iter->second->device].erase(iter->second->it);
        iter->second->it = lru_[iter->second->device].end();
        if (preserve) {
            CHECK(reserved_mem_[iter->second->device].find(iter->second->dptr)
                    == reserved_mem_[iter->second->device].end());
            reserved_mem_[iter->second->device][iter->second->dptr] =
                    iter->second->size | (1L << 63);
        }
    } else if (preserve) {
        CHECK(reserved_mem_[iter->second->device].find(iter->second->dptr)
                == reserved_mem_[iter->second->device].end());
        reserved_mem_[iter->second->device][iter->second->dptr] =
                iter->second->size | (0L << 63);
    }
    UpdateFree(iter->second->device);
    pthread_rwlock_unlock(&locks_[iter->second->device]);
    delete iter->second;
    swap_info_.erase(iter);
    pthread_rwlock_unlock(&swap_lock_);
};

void* Swap::GetAddr(handle_id_t handle_id, size_t size, bool record) {
    //std::cout << "GetAddr " << handle_id << std::endl;
    if (record) {
        MemHistory::Get()->PutRecord(handle_id, MemHistory::GET_ADDR, size);
    }
    auto key = handle_id ^ size;
    pthread_rwlock_rdlock(&swap_lock_);
    auto swap_info = swap_info_.at(key);
    pthread_rwlock_wrlock(&locks_[swap_info->device]);
    CHECK_EQ(swap_info->size, size);
    CHECK_EQ(swap_info->handle_id, handle_id);

    if (swap_info->it != lru_[swap_info->device].end()) {
        lru_[swap_info->device].erase(swap_info->it);
    }
    lru_[swap_info->device].push_front(swap_info);
    swap_info->it = lru_[swap_info->device].begin();

    if (!swap_info->swap_in && do_swap_) {
#if MXNET_USE_CUDA
        if (record) {
            cache_miss_ += 1;
        }
        SwapIn(swap_info, !record);
#else   // MXNET_USE_CUDA
        LOG(FATAL) << "Without CUDA, there should be no swap_in required.";
#endif  // MXNET_USE_CUDA
    }

    pthread_rwlock_unlock(&locks_[swap_info->device]);
    pthread_rwlock_unlock(&swap_lock_);
    return swap_info->dptr;
};

void Swap::Swapper() {
    cache_miss_ = 0;
    waiting_swapping_ = 0;
    size_t curr_pos = 0;
    std::cout << "Execute Swapper()" << std::endl;
    while (!should_stop_) {
        auto shistory = MemHistory::Get();
        while (shistory->history.size() > curr_pos &&
               (shistory->record_idx >= curr_pos ||
                (curr_pos - shistory->record_idx) < look_ahead_)) {
            auto &h = shistory->history[curr_pos];
            auto info = swap_info_.at(h.id ^ h.size);
            if (h.type == MemHistory::GET_ADDR) {
                Swap::Get()->GetAddr(h.id, h.size, false);
            } else {
                std::cout << "The history item contains not only read item : "
                          << h.type << std::endl;
            }
            curr_pos += 1;
        }
        swapper_began_ = true;
        usleep(5);
    }
}

void Swap::StartIteration() {
    //cudaProfilerStart();
    MemHistory::Get()->StartIteration();
    if (MemHistory::Get()->HistoryRecorded() && do_swap_) {
        should_stop_ = false;
        swapper_began_ = false;
        std::cout << "Prepare to execute Swapper()" << std::endl;
        swapper_ = std::thread(&Swap::Swapper, this);
        while (!swapper_began_) {
            usleep(10);
        }
    }
}

void Swap::StopIteration() {
    MemHistory::Get()->StopIteration();
    should_stop_ = true;
    if (swapper_began_) {
        swapper_.join();
        std::cout << "We have " << cache_miss_ << " cache miss." << std::endl;
        std::cout << "We have " << waiting_swapping_ << " waiting swapping." << std::endl;
    }
    //cudaProfilerStop();
}

}  // namespace mxnet
