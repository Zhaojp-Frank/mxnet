/*!
 * Copyright (c) 2015 by Contributors
 */
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

std::shared_ptr<SwapHistory> SwapHistory::_GetSharedRef() {
    static std::shared_ptr<SwapHistory> inst(new SwapHistory());
    return inst;
}

SwapHistory* SwapHistory::Get() {
    static SwapHistory *s = _GetSharedRef().get();
    return s;
}

std::shared_ptr<Swap> Swap::_GetSharedRef() {
    static std::shared_ptr<Swap> inst(new Swap());
    return inst;
}

SwapHistory::SwapHistory() {
    iteration_started_ = false;
    iteration_idx_ = 0;
    do_record_ = false;
}

SwapHistory::~SwapHistory() {
}

void SwapHistory::PutRecord(handle_id_t id, record_t type, size_t size) {
    if (!iteration_started_) {
        return;
    }
    if (!do_record_) {
        if (history_.size() > 0) {
            std::cout << "his : "
                      << history_[record_idx_].id << " "
                      << history_[record_idx_].type << " "
                      << history_[record_idx_].size << " "
                      << history_[record_idx_].time << " "
                      << history_.size() << std::endl;
            std::cout << "now : "
                      << id << " " << type << " " << size
                      << " " << record_idx_ << std::endl;
        }
    } else {
        std::lock_guard<std::mutex> lock(mutex_);
        timestamp_t t =
            (duration_cast<microseconds>(high_resolution_clock::now() -
                                         begin_time_)).count();
        SwapRecord r = {id, type, t, size};
        history_.push_back(r);
    }
    record_idx_ += 1;
}

void SwapHistory::StartIteration() {
    iteration_started_ = true;
    record_idx_ = 0;
    if (iteration_idx_ == 1) {
        do_record_ = true ;
    }
    begin_time_ = high_resolution_clock::now();
}

void SwapHistory::StopIteration() {
    iteration_started_ = false;
    iteration_idx_ += 1;
    std::unordered_set<handle_id_t> handles;
    if (do_record_) {
        size_t size = 0;
        for (auto& h : history_) {
            auto it = handles.find(h.id);
            if (it == handles.end()) {
                size += h.size;
                handles.insert(h.id);
            }
        }
        std::cout << "Size = " << size * 1.0 / 1024 / 1024 << std::endl;
        size_t free, total;
        CUDA_CALL(cudaMemGetInfo(&free, &total));
        std::cout << "Total = " << total * 1.0 / 1024 / 1024 << std::endl;
    }
    do_record_ = false;
}

Swap* Swap::Get() {
    static Swap *s = _GetSharedRef().get();
    return s;
}

Swap::Swap() {
    std::cout << "Initialize Swap" << std::endl;
    lru_ = std::vector<std::list<SwapInfo*>>(8);
    free_memory_ = std::vector<size_t>{0, 0, 0, 0, 0, 0, 0, 0};
    do_swap_ = dmlc::GetEnv("MXNET_DO_SWAP", 0);
#if MXNET_USE_CUDA
    size_t fifo_size, heap_size;
    cudaDeviceGetLimit(&fifo_size, cudaLimitPrintfFifoSize);
    cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
    swap_threshold_ = fifo_size + heap_size + 1024 * 1024;
#endif  // MXNET_USE_CUDA
};

Swap::~Swap() {
    std::cout << "Destroy Swap" << std::endl;
}

int Swap::UpdateFree() {
    int device = 0;
    size_t free = 10000000000, total = 0;
#if MXNET_USE_CUDA
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaMemGetInfo(&free, &total));
    free_memory_[device] = free;
#endif  // MXNET_USE_CUDA
    return device;
}

void Swap::SwapOut(unsigned required_memory, int device) {
    if (!do_swap_) {
        return;
    }
#if MXNET_USE_CUDA
    if (device == -1) {
        CUDA_CALL(cudaGetDevice(&device));
    }
    UpdateFree();
    if (free_memory_[device] > required_memory + swap_threshold_) {
        return;
    }
    while (free_memory_[device] < required_memory + swap_threshold_ * 64) {
        if (lru_[device].size() > 0) {
            CHECK(lru_[device].size() > 0);
            auto target = lru_[device].back();
            lru_[device].pop_back();
            if (target->cpu_address == nullptr) {
                target->cpu_address = new char[int(target->size)];
            }
            CHECK(target->swap_in);
            CHECK(target->dptr != nullptr);
            target->swap_in = false;
            CUDA_CALL(cudaSetDevice(device));
            CUDA_CALL(cudaMemcpy(target->cpu_address, target->dptr, target->size,
                                 cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaFree(target->dptr));
        } else {
            std::cout << "Unfortunately, we need to free reserved memory."
                      << "This implementation is very tricky and dangerous "
                      << "since we don't have a persistent id. The correctness "
                      << "is not guaranteed."
                      << std::endl;
            auto it = reserved_mem_.begin();
            size_t needed = (required_memory + swap_threshold_ * 64) -
                            free_memory_[device];
            while (needed > 0) {
                while (!(it->second & (1L << 63))) {
                    it++;
                }
                CHECK(reserved_mem_.find(it->first) != reserved_mem_.end());
                CHECK(it != reserved_mem_.end());
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
        UpdateFree();
    }
#else  // MXNET_USE_CUDA
    LOG(FATAL) << "No swap out required without CUDA.";
#endif  // MXNET_USE_CUDA
}

void Swap::SwapIn(SwapInfo *info) {
    //std::cout << "SwapIn" << std::endl;
#if MXNET_USE_CUDA
    info->dptr = nullptr;
    CHECK(!info->swap_in);
    CHECK(info->cpu_address != nullptr);
    int device = 0;
    CUDA_CALL(cudaGetDevice(&device));
    SwapOut(info->size, info->device);
    CUDA_CALL(cudaSetDevice(info->device));
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    cudaError_t e = cudaMalloc(&(info->dptr), info->size);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
    }
    CHECK(info->cpu_address != nullptr);
    CUDA_CALL(cudaMemcpy(info->dptr, info->cpu_address, info->size,
                         cudaMemcpyHostToDevice));
    CUDA_CALL(cudaSetDevice(device));
    info->swap_count += 1;
    info->swap_in = true;
#else  // MXNET_USE_CUDA
    LOG(FATAL) << "No swap in required without CUDA.";
#endif  // MXNET_USE_CUDA
}

bool Swap::FreeReserved(void *ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = reserved_mem_.find(ptr);
    bool ret = true;
    if (it != reserved_mem_.end()) {
        ret = (it->second & (1L << 63));
        CHECK_EQ(it->second & (~(1L << 63)), size);
        reserved_mem_.erase(it);
    }
    return ret;
}

bool Swap::CheckReservedAndFree(void *ptr, size_t size) {
    //std::cout << "CheckReservedAndFree " << ptr << std::endl;
    auto it = reserved_mem_.find(ptr);
    CHECK_EQ(it->second & (~(1L << 63)), size);
    if (!(it->second & (1L << 63))) {
        reserved_mem_.erase(it);
        return true;
    } else {
        return false;
    }
}

void Swap::SetAddr(handle_id_t handle_id, void* dptr, size_t size) {
    SwapHistory::Get()->PutRecord(handle_id, SwapHistory::SET_ADDR, size);
    std::lock_guard<std::mutex> lock(mutex_);
    if (dptr == nullptr) {
        return ;
    }
#if 0
    CHECK(reserved_mem_.find(dptr) == reserved_mem_.end());
#endif
    //auto reserved_it = reserved_mem_.find(dptr);
    //if (reserved_it != reserved_mem_.end()) {
        //CHECK_EQ(reserved_it->second & (~(1L << 63)), size);
        //if (!(reserved_it->second & (1L << 63))) {
            //std::cout << "Unfortunately, we need to swapped reserved memory."
                      //<< "This implementation is very tricky and dangerous "
                      //<< "since we don't have a persistent id. The correctness "
                      //<< "is not guaranteed."
                      //<< std::endl;
            //cudaError_t e = cudaMalloc(&dptr, size);
            //if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
                //LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
            //}
        //}
        //reserved_mem_.erase(reserved_it);
    //}
    auto key = handle_id ^ size;
    auto iter = swap_info_.find(key);
    if (iter == swap_info_.end()) {
        int device = UpdateFree();
        CHECK(dptr != nullptr);
        auto swap_info = new SwapInfo{handle_id, true, 0, device, dptr, nullptr,
                                      size};
        swap_info_[key] = swap_info;
        lru_[device].push_front(swap_info);
        swap_info->it = lru_[device].begin();
    } else {
        std::cout << "SetAddr duplicated id " << handle_id << std::endl;
        std::cout << "SetAddr " << iter->second->size << " " << size << std::endl;
        CHECK(iter->second->swap_in);
        CHECK_EQ(iter->second->handle_id, handle_id);
        CHECK_EQ(iter->second->dptr, dptr);
        CHECK_EQ(iter->second->size, size);
    }
};

void Swap::DelAddr(handle_id_t handle_id, size_t size, bool preserve) {
    SwapHistory::Get()->PutRecord(handle_id, SwapHistory::DEL_ADDR, size);
    auto key = handle_id ^ size;
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = swap_info_.find(key);
    CHECK(iter != swap_info_.end());
    if (iter->second->cpu_address != nullptr) {
        delete iter->second->cpu_address;
    }
    if (iter->second->swap_in) {
        //std::cout << "DelAddr" << std::endl;
        lru_[iter->second->device].erase(iter->second->it);
        if (preserve) {
            CHECK(reserved_mem_.find(iter->second->dptr) == reserved_mem_.end());
            reserved_mem_[iter->second->dptr] = iter->second->size | (1L << 63);
        }
    } else if (preserve) {
        CHECK(reserved_mem_.find(iter->second->dptr) == reserved_mem_.end());
        reserved_mem_[iter->second->dptr] = iter->second->size | (0L << 63);
    }
    delete iter->second;
    swap_info_.erase(iter);
    UpdateFree();
};

void* Swap::GetAddr(handle_id_t handle_id, size_t size) {
    SwapHistory::Get()->PutRecord(handle_id, SwapHistory::GET_ADDR, size);
    auto key = handle_id ^ size;
    std::lock_guard<std::mutex> lock(mutex_);
    auto swap_info = swap_info_.at(key);
    CHECK_EQ(swap_info->size, size);
    CHECK_EQ(swap_info->handle_id, handle_id);
    if (!swap_info->swap_in && do_swap_) {
#if MXNET_USE_CUDA
        SwapIn(swap_info);
#else   // MXNET_USE_CUDA
        LOG(FATAL) << "Without CUDA, there should be no swap_in required.";
#endif  // MXNET_USE_CUDA
    } else {
        lru_[swap_info->device].erase(swap_info->it);
    }
    lru_[swap_info->device].push_front(swap_info);
    swap_info->it = lru_[swap_info->device].begin();
    return swap_info->dptr;
};

}  // namespace mxnet
