/*!
 * Copyright (c) 2015 by Contributors
 * \file pooled_storage_manager.h
 * \brief Storage manager with a memory pool.
 */
#ifndef MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_

#if MXNET_USE_CUDA
  #include <cuda_runtime.h>
#endif  // MXNET_USE_CUDA
#include <mxnet/base.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <new>
#include "./storage_manager.h"
#include "../common/cuda_utils.h"


namespace mxnet {
namespace storage {

#if MXNET_USE_CUDA
/*!
 * \brief Storage manager with a memory pool on gpu.
 */
class GPUPooledStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  GPUPooledStorageManager() {
    reserve_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_RESERVE", 5);
    swap = Swap::_GetSharedRef();
  }
  /*!
   * \brief Default destructor.
   */
  ~GPUPooledStorageManager() {
    ReleaseAll();
  }

  void* Alloc(size_t size) override;
  void Free(void* ptr, size_t size) override;

  void DirectFree(void* ptr, size_t size) override {
    if (swap->FreeReserved(ptr, size)) {
        cudaError_t err = cudaFree(ptr);
        // ignore unloading error, as memory has already been recycled
        if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
          LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
        }
    }
    used_memory_ -= size;
  }

 private:
  void ReleaseAll();
  // internal mutex
  std::mutex mutex_;
  // used memory
  size_t used_memory_ = 0;
  // percentage of reserved memory
  int reserve_;
  std::shared_ptr<Swap> swap;
  // memory pool
  std::unordered_map<size_t, std::vector<void*>> memory_pool_;
  DISALLOW_COPY_AND_ASSIGN(GPUPooledStorageManager);
};  // class GPUPooledStorageManager

void* GPUPooledStorageManager::Alloc(size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto&& reuse_it = memory_pool_.find(size);

  if (reuse_it != memory_pool_.end()) {
    auto&& reuse_pool = reuse_it->second;
    auto it = reuse_pool.begin();
    while (it != reuse_pool.end()) {
        if (swap->CheckReservedAndFree(*it, size)) {
            it = reuse_pool.erase(it);
        } else {
            it++;
        }
    }
  }
  if (reuse_it == memory_pool_.end() || reuse_it->second.size() == 0) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    if (free <= total * reserve_ / 100 || size > free - total * reserve_ / 100)
      ReleaseAll();

    swap->SwapOut(size, -1, true, false);
    void* ret = nullptr;
    cudaError_t e = cudaMalloc(&ret, size);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      cudaMemGetInfo(&free, &total);
      std::cout << "Why fail? " << free << " " << size << std::endl;
      LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
    }
    used_memory_ += size;
    return ret;
  } else {
    std::cout << "Reuse " << std::endl;
    auto&& reuse_pool = reuse_it->second;
    auto ret = reuse_pool.back();
    reuse_pool.pop_back();
    return ret;
  }
}

void GPUPooledStorageManager::Free(void* ptr, size_t size) {
  //std::cout << "Doing Free()" << std::endl;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto&& reuse_it = memory_pool_.begin(); reuse_it != memory_pool_.end(); reuse_it++) {
    auto&& reuse_pool = reuse_it->second;
    auto it = reuse_pool.begin();
    while (it != reuse_pool.end()) {
        if (swap->CheckReservedAndFree(*it, reuse_it->first)) {
            it = reuse_pool.erase(it);
        } else {
            it++;
        }
    }
  }
  auto&& reuse_pool = memory_pool_[size];
  reuse_pool.push_back(ptr);
}

void GPUPooledStorageManager::ReleaseAll() {
  for (auto&& i : memory_pool_) {
    for (auto&& j : i.second) {
      DirectFree(j, i.first);
    }
  }
  memory_pool_.clear();
}
#endif  // MXNET_USE_CUDA

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
