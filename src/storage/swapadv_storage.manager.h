#ifndef MXNET_STORAGE_SWAPADVISOR_STORAGE_MANAGER_H_
#define MXNET_STORAGE_SWAPADVISOR_STORAGE_MANAGER_H_

#if MXNET_USE_CUDA
  #include <cuda_runtime.h>
#endif  // MXNET_USE_CUDA
#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <mutex>
#include <new>
#include <fstream>
#include <mxnet/mm_dptr.h>
#include "./storage_manager.h"
#include "../common/cuda_utils.h"
#include "../common/utils.h"


namespace mxnet {
namespace storage {

class SA_MM_Dptr {
 public:
  void* Alloc(handle_id_t id, void* ptr) {
    dptr_mapping_[id] = ptr;
    return ptr;
  }

  void* Free(handle_id_t id) {
    auto it = dptr_mapping_.find(id);
    void* ptr = *it;
    dptr_mapping_.erase(it);
    return ptr;
  }

  void Release(handle_id_t id, void* ptr) {
    dptr_mapping_[id] = ptr;
  }

  void* GetDptr(handle_id_t id) {
    dptr_mapping_.at(id);
  }

  void SetDptr(handle_id_t id, void* ptr, uint32_t dev_id) {
    dptr_mapping_[id] = ptr;
  }

 private:
  std::unordered_map<handle_id_t, void*> dptr_mapping_;
};

#if MXNET_USE_CUDA
/*!
 * \brief Storage manager with a memory pool on gpu. Memory chunks are reused based on exact size
 * match.
 */
class GPUSwapAdvStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  GPUSwapAdvStorageManager(int device_id) {
    device_id_ = device_id;
    memory_size_ = 10L * 1024 * 1024 * 1024;
    cudaError_t e = cudaMalloc(&memory_, memory_size_);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "Can't allocate the memory in the initialization of "
                 << "the memory manager. The required size = " << memory_size_  
                 <<cudaGetErrorString(e);
    }
    ReadAllocationRst();

    void* address = memory_;
    for (size_t i = 0; i < mempool_counts_.size(); i++) {
      mempools_.emplace_back(std::unordered_set<void*>());
      used_mempools_.emplace_back(std::unordered_set<void*>());
      for (size_t j = 0; j < mempool_counts_[i]; j++) {
        mempools_[i].emplace(address);
        address = (void*)((size_t)address + mempool_to_size_[i]);
      }
    }
  }
  /*!
   * \brief Default destructor.
   */
  ~GPUSwapAdvStorageManager() {
    cudaFree(memory_);
    memory_ = nullptr;
  }

  void Alloc(Storage::Handle* handle) override;
  void Free(Storage::Handle handle) override;

  void DirectFree(Storage::Handle handle) override {
    Free(handle);
  }

 private:
  void ReadAllocationRst();
  // Pointer to the main memory allocation.
  void *memory_;
  // The size which the memory manager is allowed to use for the main memory.
  size_t memory_size_;
  // Pointer to the temoprary memory;
  void *temp_memory_;
  // The size which the memory manager is allowed to use for the temp memory.
  size_t temp_memory_size_;
  // Memory pool index to memory pool size.
  std::vector<size_t> mempool_to_size_;
  // Memory pool objects count
  std::vector<size_t> mempool_counts_;
  // The maaping from real sizes to memory pool index.
  std::unordered_map<size_t, size_t> rsize_to_mempool_;
  // Memory pools
  std::vector<std::unordered_set<void*>> mempools_;
  // Used memory pools
  std::vector<std::unordered_set<void*>> used_mempools_;
  // Used memory
  size_t used_memory_ = 0;
  int device_id_;
  DISALLOW_COPY_AND_ASSIGN(GPUSwapAdvStorageManager);
};  // class GPUSwapAdvStorageManager

void GPUSwapAdvStorageManager::ReadAllocationRst() {
  std::cout << "ReadAllocationRst" << std::endl;
  std::ifstream ifs("memalloc.rst");
  std::string line;

  size_t next = 0, last = 0;
  std::getline(ifs, line);
  while ((next = line.find(",", last)) != std::string::npos) {
    next = line.find(",", last);
    uint32_t size = std::stoi(line.substr(last, next - last));
    mempool_to_size_.emplace_back(size);
  }

  std::getline(ifs, line);
  next = last = 0;
  while ((next = line.find(",", last)) != std::string::npos) {
    next = line.find(",", last);
    uint32_t count = std::stoi(line.substr(last, next - last));
    mempool_counts_.emplace_back(count);
  }

  std::getline(ifs, line);
  next = last = 0;
  while ((next = line.find(",", last)) != std::string::npos) {
    std::string temp = line.substr(last, next - last);
    size_t colon_idx = temp.find(":", 0);
    size_t rsize = std::stoi(temp.substr(0, colon_idx));
    size_t mempool = std::stoi(temp.substr(colon_idx + 1, std::string::npos));
    rsize_to_mempool_[rsize] = mempool;
  }
}

void GPUSwapAdvStorageManager::Alloc(Storage::Handle* handle) {
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  size_t mempool = rsize_to_mempool_.at(handle->size);
  void* address = *(mempools_.at(mempool).begin());
  mempools_.at(mempool).erase(address);
  //handle->dptr = address;
  //handle->SetDptr(address, device_id_);
  handle->SetDptr(address, device_id_);
}

void GPUSwapAdvStorageManager::Free(Storage::Handle handle) {
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  size_t mempool = rsize_to_mempool_.at(handle.size);
  CHECK_EQ(used_mempools_[mempool].erase(handle.GetDptr()), 1);
  mempools_.at(mempool).emplace(handle.GetDptr());
}


#endif  // MXNET_USE_CUDA

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_GPU_SWAPADVISOR_STORAGE_MANAGER_H_
