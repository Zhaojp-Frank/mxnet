#ifndef MXNET_STORAGE_SWAPADVISOR_MM_DPTR_H_
#define MXNET_STORAGE_SWAPADVISOR_MM_DPTR_H_

#include <cuda_runtime.h>
#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <fstream>

namespace mxnet {
namespace storage {
class SA_MM_Dptr : virtual public MM_Dptr {
 public:
  SA_MM_Dptr() {
    memory_size_ = 10L * 1024 * 1024 * 1024;
    temp_size_ = 1L * 1024 * 1024 * 1024;
    cudaError_t e = cudaMalloc(&memory_, memory_size_);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "Can't allocate the memory in the initialization of "
                 << "the memory manager. The required size = " << memory_size_
                 <<cudaGetErrorString(e);
    }
    e = cudaMalloc(&temp_memory_, temp_size_);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "Can't allocate the memory in the initialization of "
                 << "the memory manager. The required size = " << temp_size_
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
    regular_finalized_ = false;
    used_memory_ = 0;
    temp_user = 0;
  }

  void* Alloc(handle_id_t id, size_t size, void* ptr=nullptr) {
    //std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
    if (regular_finalized_) {
      CHECK_EQ(temp_user, 0);
      temp_user = id;
      return temp_memory_;
    }
    size_t mempool = rsize_to_mempool_.at(size);
    void* address = *(mempools_.at(mempool).begin());
    mempools_.at(mempool).erase(address);
    hdl_dptr_mapping_[id] = address;
    hdl_size_mapping_[id] = size;
    return address;
  }

  void* Free(handle_id_t id) {
    if (id == temp_user) {
      temp_user = 0;
      return nullptr;
    }
    //std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
    size_t size = hdl_size_mapping_.at(id);
    size_t mempool = rsize_to_mempool_.at(size);
    auto it = hdl_dptr_mapping_.find(id);
    void* address = it->second;
    hdl_dptr_mapping_.erase(it);
    CHECK_EQ(used_mempools_[mempool].erase(address), 1);
    mempools_.at(mempool).emplace(address);
    return nullptr;
  }

  void Release(handle_id_t id, void* ptr) {
    // No need to implement for this manager;
    CHECK(0);
  }

  void RegisterEntry(uint32_t nid, uint32_t idx, handle_id_t hid, bool is_var) {
    uint32_t eid = nid * hash_const + idx;
    entry_hdl_mapping_[eid] = std::make_pair(hid, is_var);
  }

  void FinalizeRegular() {
    regular_finalized_ = true;
  }

  void* GetDptr(handle_id_t id) {
    return hdl_dptr_mapping_.at(id);
  }

  void SetDptr(handle_id_t id, void* ptr, uint32_t dev_id) {
    hdl_dptr_mapping_[id] = ptr;
  }

 private:
  // Handle to dptr mapping. If the result it nulldptr, the handle is swapped
  // out.
  std::unordered_map<handle_id_t, void*> hdl_dptr_mapping_;
  // Handle to size mapping.
  std::unordered_map<handle_id_t, size_t> hdl_size_mapping_;
  // Let we know which device does the handle belong to.
  // Not useful now since we have not supported multiple devices.
  std::unordered_map<handle_id_t, size_t> hdl_dev_mapping_;
  // An entry (NDArray) to handle mapping.
  std::unordered_map<uint32_t, std::pair<handle_id_t, bool>> entry_hdl_mapping_;

  static constexpr uint32_t hash_const = 33;
  // Read the memory allocation result from the SwapAdvisor.
  void ReadAllocationRst();
  // Pointer to the main memory allocation.
  void *memory_;
  // The size which the memory manager is allowed to use for the main memory.
  size_t memory_size_;
  // The temporary memory size. This should be dynamically determinited but is
  // predefined for now.
  size_t temp_size_;
  // Pointer to the temoprary memory;
  void *temp_memory_;
  // The size which the memory manager is allowed to use for the temp memory.
  size_t temp_memory_size_;
  // Who is using temp_memory
  size_t temp_user;
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
  size_t used_memory_;
  // Are all regular memory allocations finalzed.
  bool regular_finalized_;
};

void SA_MM_Dptr::ReadAllocationRst() {
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
}

}  // namespace storage
}  // namespace mxnet
#endif
