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
  SA_MM_Dptr();

  void* Alloc_(handle_id_t id) {
    std::cout << "Alloc_ " << id << std::endl;
    uint32_t mempool = hdl_to_mempool_.at(id);
    void* address = mempools_.at(mempool).back();
    mempools_.at(mempool).pop_back();
    used_mempools_[mempool][address] = id;
    hdl_dptr_mapping_[id] = address;
    return address;
  }

  void Free_(handle_id_t id) {
    uint32_t mempool = hdl_to_mempool_.at(id);
    auto it = hdl_dptr_mapping_.find(id);
    void* address = it->second;
    hdl_dptr_mapping_.erase(it);
    CHECK_EQ(used_mempools_[mempool].erase(address), 1);
    mempools_.at(mempool).push_back(address);
  }

  // Do a random free then alloc. This should only be used during the binding
  // process.
  void* FreeAlloc_(handle_id_t id) {
    std::cout << "FreeAlloc_ " << id << std::endl;
    CHECK(!iteration_started);
    CHECK_EQ(curr_iteration, -1);
    uint32_t mempool = hdl_to_mempool_.at(id);
    auto it = used_mempools_[mempool].begin();
    void* address = it->first;
    CHECK_EQ(hdl_dptr_mapping_.erase(it->second), 1);
    it->second = id;
    hdl_dptr_mapping_[id] = address;
    return address;
  }

  void* Alloc(handle_id_t id, size_t size, void* ptr);

  void* Free(handle_id_t id) {
    std::cout << "SA_MM_Dptr Free" << std::endl;
    if (id == temp_user_) {
      temp_user_ = 0;
      CHECK_EQ(hdl_dptr_mapping_.erase(id), 1);
    } else {
      Free_(id);
    }
    return nullptr;
  }

  void Release(handle_id_t id, void* ptr) {
    std::cout << "SA_MM_Dptr Release" << std::endl;
    // No need to implement for this manager;
    CHECK(0);
  }

  void StartBinding() {
    std::cout << "StartBinding " << std::endl;
    // Nothing to do for swapadv_mm_dptr.
    return;
  }

  void StopBinding() {
    std::cout << "StopBinding " << std::endl;
    regular_finalized_ = true;
    return;
  }

  void StartIteration();

  void StopIteration() {
    std::cout << "StopIteration " << std::endl;
    iteration_started = false;
    return;
  }

  void RegisterEntry(uint32_t nid, uint32_t idx, handle_id_t hid,
                     uint32_t old_nid, uint32_t old_idx, handle_id_t old_hid,
                     size_t hdl_size, bool is_var) {
    uint32_t eid = nid * hash_const + idx;
    entry_hdl_mapping_[eid] = std::make_pair(hid, is_var);
    new_to_old_hids_[hid] = old_hid;
  }

  void FinalizeRegular() {
    std::cout << "SA_MM_Dptr FinalizeRegular" << std::endl;
    regular_finalized_ = true;
  }

  void* Swapin(handle_id_t id);

  void* GetDptr(handle_id_t id) {
    std::cout << "SA_MM_Dptr GetDptr " << id << std::endl;
    void* address = nullptr;
    auto it = hdl_dptr_mapping_.find(id);
    if (it != hdl_dptr_mapping_.end()) {
      address = it->second;
    } else {
      if (regular_finalized_) {
        address = Swapin(id);
      } else {
        address = FreeAlloc_(id);
      }
    }
    std::cout << "SA_MM_Dptr GetDptr done " << address << std::endl;
    return address;
  }

  void SetDptr(handle_id_t id, void* ptr, uint32_t dev_id) {
    if (ptr == nullptr) return;
    std::cout << "SA_MM_Dptr SetDptr " << id << " " << ptr << std::endl;
    for (auto& used_mempool : used_mempools_) {
      auto it = used_mempool.find(ptr);
      if (it != used_mempool.end()) {
        size_t size = hdl_size_mapping_.at(it->second);
        uint32_t mempool = hdl_to_mempool_.at(it->second);
        hdl_size_mapping_[id] = size;
        hdl_to_mempool_[id] = mempool;
        break;
      }
    }
    CHECK_EQ(hdl_size_mapping_.count(id), 1);
    CHECK_EQ(hdl_to_mempool_.count(id), 1);
    hdl_dptr_mapping_[id] = ptr;
  }

 private:
  //
  std::unordered_map<handle_id_t, handle_id_t> new_to_old_hids_;
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
  // Initial handles
  std::vector<handle_id_t> initial_handles_;
  // Iterations info
  bool iteration_started;
  int curr_iteration;

  static constexpr uint32_t hash_const = 33;
  // Read the memory allocation result from the SwapAdvisor.
  void ReadAllocationRst();
  // Read the initial handle allocation result from the SwapAdvisor.
  void ReadInitialHandlesRst();
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
  size_t temp_user_;
  // Memory pool index to memory pool size.
  std::vector<size_t> mempool_to_size_;
  // Memory pool objects count
  std::vector<size_t> mempool_counts_;
  // The mapping from real sizes to memory pool index.
  std::unordered_map<size_t, size_t> rsize_to_mempool_;
  // The mapping from hdl to memory pool index.
  std::unordered_map<handle_id_t, size_t> hdl_to_mempool_;
  // Memory pools
  std::vector<std::vector<void*>> mempools_;
  // Used memory pools
  std::vector<std::unordered_map<void*, uint32_t>> used_mempools_;
  // Used memory
  size_t used_memory_;
  // Are all regular memory allocations finalzed.
  bool regular_finalized_;
};

}  // namespace storage
}  // namespace mxnet
#endif
