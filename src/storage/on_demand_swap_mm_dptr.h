#ifndef MXNET_STORAGE_ON_DEMAND_MM_DPTR_H_
#define MXNET_STORAGE_ON_DEMAND_MM_DPTR_H_

#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <mxnet/sa_util.h>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include "./gpu_odswap.h"
#include "./gpu_swap_prefetch.h"

namespace mxnet {
namespace storage {

class OD_MM_Dptr : virtual public MM_Dptr {
 public:
  OD_MM_Dptr() {
    swap_ = ODSwap::_GetSharedRef();
    memory_manager_ = GetMemoryManagerRef();
    memory_history_ = MemoryHistory::_GetSharedRef();
    device_id_ = 0;
    temp_size_ = 0.5L * 1024 * 1024 * 1024;
    cudaError_t e = cudaMalloc(&temp_memory_, temp_size_);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
    }
  }
  ~OD_MM_Dptr() {
    cudaError_t e =  cudaFree(temp_memory_);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
    }
  }

  void* Alloc(handle_t id, size_t size, void* ptr = nullptr) {
    sa_log << "Alloc " << id << std::endl;
    size_t iteration_idx = MemoryHistory::Get()->GetIterationIdx();
    if(iteration_idx == 0) { 
      ptr = (void*)id;
      unalloced_dptrs_.insert(ptr);
      SetDptr(id, ptr, device_id_);
    } else if (iteration_idx == 1) {
      ptr = temp_memory_;
      temp_handles_.insert(id);
    } else {
      LOG(FATAL) << "Alloc after iteration 1, or not temporary: " 
      << (int)(temp_handles_.find(id) != temp_handles_.end()) << std::endl; 
    }
    dptr_mapping_[id] = ptr;
    dptr_size_[ptr] = size;
    return ptr;
  }

  void* Free(handle_t id) override {
    sa_log << "Free " << id << std::endl;
    if(temp_handles_.find(id) == temp_handles_.end()) {
      return nullptr;
    }
    auto it = dptr_mapping_.find(id);
    void* ptr = it->second;
    dptr_mapping_.erase(it);
    ODSwap::Get()->DelAddr(id);
    return ptr;
  }

  void Release (handle_t id, void* ptr) override {}

  void StartAllocArgs () override { }

  void StopAllocArgs () override { }

  void StartBinding () override { MemoryHistory::Get()->StartPreparation(); }

  void StopBinding () override { MemoryHistory::Get()->EndPreparation(); }

  void StartIteration () override { MemoryHistory::Get()->StartIteration(); }

  void StopIteration () override { MemoryHistory::Get()->StopIteration(); }

  void Statistics () override { MemoryHistory::Get()->Statistics(); }

  void RegisterEntry (node_t nid, uint32_t idx, handle_t hid,
                      node_t old_nid, uint32_t old_idx, handle_t old_hid,
                      size_t hdl_size, bool is_var, bool is_swap) override { }

  void NotifyBegin (node_t nid, const std::string& name) override {
    ODSwap::Get()->PrePostAccess(true);
    Prefetch::Get()->SignalStartComputing();
  }

  void NotifyDone (node_t nid) override {
    ODSwap::Get()->PrePostAccess(false);
    Prefetch::Get()->SignalStopComputing();
  }

  void Finish() override { }

#if 0
  std::vector<uint32_t> GetScheduleDeps(uint32_t nid) override {
    return std::vector<uint32_t>();
  }
#endif

  void* GetDptr (handle_t id) override {
    sa_log << "GetDptr " << id << std::endl;
    size_t iteration_idx = MemoryHistory::Get()->GetIterationIdx();
    if(temp_handles_.find(id) != temp_handles_.end()) {
      return temp_memory_;
    } 
    void* ptr = dptr_mapping_[id]; 
    if (iteration_idx == 1 && unalloced_dptrs_.find(ptr) != unalloced_dptrs_.end()) {
      size_t ptr_size = dptr_size_[ptr];
      void* new_ptr = Alloc_(ptr_size);
      dptr_mapping_[id] = new_ptr;
      dptr_size_[new_ptr] = dptr_size_[ptr];
      dptr_size_.erase(ptr);
      swap_->SetAddr(id, new_ptr, ptr_size, 0, false);
    } else {
      void* new_ptr = ODSwap::Get()->GetAddr(id);
      dptr_size_[new_ptr] = dptr_size_[ptr];
      dptr_size_.erase(ptr);
    }
    return dptr_mapping_[id];
  }

  void SetDptr (handle_t id, void* ptr, uint32_t dev_id) override {
    sa_log << "SetDptr " << id << " " << ptr << std::endl;
    size_t ptr_size = 0;
    if(ptr != nullptr) {
      CHECK(dptr_size_.find(ptr) != dptr_size_.end()) 
      << "Can't find the size for id " << id << ".";
      ptr_size = dptr_size_[ptr];
    }
    ODSwap::Get()->SetAddr(id, ptr, ptr_size, dev_id, true);
    dptr_mapping_[id] = ptr;
  }

 private:
  void* Alloc_(size_t ptr_size) {
    swap_->SwapOutLocked(ptr_size, device_id_, false);
    void* ret = nullptr;
    cudaError_t e = memory_manager_->Malloc(ret, ptr_size, device_id_);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
    }
    return ret;
  }

  std::shared_ptr<ODSwap> swap_;
  std::shared_ptr<MemoryManager> memory_manager_;
  std::shared_ptr<MemoryHistory> memory_history_;
  std::unordered_set<handle_t> temp_handles_;
  std::unordered_set<void*> unalloced_dptrs_;
  std::unordered_map<handle_t, void*> dptr_mapping_;
  std::unordered_map<void*, size_t> dptr_size_;
  void* temp_memory_;
  size_t temp_size_;
  int device_id_; // Only support dev_id = 0 currently.
};

}  // namespace storage
}  // namespace mxnet
#endif
