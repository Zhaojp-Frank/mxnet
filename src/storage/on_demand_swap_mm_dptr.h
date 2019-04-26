#ifndef MXNET_STORAGE_ON_DEMAND_MM_DPTR_H_
#define MXNET_STORAGE_ON_DEMAND_MM_DPTR_H_

#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include "./gpu_swap.h"
#define SWAP_ADVISOR_FLOW_TRACE 1

namespace mxnet {
namespace storage {

class OD_MM_Dptr : virtual public MM_Dptr {
 public:
  void* Alloc(handle_id_t id, size_t size, void* ptr) {
#if SWAP_ADVISOR_FLOW_TRACE
    std::cout << "Alloc " << id << std::endl;
#endif
    dptr_size_[id] = size;
    dptr_mapping_[id] = ptr;
    return ptr;
  }

  void* Free(handle_id_t id) {
#if SWAP_ADVISOR_FLOW_TRACE
    std::cout << "Free " << id << std::endl;
#endif
    auto it = dptr_mapping_.find(id);
    void* ptr = it->second;
    dptr_mapping_.erase(it);
    Swap::Get()->DelAddr(id);
    return ptr;
  }

  void Release(handle_id_t id, void* ptr) {
#if SWAP_ADVISOR_FLOW_TRACE
    std::cout << "Release " << id << std::endl;
#endif
    dptr_mapping_[id] = ptr;
  }

  void RegisterEntry(uint32_t nid, uint32_t idx, handle_id_t hid, bool is_var) {
    // Nothing to do for od_mm_dptr.
    return;
  }

  void* GetDptr(handle_id_t id) {
#if SWAP_ADVISOR_FLOW_TRACE
    std::cout << "GetDptr " << id << std::endl;
#endif
    return dptr_mapping_[id] = Swap::Get()->GetAddr(id);
  }

  void SetDptr(handle_id_t id, void* ptr, uint32_t dev_id) {
#if SWAP_ADVISOR_FLOW_TRACE
    std::cout << "SetDptr " << id << std::endl;
#endif
    if(dptr_size_.find(id) == dptr_size_.end()) {
        LOG(FATAL) << "Can't find the size for id " << id << ".";
    }
    Swap::Get()->SetAddr(id, ptr, dptr_size_[id], dev_id);
    dptr_mapping_[id] = ptr;
  }

 private:
  std::unordered_map<handle_id_t, void*> dptr_mapping_;
  std::unordered_map<handle_id_t, size_t> dptr_size_;
};

}  // namespace storage
}  // namespace mxnet
#endif
