#ifndef MXNET_STORAGE_POOLED_MM_DPTR_H_
#define MXNET_STORAGE_POOLED_MM_DPTR_H_

#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <unordered_map>
#include <algorithm>
#include <vector>

namespace mxnet {
namespace storage {

class Pooled_MM_Dptr : virtual public MM_Dptr {
 public:
  void* Alloc(handle_id_t id, size_t size, void* ptr) override {
#if SWAP_ADVISOR_FLOW_TRACE
    std::cout << "Alloc " << id << std::endl;
#endif
    dptr_mapping_[id] = ptr;
    return ptr;
  }

  void* Free(handle_id_t id) override {
#if SWAP_ADVISOR_FLOW_TRACE
    std::cout << "Free " << id << std::endl;
#endif
    auto it = dptr_mapping_.find(id);
    void* ptr = it->second;
    dptr_mapping_.erase(it);
    return ptr;
  }

  void Release(handle_id_t id, void* ptr) override {
#if SWAP_ADVISOR_FLOW_TRACE
    std::cout << "Release " << id << std::endl;
#endif
    dptr_mapping_[id] = ptr;
  }

  void StartAllocArgs() override { }

  void StopAllocArgs() override { }

  void StartBinding() override { }

  void StopBinding() override { }

  void StartIteration() override { }

  void StopIteration() override { }

  void RegisterEntry(uint32_t nid, uint32_t idx, handle_id_t hid,
                     uint32_t old_nid, uint32_t old_idx, handle_id_t old_hid,
                     size_t hdl_size, bool is_var) override { }

  void FinalizeRegular() override { }

  void NotifyBegin(uint32_t nid, const std::string& name) override { }

  void NotifyDone(uint32_t nid) override { }

  std::vector<uint32_t> GetScheduleDeps(uint32_t nid) override {
    return std::vector<uint32_t>();
  }

  void* GetDptr(handle_id_t id) override {
#if SWAP_ADVISOR_FLOW_TRACE
    std::cout << "GetDptr " << id << std::endl;
#endif
    return dptr_mapping_.at(id);
  }

  void SetDptr(handle_id_t id, void* ptr, uint32_t dev_id) override {
#if SWAP_ADVISOR_FLOW_TRACE
    std::cout << "SetDptr " << id << std::endl;
#endif
    dptr_mapping_[id] = ptr;
  }

 private:
  std::unordered_map<handle_id_t, void*> dptr_mapping_;
};

}  // namespace storage
}  // namespace mxnet
#endif
