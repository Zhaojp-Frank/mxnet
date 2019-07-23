#ifndef MXNET_STORAGE_POOLED_MM_DPTR_H_
#define MXNET_STORAGE_POOLED_MM_DPTR_H_

#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <mxnet/sa_util.h>
#include <unordered_map>
#include <algorithm>
#include <vector>

namespace mxnet {
namespace storage {

class Pooled_MM_Dptr : virtual public MM_Dptr {
 public:
  Pooled_MM_Dptr () {
    alloc_finalized_ = false;
  }

  bool AllocFinished () { return alloc_finalized_; }

  void* Alloc(handle_t id, size_t size, void* ptr) override {
    sa_log << "Alloc " << id << std::endl; dptr_mapping_[id] = ptr;
    return ptr;
  }

  void* Free(handle_t id) override {
    sa_log << "Free " << id << std::endl;
    auto it = dptr_mapping_.find(id);
    void* ptr = it->second;
    dptr_mapping_.erase(it);
    return ptr;
  }

  void Release(handle_t id, void* ptr) override {
    sa_log << "Release " << id << std::endl;
    dptr_mapping_[id] = ptr;
  }

  void StartAllocArgs() override { }

  void StopAllocArgs() override { }

  void StartBinding() override { }

  void StopBinding() override { alloc_finalized_ = true; }

  void StartIteration() override { }

  void StopIteration() override { }

  void Statistics () override { }

  void FakeContextSwitch() override { }

  void RegisterEntry(node_t nid, uint32_t idx, handle_t hid,
                     node_t old_nid, uint32_t old_idx, handle_t old_hid,
                     size_t hdl_size, bool is_var, bool is_swap) override { }

  void NotifyBegin(uint32_t nid, const std::string& name) override { }

  void NotifyDone(uint32_t nid) override { }

  void Finish() override { }

#if 0
  std::vector<uint32_t> GetScheduleDeps(node_id nid) override {
    return std::vector<uint32_t>();
  }
#endif

  void* GetDptr(handle_t id) override {
    sa_log << "GetDptr " << id << std::endl;
    return dptr_mapping_.at(id);
  }

  void SetDptr(handle_t id, void* ptr, uint32_t dev_id) override {
    sa_log << "SetDptr " << id << std::endl;
    dptr_mapping_[id] = ptr;
  }

 private:
  std::unordered_map<handle_t, void*> dptr_mapping_;
  bool alloc_finalized_;
};

Pooled_MM_Dptr* POOLED_MM_DPTR();

}  // namespace storage
}  // namespace mxnet
#endif
