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
  void* Alloc(handle_id_t id, size_t size, void* ptr) {
    sa_log << "Alloc " << id << std::endl;
    dptr_mapping_[id] = ptr;
    dptr_size_[ptr] = size;
    return ptr;
  }

  void* Free (handle_id_t id) override {
    sa_log << "Free " << id << std::endl;
    auto it = dptr_mapping_.find(id);
    void* ptr = it->second;
    dptr_mapping_.erase(it);
    ODSwap::Get()->DelAddr(id);
    return ptr;
  }

  void Release (handle_id_t id, void* ptr) override {
    sa_log << "Release " << id << std::endl;
    dptr_mapping_[id] = ptr;
  }

  void StartAllocArgs () override { }

  void StopAllocArgs () override { }

  void StartBinding () override { MemoryHistory::Get()->StartIteration(); }

  void StopBinding () override { MemoryHistory::Get()->StopIteration(); }

  void StartIteration () override { MemoryHistory::Get()->StartIteration(); }

  void StopIteration () override { MemoryHistory::Get()->StopIteration(); }

  void Statistics () override { MemoryHistory::Get()->Statistics(); }

  void RegisterEntry (uint32_t nid, uint32_t idx, handle_id_t hid,
                      uint32_t old_nid, uint32_t old_idx, handle_id_t old_hid,
                      size_t hdl_size, bool is_var) override { }

  void FinalizeRegular() override { }

  void NotifyBegin (uint32_t nid, const std::string& name) override {
    ODSwap::Get()->PrePostAccess(true);
    Prefetch::Get()->SignalStartComputing();
  }

  void NotifyDone (uint32_t nid) override {
    ODSwap::Get()->PrePostAccess(false);
    Prefetch::Get()->SignalStopComputing();
  }

  std::vector<uint32_t> GetScheduleDeps(uint32_t nid) override {
    return std::vector<uint32_t>();
  }

  void* GetDptr (handle_id_t id) override {
    sa_log << "GetDptr " << id << std::endl;
    void* old_ptr = dptr_mapping_[id];
    dptr_mapping_[id] = ODSwap::Get()->GetAddr(id);
    dptr_size_[dptr_mapping_[id]] = dptr_size_[old_ptr];
    dptr_size_.erase(old_ptr);
    return dptr_mapping_[id];
  }

  void SetDptr (handle_id_t id, void* ptr, uint32_t dev_id) override {
    sa_log << "SetDptr " << id << " " << ptr << std::endl;
    if(ptr != nullptr && dptr_size_.find(ptr) == dptr_size_.end()) {
        LOG(FATAL) << "Can't find the size for id " << id << ".";
    }
    size_t ptr_size = 0;
    if(ptr != nullptr) {
      ptr_size = dptr_size_[ptr];
    }
    ODSwap::Get()->SetAddr(id, ptr, ptr_size, dev_id);
    dptr_mapping_[id] = ptr;
  }

 private:
  std::unordered_map<handle_id_t, void*> dptr_mapping_;
  std::unordered_map<void*, size_t> dptr_size_;
};

}  // namespace storage
}  // namespace mxnet
#endif
