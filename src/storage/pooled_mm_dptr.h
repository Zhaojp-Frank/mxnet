#ifndef MXNET_STORAGE_POOLED_MM_DPTR_H_
#define MXNET_STORAGE_POOLED_MM_DPTR_H_

#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <unordered_map>
#include <algorithm>
#include <vector>

namespace mxnet {
namespace storage {

class Pooled_MM_Dptr : public MM_Dptr {
 public:
  void* Alloc(handle_id_t id, size_t size, void* ptr) {
    dptr_mapping_[id] = ptr;
    return ptr;
  }

  void* Free(handle_id_t id) {
    auto it = dptr_mapping_.find(id);
    void* ptr = it->second;
    dptr_mapping_.erase(it);
    return ptr;
  }

  void Release(handle_id_t id, void* ptr) {
    dptr_mapping_[id] = ptr;
  }

  void* GetDptr(handle_id_t id) {
    return dptr_mapping_.at(id);
  }

  void SetDptr(handle_id_t id, void* ptr, uint32_t dev_id) {
    dptr_mapping_[id] = ptr;
  }

 private:
  std::unordered_map<handle_id_t, void*> dptr_mapping_;
};

}  // namespace storage
}  // namespace mxnet
#endif
