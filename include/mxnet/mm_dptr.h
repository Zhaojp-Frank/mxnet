#ifndef MXNET_MM_DPTR_H_
#define MXNET_MM_DPTR_H_

#include<vector>

namespace mxnet {
namespace storage {

using handle_t = unsigned long;
using node_t = uint32_t;

// An interface to intercept memory access/allocation/free. All memmgrs should
// have a corresponding object inherited from MM_Dptr.
class MM_Dptr {
 public:
  // Allocate an address for the handle id.
  // This function should be used exclusively for storage managers.
  // The ptr parameter is used for pooled_storage_manager since we don't want to
  // dramatically change pooled_storage_manager.
  // (Should on-demand swap memory manager also be the same implementation?)
  virtual void* Alloc(handle_t id, size_t size, void* ptr) = 0;

  // Free an address captured by the handle.
  // This function should be used exclusively for storage managers.
  virtual void* Free(handle_t id) = 0;

  // This is almost the same as SetDptr() but the memory will be immeidately
  // released by the storage manager. See ReleaseAll() in
  // pooled_storage_manager(). We create this so that we can know that the
  // setting address is called from ReleaseAll().
  // This function should be used exclusively for storage managers.
  virtual void  Release(handle_t id, void* ptr) = 0;

  // StartAllocArgs()
  virtual void  StartAllocArgs() = 0;

  // StopAllocArgs()
  virtual void  StopAllocArgs() = 0;

  // StartBinding()
  virtual void  StartBinding() = 0;

  // StopBinding()
  virtual void  StopBinding() = 0;

  // StartIteration()
  virtual void  StartIteration() = 0;

  // StopIteration()
  virtual void  StopIteration() = 0;

  virtual void Statistics () = 0;

  // Register the mapping from an entry to a handle. The mapping is
  // many-(entry)-to-one(handle).
  virtual void  RegisterEntry(node_t nid, uint32_t idx, handle_t hid,
                              node_t old_nid, uint32_t old_idx,
                              handle_t old_hid, size_t hdl_size,
                              bool is_var, bool is_swap) = 0;

  virtual void NotifyBegin(node_t nid, const std::string& name) = 0 ;

  // Notify mm_dptr that a node is executed.
  virtual void NotifyDone(node_t nid) = 0;

  //virtual std::vector<node_t> GetScheduleDeps(node_t nid) = 0;


  // Can be called by both handles and storage managers.
  virtual void* GetDptr(handle_t id) = 0;

  // Can be called by both handles and storage managers.
  virtual void  SetDptr(handle_t id, void* ptr, uint32_t dev_id) = 0;
};  // class MM_Dptr

MM_Dptr* MM_DPTR();

}   // namespace storage
}   // namespace mxnet
#endif // MXNET_MM_DPTR_H_
