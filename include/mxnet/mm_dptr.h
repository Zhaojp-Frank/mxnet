#ifndef MXNET_MM_DPTR_H_
#define MXNET_MM_DPTR_H_


namespace mxnet {
namespace storage {

using handle_id_t = unsigned long long;
// An interface to intercept memory access/allocation/free. All memmgrs should
// have a corresponding object inherited from MM_Dptr.
class MM_Dptr {
 public:
  // Allocate an address for the handle id.
  // This function should be used exclusively for storage managers.
  // The ptr parameter is used for pooled_storage_manager since we don't want to
  // dramatically change pooled_storage_manager.
  // (Should on-demand swap memory manager also be the same implementation?)
  virtual void* Alloc(handle_id_t id, size_t size, void* ptr) = 0;

  // Free an address captured by the handle.
  // This function should be used exclusively for storage managers.
  virtual void* Free(handle_id_t id) = 0;

  // This is almost the same as SetDptr() but the memory will be immeidately
  // released by the storage manager. See ReleaseAll() in
  // pooled_storage_manager(). We create this so that we can know that the
  // setting address is called from ReleaseAll().
  // This function should be used exclusively for storage managers.
  virtual void  Release(handle_id_t id, void* ptr) = 0;

  // StartBinding()
  virtual void  StartBinding() = 0;

  // StopBinding()
  virtual void  StopBinding() = 0;

  // StartIteration()
  virtual void  StartIteration() = 0;

  // StopIteration()
  virtual void  StopIteration() = 0;

  // Register the mapping from an entry to a handle. The mapping is
  // many-(entry)-to-one(handle).
  virtual void  RegisterEntry(uint32_t nid, uint32_t idx, handle_id_t hid,
                              uint32_t old_nid, uint32_t old_idx,
                              handle_id_t old_hid, size_t hdl_size,
                              bool is_var) = 0;

  // Let the manager know that all regular tensor allocations are finished.
  // All rest memory allocations will be temporary memory allocations.
  virtual void FinalizeRegular() = 0;

  // Can be called by both handles and storage managers.
  virtual void* GetDptr(handle_id_t id) = 0;

  // Can be called by both handles and storage managers.
  virtual void  SetDptr(handle_id_t id, void* ptr, uint32_t dev_id) = 0;
};  // class MM_Dptr

MM_Dptr* MM_DPTR();

}   // namespace storage
}   // namespace mxnet
#endif // MXNET_MM_DPTR_H_
