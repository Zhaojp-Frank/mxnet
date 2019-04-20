#ifndef MXNET_MM_DPTR_H_
#define MXNET_MM_DPTR_H_

using handle_id_t = unsigned long long;

namespace mxnet {

// An interface to intercept memory access/allocation/free. All memmgrs should
// have a corresponding object inherited from MM_Dptr.
class MM_Dptr {
 public:
  // Allocate an address for the handle id.
  // This function should be used exclusively for storage managers.
  // The ptr parameter is used for pooled_storage_manager since we don't want to
  // dramatically change pooled_storage_manager.
  // (Should on-demand swap memory manager also be the same implementation?)
  virtual void* Alloc(handle_id_t id, void* ptr);

  // Free an address captured by the handle.
  // This function should be used exclusively for storage managers.
  virtual void* Free(handle_id_t id);

  // This is almost the same as SetDptr() but the memory will be immeidately
  // released by the storage manager. See ReleaseAll() in
  // pooled_storage_manager(). We create this so that we can know that the
  // setting address is called from ReleaseAll().
  // This function should be used exclusively for storage managers.
  virtual void  Release(handle_id_t id, void* ptr);

  // Can be called by both handles and storage managers.
  virtual void* GetDptr(handle_id_t id);

  // Can be called by both handles and storage managers.
  virtual void  SetDptr(handle_id_t id, void* ptr, uint32_t dev_id);
};  // class MM_Dptr
}   // namespace mxnet

MM_Dptr* MM_DPTR();

#endif // MXNET_MM_DPTR_H_
