#ifndef MXNET_STORAGE_SWAPADVISOR_STORAGE_MANAGER_H_
#define MXNET_STORAGE_SWAPADVISOR_STORAGE_MANAGER_H_

#if MXNET_USE_CUDA
  #include <cuda_runtime.h>
#endif  // MXNET_USE_CUDA
#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <mutex>
#include <new>
#include <fstream>
#include <mxnet/mm_dptr.h>
#include "./storage_manager.h"
#include "../common/cuda_utils.h"
#include "../common/utils.h"


namespace mxnet {
namespace storage {

#if MXNET_USE_CUDA
/*!
 * \brief Storage manager with a memory pool on gpu. Memory chunks are reused based on exact size
 * match.
 */
class GPUSwapAdvStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  GPUSwapAdvStorageManager() {}
  /*!
   * \brief Default destructor.
   */
  ~GPUSwapAdvStorageManager() {}
  void Alloc(Storage::Handle* handle) {
    MM_DPTR()->Alloc(handle->ID(), handle->size, nullptr);
  }
  void Free(Storage::Handle handle) { MM_DPTR()->Free(handle.ID()); }
  void DirectFree(Storage::Handle handle) { MM_DPTR()->Free(handle.ID()); }
  DISALLOW_COPY_AND_ASSIGN(GPUSwapAdvStorageManager);
};  // class GPUSwapAdvStorageManager


#endif  // MXNET_USE_CUDA

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_GPU_SWAPADVISOR_STORAGE_MANAGER_H_
