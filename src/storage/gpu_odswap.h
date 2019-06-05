#ifndef MXNET_STORAGE_ODSWAP_H_
#define MXNET_STORAGE_ODSWAP_H_

#include <atomic>
#include <map>
#include <memory>
#include <pthread.h>
#include <thread>
#include <semaphore.h>
#include <stack>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#if MXNET_USE_CUDA
#include <cuda_runtime.h>
#endif // MXNET_USE_CUDA
#include "./gpu_swap_history.h"
#include "./gpu_swap_memmgr.h"

namespace mxnet {

struct SwapInfo {
  handle_t handle_id;
  bool swapped_in;
  int device_id;
  void* dptr;
  char* cpu_address;
  size_t size;
  size_t swap_count;
  std::atomic_flag is_swapping;
  bool is_waiting;
};

class ODSwap {
public:
  ~ODSwap();
  static ODSwap* Get();
  static std::shared_ptr<ODSwap> _GetSharedRef();
  bool SwapOut(unsigned required_memory, int device_id, bool async);
  void SwapOutLocked(unsigned required_memory, int device_id, bool async);
  bool SwapIn(SwapInfo *info, bool async);
  void SetAddr(handle_t handle_id, void* dptr, size_t size, int device_id, bool is_pre);
  void DelAddr(handle_t handle_id);
  void FreeAddr(handle_t handle_id);
  void* GetAddr(handle_t handle_id, int type, bool& success);
  void StartComputing(const std::unordered_set<handle_t>& handles);
  void StopComputing(const std::unordered_set<handle_t>& handles);
#if 0
  void LockSwap();
  void UnlockSwap();
#endif
  void PrintHandles();

private:
  ODSwap();
  // FIXME(fegin): The design of the following two variables doesn't support
  // multiple GPUs.
  std::unordered_map<handle_t, SwapInfo*> swap_info_;
  std::unordered_set<handle_t> swappable_handles_[NUMBER_OF_GPU];
  std::unordered_map<handle_t, int> locked_handles_;
  std::map<size_t, std::unordered_set<handle_t> > divided_handles_[NUMBER_OF_GPU];
  std::shared_ptr<MemoryHistory> memory_history_;
  std::shared_ptr<MemoryManager> memory_manager_;
  pthread_rwlock_t swap_lock_;
  sem_t swap_sem_;
  bool swap_async_;
  bool infinite_memory_;
  bool infinite_cpu_memory_;
  char* fake_cpu_address_;  
  cudaStream_t streams_out_[NUMBER_OF_GPU];
  cudaStream_t streams_in_[NUMBER_OF_GPU];
}; // Class Swap

} // namespace mxnet

#endif // MXNET_STORAGE_SWAP_H_
