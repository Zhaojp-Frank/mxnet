#ifndef MXNET_STORAGE_ODSWAP_H_
#define MXNET_STORAGE_ODSWAP_H_

#include <atomic>
#include <map>
#include <memory>
#include <pthread.h>
#include <thread>
#include <stack>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <queue>
#include <set>
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
};


using SIGroup = std::vector<SwapInfo*>;
class SwapInfoGroups {
public:
  ~SwapInfoGroups();
  static SwapInfoGroups* Get();
  static std::shared_ptr<SwapInfoGroups> _GetSharedRef();
  void NewInfo(void* addr, SwapInfo* info);
  std::shared_ptr<SIGroup> SwapOut(void* addr);
  std::shared_ptr<SIGroup> SwapIn(void* addr, SwapInfo* info);

private:
  SwapInfoGroups();
  std::unordered_map<void*, std::shared_ptr<SIGroup>> in_memory_;
  std::unordered_map<SwapInfo*, std::shared_ptr<SIGroup>> all_;
};

class ThreadAccessInfo {
public:
  static const int kAccessThreshold = 0;
  static const unsigned kInvalidID = 2147483648;

  ~ThreadAccessInfo();
  static ThreadAccessInfo* Get();
  static std::shared_ptr<ThreadAccessInfo> _GetSharedRef();
  std::set<handle_t>& CheckAndCreate(handle_t hid, bool access, bool& running);
  handle_t Access(handle_t hid);
  void Remove(handle_t hid);

private:
  ThreadAccessInfo();
  std::unordered_map<std::thread::id, std::set<handle_t>> all_threads_;
  std::unordered_map<handle_t,
                     std::unordered_set<std::thread::id>> hid_to_threads_;
  std::unordered_map<std::thread::id, bool> is_running;
  unsigned running_threshold_;
};

class ODSwap {
public:
  ~ODSwap();
  static ODSwap* Get();
  static std::shared_ptr<ODSwap> _GetSharedRef();
  void SwapOut(unsigned required_memory, int device_id, bool async);
  void SwapOutLocked(unsigned required_memory, int device_id, bool async);
  void SwapIn(SwapInfo *info, bool async);
  void SetAddr(handle_t handle_id, void* dptr, size_t size, int device_id);
  void DelAddr(handle_t handle_id);
  void FreeAddr(handle_t handle_id);
  void* GetAddr(handle_t handle_id, bool prefetch = false);
  unsigned AccessID() {
    return access_id_.fetch_add(1, std::memory_order_relaxed);
  }
  void PrePostAccess(bool is_pre);

#if 0
  void LockSwap();
  void UnlockSwap();
#endif
  void PrintHandles();

private:
  ODSwap();
  // FIXME(fegin): The design of the following two variables doesn't support
  // multiple GPUs.
  std::shared_ptr<SwapInfoGroups> swapinfo_groups_;
  std::shared_ptr<ThreadAccessInfo> thread_info_;
  std::unordered_map<handle_t, SwapInfo*> swap_info_;
  std::unordered_set<handle_t> swappable_handles_[NUMBER_OF_GPU];
  std::map<size_t, std::unordered_set<handle_t> > divided_handles_[NUMBER_OF_GPU];
  std::stack<handle_t> locked_handles_[NUMBER_OF_GPU];
  std::vector<size_t> free_memory_;
  std::shared_ptr<MemoryHistory> memory_history_;
  std::shared_ptr<MemoryManager> memory_manager_;
  std::atomic<unsigned> access_id_;
  pthread_rwlock_t swap_lock_;
  pthread_rwlock_t locks_[NUMBER_OF_GPU];
  bool swap_async_;
  bool infinite_memory_;
  cudaStream_t streams_out_[NUMBER_OF_GPU];
  cudaStream_t streams_in_[NUMBER_OF_GPU];
}; // Class Swap

} // namespace mxnet

#endif // MXNET_STORAGE_SWAP_H_
