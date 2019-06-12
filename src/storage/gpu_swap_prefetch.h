#ifndef GPU_SWAP_PREFETCH_H
#define GPU_SWAP_PREFETCH_H

#include <mutex>
#include <queue>
#include <semaphore.h>
#include <thread>
#include <unordered_set>
#include <vector>

#if MXNET_USE_CUDA
#include <cuda_runtime.h>
#endif
#include "./gpu_odswap.h"

namespace mxnet {

class Prefetch {
public:
  ~Prefetch();
  static Prefetch* Get();
  static std::shared_ptr<Prefetch> _GetSharedRef();
  void StartPrefetching(std::pair<size_t&, size_t&> exe_cur_node);
  void StopPrefetching();
  void PushHandlesToPrefetch(const std::vector<handle_t>& handles);
  void SignalContinue();

private:
  Prefetch();
  void Prefetching(std::pair<size_t&, size_t&> exe_cur_node);

  std::thread prefetcher_;
  std::vector<std::vector<handle_t>> prefetch_sequence_;
  sem_t prefetch_sem_;
  bool prefetching_;
  bool prefetch_enabled_;

  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  std::chrono::time_point<std::chrono::high_resolution_clock> cur;
}; // class prefetch

} // namespace mxnet


#endif
