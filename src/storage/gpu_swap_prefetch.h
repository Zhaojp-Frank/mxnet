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
#include "./gpu_swap_history.h"

namespace mxnet {

class Prefetch {
public:
  ~Prefetch();
  static Prefetch* Get();
  static std::shared_ptr<Prefetch> _GetSharedRef();
  void StartPrefetching();
  void StopPrefetching();
  void PushHandlesToPrefetch(const std::vector<handle_t>& handles);
  void SignalContinue();

private:
  Prefetch();
  void Prefetching();

  std::thread prefetcher_;
  std::size_t cur_node_idx_;
  std::size_t cur_idx_in_node_;
  std::vector<std::vector<handle_t>> prefetch_sequence_;
  sem_t prefetch_sem_;
  bool prefetching_;
  bool prefetch_enabled_;
  size_t num_loop_;

  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  std::chrono::time_point<std::chrono::high_resolution_clock> cur;
}; // class prefetch

} // namespace mxnet


#endif
