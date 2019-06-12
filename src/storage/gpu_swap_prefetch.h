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
#include "./on_demand_swap_mm_dptr.h"

namespace mxnet {

class Prefetch {
public:
  ~Prefetch();
  static Prefetch* Get();
  static std::shared_ptr<Prefetch> _GetSharedRef();
  void StartPrefetching(pair<size_t&, size_t&> exe_cur_node);
  void StopPrefetching();
  void PushHandlesToPrefetch(pair<size_t&, size_t&> exe_cur_node);
  void SignalContinue();

private:
  Prefetch();
  void Prefetching();

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
