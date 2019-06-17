#include <iostream>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <map>
#include <mxnet/sa_util.h>
#include <dmlc/parameter.h>
#include <dmlc/logging.h>
#include "./gpu_odswap.h"
#include "./gpu_swap_history.h"
#include "./gpu_swap_prefetch.h"

#include <chrono>


namespace mxnet {

Prefetch::Prefetch() {
  prefetch_enabled_ = dmlc::GetEnv("MXNET_ENABLE_PREFETCH", false);
  std::cout << "Prefetch enabled: " << (prefetch_enabled_?1:0) << std::endl;
  prefetching_ = false;
  sem_init(&prefetch_sem_, 0, 1);
  num_loop_ = dmlc::GetEnv("MXNET_NUM_LOOP", 10);
}

Prefetch::~Prefetch() {}

Prefetch* Prefetch::Get() {
  static Prefetch *s = _GetSharedRef().get();
  return s;
}

std::shared_ptr<Prefetch> Prefetch::_GetSharedRef() {
  static std::shared_ptr<Prefetch> inst(new Prefetch());
  return inst;
}

void Prefetch::StartPrefetching(std::pair<size_t&, size_t&> exe_cur_node) {
  if (!prefetch_enabled_) {
    return;
  }
  /*
  start = std::chrono::high_resolution_clock::now();
  */
  sa_log << "Prefetch: Start Prefetching" << std::endl;
  prefetching_ = true;
  prefetcher_ = std::thread(&Prefetch::Prefetching, this, exe_cur_node);
  sa_log << "Prefetch: Start Prefetching, thread created" << std::endl;
}

void Prefetch::StopPrefetching(size_t iteration_idx) {
  if (!prefetch_enabled_ || iteration_idx != num_loop_) {
    return;
  }
  prefetching_ = false;
  prefetcher_.join();
}

void Prefetch::Prefetching(std::pair<size_t&, size_t&> exe_cur_node) {
  bool success;
  std::pair<size_t, size_t> pre_cur_node = exe_cur_node;
  size_t cur_idx_in_node = 0;
  while (prefetching_) {
    // Make sure prefetch is not slower than getaddr in terms of node.
    if (pre_cur_node <= std::make_pair(exe_cur_node.first, exe_cur_node.second)) {
      sa_log << "Prefetch: slower than execution, fast forward to " 
             << exe_cur_node.first << " " << exe_cur_node.second << std::endl;
      pre_cur_node = exe_cur_node;
      pre_cur_node.second ++;
      cur_idx_in_node = 0;
    }
    if (pre_cur_node.second == prefetch_sequence_.size()) {
      sa_log << "Prefetch: End of iteration" << std::endl;
      /*
      end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end-start;
      sa_log << "Prefetch stops at: " << diff.count() << " s" << std::endl;
      */
      pre_cur_node.first ++;
      pre_cur_node.second = 0;
      cur_idx_in_node = 0;
    }
    if (pre_cur_node.first > num_loop_) {
      sa_log << "Prefetch: End of prefetch thread" << std::endl;
      break;
    }
    // Lock the node being prefetched
    if (cur_idx_in_node == 0) {
      sa_log << "Prefetch: Locking node idx = " << pre_cur_node.second << std::endl;
      ODSwap::Get()->LockHandles(prefetch_sequence_[pre_cur_node.second],
          pre_cur_node.second);
    }
    // Prefetching Handle
    sa_log << "Prefetch: prefetching (" << pre_cur_node.second << "," << cur_idx_in_node 
           << ")" << std::endl;
    sa_log << "Prefetch: prefetching " 
           << prefetch_sequence_[pre_cur_node.second][cur_idx_in_node] << std::endl;
    ODSwap::Get()->GetAddr(prefetch_sequence_[pre_cur_node.second][cur_idx_in_node],
        true, success, pre_cur_node.first > exe_cur_node.first);
    sa_log << "Prefetch: " << (success?"success":"failure") << std::endl;
    if (!success) {
      std::cout << "Prefetch failed " << std::endl;
      sem_wait(&prefetch_sem_);
    } else {
      cur_idx_in_node++;
      if (cur_idx_in_node == prefetch_sequence_[pre_cur_node.second].size()) {
        sa_log << "Prefetch: End of node with index " << pre_cur_node.second << std::endl;
        /*
        cur = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = cur-start;
        sa_log << "Prefetch: time now: " <<  diff.count() << " s" << std::endl;
        */
        cur_idx_in_node = 0;
        pre_cur_node.second ++;
        
      }
    } // if (!success)
    
  } // While prefetching_
}

void Prefetch::PushHandlesToPrefetch(const std::vector<handle_t>& handles) {
  if (!prefetch_enabled_) {
    return;
  }
  prefetch_sequence_.push_back(std::vector<handle_t>{});  
  sa_log << "Prefetch: push handle for node " << prefetch_sequence_.size() 
    << std::endl;
  auto& cur_subseq = prefetch_sequence_[prefetch_sequence_.size()-1];
  for (auto handle: handles) {
    sa_log << "Prefetch: Add handle " << handle << " to prefetch sequence"
      << std::endl;
    cur_subseq.push_back(handle);
  }
}


void Prefetch::SignalContinue() {
  if (!prefetch_enabled_) {
    return;
  }
  sa_log << "Prefetch: SignalContinue" << std::endl;
  sem_post(&prefetch_sem_);
}

} // namespace mxnet
