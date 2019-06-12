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
  num_loop_ = dmlc::GetEnv("MXNET_NUM_LOOP", 10); 
  prefetching_ = false;
  sem_init(&prefetch_sem_, 0, 1);
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

void Prefetch::StartPrefetching(pair<size_t&, size_t&> exe_cur_node) {
  start = std::chrono::high_resolution_clock::now();
  if (!prefetch_enabled_) {
    return;
  }
  sa_log << "Prefetch: Start Prefetching" << std::endl;
  prefetching_ = true;
  cur_node_idx_ = cur_idx_in_node = 0;
  prefetcher_ = std::thread(&Prefetch::Prefetching, exe_cur_node);
  sa_log << "Prefetch: Start Prefetching, thread created" << std::endl;
}

void Prefetch::StopPrefetching() {
  if (!prefetch_enabled_) {
    return;
  }
  prefetching_ = false;
  prefetcher_.join();
}

void Prefetch::Prefetching(pair<size_t&, size_t&> exe_cur_node) {
  bool success;
  pair<size_t, size_t> pre_cur_node = exe_cur_node;
  size_t cur_idx_in_node = 0;
  while (prefetching_) {
    sa_log << "Prefetch: prefetching " 
           << prefetch_sequence_[pre_cur_node_idx_][cur_idx_in_node_] << std::endl;
    ODSwap::Get()->GetAddr(prefetch_sequence_[pre_cur_node_idx_][cur_idx_in_node_],
        true, success);
    sa_log << "Prefetch: " << (success?"success":"failure") << std::endl;
    if (!success) {
      std::cout << "Prefetch failed " << std::endl;
      sem_wait(&prefetch_sem_);
    } else {
      cur_idx_in_node_++;
      if (cur_idx_in_node_ == prefetch_sequence_[pre_cur_node.second].size()) {
        sa_log << "Prefetch: End of node with index " << pre_cur_node.second << std::endl;
        cur = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = cur-start;
        sa_log << "Prefetch: time now: " <<  diff.count() << " s" << std::endl;
        cur_idx_in_node_ = 0;
        pre_cur_node.second ++;
        
      }
    } // if (!success)
    // Make sure prefetch is now slower than getaddr in terms of node.
    if (pre_cur_node <= exe_cur_node) {
      pre_cur_node = exe_cur_node;
      pre_cur_node.second++;
    }
    if (pre_cur_node.second == prefetch_sequence_.size()) {
      sa_log << "Prefetch: End of iteration" << std::endl;
      end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end-start;
      sa_log << "Prefetch stops at: " << diff.count() << " s" << std::endl;
      pre_cur_node.first ++;
      pre_cur_node.second = 0;
      cur_idx_in_node_ = 0;
    }
    if (cur_idx_in_node_ == 0) {
      ODSwap::Get()->LockHandles(prefetch_sequence_[pre_cur_node.second],
          pre_cur_node.second);
    }
  } // While prefetching_
}

void Prefetch::SignalContinue(size_t exe_cur_node_idx) {
  if (!prefetch_enabled_) {
    return;
  }
  sa_log << "Prefetch: SignalContinue" << std::endl;
  sem_post(&prefetch_sem_);
}

} // namespace mxnet
