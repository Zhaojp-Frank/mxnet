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
  num_loop_ = dmlc::GetEnv("MXNET_NUM_LOOP", 12);
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

void Prefetch::StartPrefetching(size_t iteration_idx, size_t node_idx) {
  if (!prefetch_enabled_) {
    return;
  }
  /*
  start = std::chrono::high_resolution_clock::now();
  */
  prefetch_iteration_idx_ = execution_iteration_idx_ = iteration_idx;
  prefetch_node_idx_ = execution_node_idx_ = node_idx;
  std::cout << "Prefetch: Start Prefetching" << std::endl;
  prefetching_ = true;
  prefetcher_ = std::thread(&Prefetch::Prefetching, this);
}

void Prefetch::StopPrefetching(size_t iteration_idx) {
  if (!prefetch_enabled_ || iteration_idx != num_loop_) {
    return;
  }
  std::cout << "Prefetch thread is stopped" << std::endl;
  prefetching_ = false;
  prefetcher_.join();
}

void Prefetch::Prefetching() {
  bool success;
  size_t cur_idx_in_node = 0;
  while (prefetching_) {
    // Make sure prefetch is not slower than getaddr in terms of node.
    prefetch_mut_.lock();
    if (std::make_pair(prefetch_iteration_idx_, prefetch_node_idx_) 
        < std::make_pair(execution_iteration_idx_, execution_node_idx_)) {
      sa_log << "Prefetch: slower than execution, fast forward to " 
             << execution_iteration_idx_ << " " << execution_node_idx_ << std::endl;
      prefetch_iteration_idx_ = execution_iteration_idx_;
      prefetch_node_idx_ = execution_node_idx_;
      cur_idx_in_node = 0;
    }
    bool prefetch_ahead = prefetch_iteration_idx_ > execution_iteration_idx_;
    prefetch_mut_.unlock();
    if (prefetch_node_idx_ >= prefetch_sequence_.size()) {
      sa_log << "Prefetch: End of iteration" << std::endl;
      if (prefetch_node_idx_ > prefetch_sequence_.size()) {
        sa_log << "Prefetch: cur idx > total history (CHECK) " << std::endl;
      }
      /*
      end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end-start;
      sa_log << "Prefetch stops at: " << diff.count() << " s" << std::endl;
      */
      prefetch_iteration_idx_ ++;
      prefetch_node_idx_ = 0;
      cur_idx_in_node = 0;
    }
    if (prefetch_iteration_idx_ > num_loop_) {
      sa_log << "Prefetch: End of prefetch thread" << std::endl;
      break;
    }
    // Lock the node being prefetched
    if (cur_idx_in_node == 0) {
      sa_log << "Prefetch: Locking node idx = " << prefetch_node_idx_ << std::endl;
      CHECK(prefetch_node_idx_ < prefetch_sequence_.size());
      ODSwap::Get()->LockHandles(prefetch_sequence_[prefetch_node_idx_],
          prefetch_node_idx_);
    }
    // Prefetching Handle
    sa_log << "Prefetch: prefetching (" << prefetch_node_idx_ << "," << cur_idx_in_node 
           << ")" << std::endl;
    CHECK(prefetch_node_idx_ < prefetch_sequence_.size());
    CHECK(cur_idx_in_node < prefetch_sequence_[prefetch_node_idx_].size());
    sa_log << "Prefetch: prefetching " 
           << prefetch_sequence_[prefetch_node_idx_][cur_idx_in_node] << std::endl;
    ODSwap::Get()->GetAddr(prefetch_sequence_[prefetch_node_idx_][cur_idx_in_node],
        true, success, prefetch_ahead);
    sa_log << "Prefetch: " << (success?"success":"failure") << std::endl;
    if (!success) {
      sa_log << "Prefetch failed " << std::endl;
      sem_wait(&prefetch_sem_);
    } else {
      cur_idx_in_node++;
      if (cur_idx_in_node == prefetch_sequence_[prefetch_node_idx_].size()) {
        sa_log << "Prefetch: End of node with index " << prefetch_node_idx_ << std::endl;
        /*
        cur = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = cur-start;
        sa_log << "Prefetch: time now: " <<  diff.count() << " s" << std::endl;
        */
        cur_idx_in_node = 0;
        prefetch_node_idx_ ++;
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


void Prefetch::SignalContinue(size_t iteration_idx, size_t node_idx) {
  if (!prefetch_enabled_) {
    return;
  }
  prefetch_mut_.lock();
  sa_log << "Prefetch: SignalContinue" << std::endl;
  execution_iteration_idx_ = iteration_idx;
  execution_node_idx_ = node_idx;
  prefetch_mut_.unlock();
  sem_post(&prefetch_sem_);
}

} // namespace mxnet
