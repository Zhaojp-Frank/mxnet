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
  cur_node_idx_ = cur_idx_in_node_ = 0;
  prefetching_ = false;
  sem_init(&prefetch_sem_, 0, 1);
  history_ = MemoryHistory::_GetSharedRef();
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

void Prefetch::StartPrefetching() {
  start = std::chrono::high_resolution_clock::now();
  if (!prefetch_enabled_ || history_->GetIterationIdx() != 3) {
    return;
  }
  sa_log << "Prefetch: Start Prefetching" << std::endl;
  prefetching_ = true;
  cur_idx_in_node_ = 0;
  cur_node_idx_ = 0;
  prefetcher_ = std::thread(&Prefetch::Prefetching, this);
  sa_log << "Prefetch: Start Prefetching, thread created" << std::endl;
}

void Prefetch::StopPrefetching() {
  if (!prefetch_enabled_ || history_->GetIterationIdx() < num_loop_) {
    return;
  }
  prefetching_ = false;
  prefetcher_.join();
}

void Prefetch::Prefetching() {
  bool success;
  while (prefetching_) {
    sa_log << "Prefetch: prefetching " 
           << prefetch_sequence_[cur_node_idx_][cur_idx_in_node_] << std::endl;
    ODSwap::Get()->GetAddr(prefetch_sequence_[cur_node_idx_][cur_idx_in_node_],
        true, success);
    sa_log << "Prefetch: " << (success?"success":"failure") << std::endl;
    if (!success) {
      std::cout << "Prefetch failed " << std::endl;
      sem_wait(&prefetch_sem_);
    } else {
      cur_idx_in_node_++;
      if (cur_idx_in_node_ == prefetch_sequence_[cur_node_idx_].size()) {
        sa_log << "Prefetch: End of node with index " << cur_node_idx_ << std::endl;
        cur = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = cur-start;
        sa_log << "Prefetch: time now: " <<  diff.count() << "s \n";
        cur_idx_in_node_ = 0;
        cur_node_idx_++;
        if (cur_node_idx_ == prefetch_sequence_.size()) {
          sa_log << "Prefetch: End of iteration" << std::endl;
          end = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> diff = end-start;
          sa_log << "Prefetch stops at: " << diff.count() << " s\n";
          cur_node_idx_ = 0;
          cur_idx_in_node_ = 0;
          //break;
        }
      }
    } // if (!success)
  } // While true
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
