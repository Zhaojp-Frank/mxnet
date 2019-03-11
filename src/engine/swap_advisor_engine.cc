/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2015 by Contributors
 * \file swap_advisor_engine.cc
 * \brief Implementation of SwapAdvisorEngine
 *  It is build upon implementation of Naive Engine.
 */
#include <atomic>
#include <thread>
#include "./threaded_engine.h"

namespace mxnet {
namespace engine {

// implement naive engine
class SwapAdvisorEngine final : public ThreadedEngine {
 public:
  SwapAdvisorEngine() {
    ReadSchedule();
    next_opr_ = 0;
  }
  // virtual destructor
  virtual ~SwapAdvisorEngine() {
#if MXNET_USE_CUDA
    LOG(INFO) << "Engine shutdown";
    for (size_t i = 0; i < streams_.size(); ++i) {
      if (streams_[i] != nullptr) {
        // Catch exception for CUDA driver shutdown
        MSHADOW_CATCH_ERROR(mshadow::DeleteStream(streams_[i]));
        streams_[i] = nullptr;
      }
    }
#endif
  }

  void Stop() override {
  }

  void Start() override {
  }

 protected:
  // priority variable stores node id of the node for this engine.
  /* (TODO: Sotskin) Test this function. Current idea
   * Use a map to store mapping: node_id -> {op, exec_ctx, profiling}
   * If current op is the next to be run according to scheduling, run it,
   * Otherwise, store the information in map.
   *
   * After current op is ran, call a function to check if next op is in the
   * map, and run it if it is.
   */
  void PushToExecute(OprBlock *opr_block, bool pusher_thread) override {
    if((size_t)opr_block->priority == exec_order_[next_opr_]) {
      DoExecute(opr_block); 
      next_opr_++;
      if(opr_block_map_.find(exec_order_[next_opr_]) != 
            opr_block_map_.end()) {
        PushToExecute(opr_block_map_[exec_order_[next_opr_]], false);
      }
    } else {
      opr_block_map_[opr_block->priority] = opr_block;
      // Current is not the one we want
    }
  }

 private:
  // Read dataflow scheduling.
  void ReadSchedule() {
     
  }
  // Execute the operation
  void DoExecute(OprBlock* opr_block) {
    assert(opr_block->wait.load() == 0);
    if (opr_block->ctx.dev_mask() == gpu::kDevMask) {
#if MXNET_USE_CUDA
      size_t dev_id = static_cast<size_t>(exec_ctx.dev_id);
      MSHADOW_CATCH_ERROR(mshadow::SetDevice<gpu>(exec_ctx.dev_id));
      if (streams_.size() <= dev_id) {
        streams_.resize(dev_id + 1, nullptr);
      }
      if (streams_[dev_id] == nullptr) {
        streams_[dev_id] = mshadow:NewStream<gpu>(true,
            MXNET_USE_CUDNN != 0, dev_id);
      }
      this->ExecuteOprBlock(opr_block, 
          RunContext{opr_block->ctx, streams_[dev_id]});
#else //MXNET_USE_CUDA
      LOG(FATAL) << "Please compoile with CUDA enabled";
#endif //MXNET_USE_CUDA
    }
  } 
  // CPU stream
  mshadow::Stream<cpu> cpu_stream_;
  // GPU streams
  std::vector<mshadow::Stream<gpu>*> streams_;
  /*! \brief order of execution by nodeid given by scheduler. */
  std::vector<size_t> exec_order_;
  /*! \brief node id of next opr to be run */
  size_t next_opr_;
  /*! \brief map that stores operator info for nodes that haven't been run yet */
  std::map<size_t, OprBlock*> opr_block_map_;
};  // class SwapAdvisorEngine

Engine *CreateSwapAdvisorEngine() {
  return new SwapAdvisorEngine();
}
}  // namespace engine
}  // namespace mxnet
