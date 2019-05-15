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
#include<dmlc/base.h>
#include<dmlc/concurrency.h>
#include <atomic>
#include <thread>
#include "./threaded_engine.h"
#include "./thread_pool.h"
#include "../common/cuda_utils.h"

namespace mxnet {
namespace engine {

// implement naive engine
class SwapAdvisorEngine final : public ThreadedEngine {
 public:
  SwapAdvisorEngine() {
    this->Start();
  }
  // virtual destructor
  virtual ~SwapAdvisorEngine() {
    this->Stop();
  }

  void Stop() override {
#if MXNET_USE_CUDA
    for (size_t i = 0; i < streams_.size(); ++i) {
      MSHADOW_CATCH_ERROR(mshadow::DeleteStream<gpu>(streams_[i]));
      streams_[i] = nullptr;
    }
    for (size_t i = 0; i < swapin_streams_.size(); ++i) {
      MSHADOW_CATCH_ERROR(mshadow::DeleteStream<gpu>(swapin_streams_[i]));
      swapin_streams_[i] = nullptr;
    }
    for (size_t i = 0; i < swapout_streams_.size(); ++i) {
      MSHADOW_CATCH_ERROR(mshadow::DeleteStream<gpu>(swapout_streams_[i]));
      swapout_streams_[i] = nullptr;
    }
#endif // MXNET_USE_CUDA
    task_queue_->SignalForKill();
    swapin_task_queue_->SignalForKill();
    swapout_task_queue_->SignalForKill();
    task_queue_ = nullptr;
    swapin_task_queue_ = nullptr;
    swapout_task_queue_ = nullptr;
    thread_pool_ = nullptr;
    swapin_thread_pool_ = nullptr;
    swapout_thread_pool_ = nullptr;
  }

  void Start() override {
    task_queue_.reset(new dmlc::ConcurrentBlockingQueue<OprBlock*>());
    swapin_task_queue_.reset(new dmlc::ConcurrentBlockingQueue<OprBlock*>());
    swapout_task_queue_.reset(new dmlc::ConcurrentBlockingQueue<OprBlock*>());
    thread_pool_.reset(new ThreadPool(1,
                       [this](std::shared_ptr<dmlc::ManualEvent> ready_event) {
                        ThreadWorker(task_queue_, ready_event);
                       }, true));
    swapin_thread_pool_.reset(new ThreadPool(1,
                       [this](std::shared_ptr<dmlc::ManualEvent> ready_event) {
                        ThreadWorker(swapin_task_queue_, ready_event);
                       }, true));
    swapout_thread_pool_.reset(new ThreadPool(1,
                       [this](std::shared_ptr<dmlc::ManualEvent> ready_event) {
                        ThreadWorker(swapout_task_queue_, ready_event);
                       }, true));
  }

 protected:
  // priority variable stores node id of the node for this engine.
  void PushToExecute(OprBlock *opr_block, bool pusher_thread) override {
    const char* opr_name = opr_block->opr->opr_name;
    if (strlen(opr_name) < 6 || opr_name[0] != 'S' || opr_name[1] != 'w' ||
        opr_name[2] != 'a' || opr_name[3] != 'p') {
      opr_block->priority = 0;
    } else if (opr_name[4] == 'i' && opr_name[5] == 'n' &&
               opr_name[6] == '\0') {
      opr_block->priority = 1;
    } else if (opr_name[4] == 'o' && opr_name[5] == 'u' &&
               opr_name[6] == 't') {
      opr_block->priority = 2;
    } else if (opr_name[4] == '_' && opr_name[5] == 'e' &&
               opr_name[6] == 'n' && opr_name[7] == 't') {
      opr_block->priority = 1;
    } else {
      opr_block->priority = 0;
    }
    if (opr_block->opr->node_name != nullptr) {
      std::cout << "Opr = " << opr_block->opr->opr_name << ", "
                << "name = " << opr_block->opr->node_name << ", isGPU: "
                << (int)(opr_block->ctx.dev_mask() == gpu::kDevMask) << std::endl;
    } else {
      std::cout << "Opr = " << opr_block->opr->opr_name << ", isGPU: "
                << (int)(opr_block->ctx.dev_mask() == gpu::kDevMask) << std::endl;
    }
    if (opr_block->priority == 0) {
      task_queue_->Push(opr_block);
    } else if (opr_block->priority == 1) {
      swapin_task_queue_->Push(opr_block);
    } else if (opr_block->priority == 2) {
      swapout_task_queue_->Push(opr_block);
    } else {
      CHECK(false) << "No right engine priority";
    }
  }

 private:
  // Execute the operation
  void DoExecute(OprBlock* opr_block) {
    assert(opr_block->wait.load() == 0);
    if (opr_block->ctx.dev_mask() == gpu::kDevMask) {
#if MXNET_USE_CUDA
      size_t dev_id = static_cast<size_t>(opr_block->ctx.dev_id);
      //MSHADOW_CATCH_ERROR(mshadow::SetDevice<gpu>(opr_block->ctx.dev_id));
      CUDA_CALL(cudaSetDevice(opr_block->ctx.dev_id));
      mshadow::Stream<gpu>* cur_stream = nullptr;
      if (opr_block->priority == 0) {
        cur_stream = this->GetStream(streams_, dev_id);
      } else if (opr_block->priority == 1) {
        cur_stream = this->GetStream(swapin_streams_, dev_id);
      } else if (opr_block->priority == 2) {
        cur_stream = this->GetStream(swapout_streams_, dev_id);
      } else {
        CHECK(false) << "No right engine priority";
      }
      this->ExecuteOprBlock((RunContext){opr_block->ctx, cur_stream},
            opr_block);
#else //MXNET_USE_CUDA
      LOG(FATAL) << "Please compoile with CUDA enabled";
#endif //MXNET_USE_CUDA
    } else {
      this->ExecuteOprBlock((RunContext){opr_block->ctx, &cpu_stream_}, opr_block);
    }
  }

  void ThreadWorker(std::shared_ptr<dmlc::ConcurrentBlockingQueue<OprBlock*>> task_queue,
                    const std::shared_ptr<dmlc::ManualEvent>& ready_event) {
    OprBlock* opr_block;
    ready_event->signal();
    while(task_queue->Pop(&opr_block)) {
      DoExecute(opr_block);
    }
  }

  mshadow::Stream<gpu>* GetStream(std::vector<mshadow::Stream<gpu>*> &streams,
      size_t dev_id) {
    if (streams.size() <= dev_id) {
      streams.resize(dev_id + 1, nullptr);
    }
    if (streams[dev_id] == nullptr) {
      streams[dev_id] = mshadow::NewStream<gpu>(true,
          MXNET_USE_CUDNN != 0, dev_id);
    }
    return streams[dev_id];
  }

  // CPU stream
  mshadow::Stream<cpu> cpu_stream_;
  // GPU streams
  std::vector<mshadow::Stream<gpu>*> streams_;
  // Swapout Streams
  std::vector<mshadow::Stream<gpu>*> swapout_streams_;
  // Swapin Streams
  std::vector<mshadow::Stream<gpu>*> swapin_streams_;
  // Task queues
  std::shared_ptr<dmlc::ConcurrentBlockingQueue<OprBlock*>> task_queue_;
  std::shared_ptr<dmlc::ConcurrentBlockingQueue<OprBlock*>> swapout_task_queue_;
  std::shared_ptr<dmlc::ConcurrentBlockingQueue<OprBlock*>> swapin_task_queue_;
  // Thread pools
  std::unique_ptr<ThreadPool> thread_pool_;
  std::unique_ptr<ThreadPool> swapin_thread_pool_;
  std::unique_ptr<ThreadPool> swapout_thread_pool_;
};  // class SwapAdvisorEngine

Engine *CreateSwapAdvisorEngine() {
  return new SwapAdvisorEngine();
}
}  // namespace engine
}  // namespace mxnet
