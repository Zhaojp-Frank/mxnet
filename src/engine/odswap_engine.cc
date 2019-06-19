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
 * \brief Implementation of ODSwapEngine
 *  It is build upon implementation of Naive Engine.
 */
#include <dmlc/base.h>
#include <dmlc/concurrency.h>
#include <atomic>
#include <thread>
#include "./threaded_engine.h"
#include "./thread_pool.h"
#include "../common/cuda_utils.h"

namespace mxnet {
namespace engine {

// implement naive engine
class ODSwapEngine final : public ThreadedEngine {
 public:
  ODSwapEngine() {
    this->Start();
  }
  // virtual destructor
  virtual ~ODSwapEngine() {
    this->Stop();
  }

  void Stop() override {
#if MXNET_USE_CUDA
    for (size_t i = 0; i < streams_.size(); ++i) {
      MSHADOW_CATCH_ERROR(mshadow::DeleteStream<gpu>(streams_[i]));
      streams_[i] = nullptr;
    }
#endif // MXNET_USE_CUDA
    task_queue_->SignalForKill();
    task_queue_ = nullptr;
    thread_pool_ = nullptr;
  }

  void Start() override {
    task_queue_.reset(new dmlc::ConcurrentBlockingQueue<OprBlock*>());
    thread_pool_.reset(new ThreadPool(1,
                       [this](std::shared_ptr<dmlc::ManualEvent> ready_event) {
                        ThreadWorker(task_queue_, ready_event, 1);
                       }, true));
  }

 protected:
  // priority variable stores node id of the node for this engine.
  void PushToExecute(OprBlock *opr_block, bool pusher_thread) override {
    task_queue_->Push(opr_block);
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
      mshadow::Stream<gpu>* cur_stream = this->GetStream(streams_, dev_id);
      this->ExecuteOprBlock((RunContext){opr_block->ctx, cur_stream},
            opr_block);
#else //MXNET_USE_CUDA
      LOG(FATAL) << "Please compoile with CUDA enabled";
#endif //MXNET_USE_CUDA
    } else {
      this->ExecuteOprBlock((RunContext){opr_block->ctx, &cpu_stream_}, opr_block);
    }
  }

  void ThreadWorker(
          std::shared_ptr<dmlc::ConcurrentBlockingQueue<OprBlock*>> task_queue,
          const std::shared_ptr<dmlc::ManualEvent>& ready_event,
          int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    CHECK_EQ(rc, 0) << "Error calling pthread_setaffinity_np: " << rc;

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
  // Task queues
  std::shared_ptr<dmlc::ConcurrentBlockingQueue<OprBlock*>> task_queue_;
  // Thread pools
  std::unique_ptr<ThreadPool> thread_pool_;
};  // class ODSwapEngine

Engine *CreateODSwapEngine() {
  return new ODSwapEngine();
}
}  // namespace engine
}  // namespace mxnet
