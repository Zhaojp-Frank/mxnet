/*!
 * Copyright (c) 2017 by Contributors
 * \file dfge_profiling.h
 * \brief
 * \author Chien-Chin Huang
*/
#ifndef MXNET_EXECUTOR_DFGE_PROFILING_H_
#define MXNET_EXECUTOR_DFGE_PROFILING_H_
#include <chrono>
using namespace std::chrono;

struct DFGEProfile {
  int32_t id;
  int8_t  is_gpu;
  int8_t  is_async;
  int8_t  unused1;
  int8_t  unused2;
  int64_t time;
};

namespace mxnet {
namespace exec {
  class DFGEProfiler {
    public:
      static DFGEProfiler& Get() {
        static DFGEProfiler instance;
        return instance;
      }

      void Begin() {
        is_begin_ = true;
        size_ = 0;
      }

      void End() {
        is_begin_ = false;
      }

      void Write(int32_t id, bool is_gpu, bool is_async) {
        if (is_begin_) {
          int8_t is_gpu_int = (int8_t) is_gpu;
          int8_t is_async_int = (int8_t) is_async;
          auto now = high_resolution_clock::now();
          int64_t time = 
              duration_cast<microseconds>(now.time_since_epoch()).count();
          DFGEProfile profile{id, is_gpu_int, is_async_int, -1, -1, time};
          *(reinterpret_cast<int32_t*>(buffer_ + size_)) = profile.id;
          *(reinterpret_cast<int8_t*>(buffer_ + size_ + 4)) = profile.is_gpu;
          *(reinterpret_cast<int8_t*>(buffer_ + size_ + 5)) = profile.is_async;
          *(reinterpret_cast<int64_t*>(buffer_ + size_ + 8)) = profile.time;
          size_ += sizeof(DFGEProfile);
        }
      }

      char* Read(size_t* size) {
        *size = size_;
        return buffer_;
      }

    private:
      DFGEProfiler() : is_begin_(false), size_(0) {
        #define MAX_NODE (32768)
        buffer_ = new char[MAX_NODE * sizeof(DFGEProfile) * 2];
      }
      bool is_begin_;
      size_t size_;
      char *buffer_;
  };
} // namespace exec
} // namespace mxnet
#endif
