/*!
 * Copyright (c) 2017 by Contributors
 * \file p2pnet_common.h
 * \brief
 * \author Chien-Chin Huang
*/
#ifndef MXNET_OPERATOR_NET_COMMON_H_
#define MXNET_OPERATOR_NET_COMMON_H_
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <dmlc/logging.h>
#include <iomanip>
#include <map>
#include <mutex>
#ifdef P2PNET_MPI
#include <mpi.h>
#endif
#include <mxnet/engine.h>
#include <mxnet/operator.h>
#include <mxnet/ndarray.h>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <zmq.h>
#include "./ctpl_stl.h"
#include "./operator_common.h"

using namespace std::chrono;

namespace mxnet {
namespace op {
class P2PNetDebugger {
 public:
  static P2PNetDebugger& Get() {
    static P2PNetDebugger instance;
    return instance;
  }

  int Level() {
    return level_;
  }

  void PrintTime(const char *fmt, ...) {
    if (level_ & kDebugPrintTime) {
      static char str_buf[2048];
      auto now = high_resolution_clock::now();
      auto now_ms = duration_cast<milliseconds>(now.time_since_epoch()).count();
      va_list args;
      va_start(args, fmt);
      vsprintf(str_buf, fmt, args);
      va_end(args);
      std::cout << str_buf << " at " << now_ms << " millisecond" << std::endl;
    }
  }

  P2PNetDebugger(P2PNetDebugger const&) = delete;
  void operator=(P2PNetDebugger const&) = delete;
  ~P2PNetDebugger() { };

  constexpr static int kDebugPrintTime = 1;
  constexpr static int kDebugNoCommunication = 2;
  constexpr static int kDebugNoReceiveCopy = 4;
  constexpr static int kDebugPrintPending = 8;

 private:
  P2PNetDebugger() {
    level_ = dmlc::GetEnv("MXNET_P2PNET_DEBUG", 0);
  };
  int level_;
};

class SpinLock {
 public:
  bool TryLock() {
    return !lock_.test_and_set(std::memory_order_acquire);
  }
  void Lock() {
    while (lock_.test_and_set(std::memory_order_acquire))
      ;  // spin
  }
  void UnLock() {
    lock_.clear(std::memory_order_release);
  }
 private:
  std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
};

class P2PNet {
 public:
  static P2PNet& Get() {
    static P2PNet instance;
    return instance;
  }
  P2PNet(P2PNet const&) = delete;
  void operator=(P2PNet const&) = delete;
  ~P2PNet();

  bool Init(const std::string& address_);

  void Start();

  enum RequestType {
    NewIndexRequest,
    AddRequest,
    SendRequest,
    RecvRequest,
  };

  struct Request {
    RequestType type;
    std::string address;
    uint64_t tensor_id;
    void* buffer;
    size_t buffer_size;
    engine::CallbackOnComplete on_complete;
    bool is_fulfilled;
#ifdef P2PNET_MPI
    MPI_Request *mpi_request;
#endif
  };
  void DoRequest(struct Request* request);
  void FreeRequest(struct Request* request);

  constexpr static int kRequestQueueSize = 1024 * 1024 * 2;
  constexpr static int kIdentitySize = 8;

 private:
  P2PNet();
  void Main();
  void SetMainAffinity();
  void DoInternalRequest(size_t request_index);
  void DoExternalRequest();
  void DoSend(struct Request* request);
  void DoRecv(void* socket);
  void DoRequestRecv(struct Request* request);

  unsigned impl_internal_polling_;
  unsigned impl_commnication_method_;
  unsigned impl_mpi_polling_time_;
  unsigned impl_main_affinity_;
  unsigned impl_use_mpi_barrier_;

#ifdef P2PNET_MPI
  void MPI_Main();
  void MPI_DoInternalRequest(struct Request* request);
  void MPI_DoSend(struct Request* request);
  void MPI_DoRecv(struct Request* request);
  void MPI_RequestOnComplete(struct Request* request);

  int mpi_rank_;
  std::vector<std::string> mpi_rank_to_host_;
  std::map<std::string, int> mpi_host_to_rank_;
  std::vector<struct Request*> mpi_request_queue_;
  int mpi_request_count_;
  size_t mpi_sent_bytes_;
  size_t mpi_recv_bytes_;
#endif

  void* zmq_context_;
  // Every worker contains a server socket to allow other workers to connect to .
  void* server_;
  // Every worker contains a internal server socket to allow send/recv requests
  // from MXNet send/recv operators.
  void* internal_server_;
  bool is_main_start_;
  bool is_bind_;
  zmq_pollitem_t* poll_items_;
  size_t poll_items_count_;
  std::thread* main_thread_;
  ctpl::thread_pool *recv_thread_pool_;

  std::mutex internal_mtx; // mutex lock for request_queue_
  SpinLock spin_lock_;  // spin lock for request queue
  std::vector<struct Request*> internal_request_queue_;
  std::atomic<size_t> internal_request_queue_size_;
  std::vector<void*> per_thread_isocket_queue_;
  std::atomic<size_t> per_thread_isocket_queue_size_;

  std::map<unsigned, std::string> tensor_to_receiver_map_;
  std::map<unsigned, size_t> tensor_to_send_request_map_;
  std::map<unsigned, size_t> tensor_to_recv_request_map_;
  std::map<std::string, void*> recv_request_sockets_;
  std::map<size_t, unsigned> recv_request_tensor_id_;
  std::map<void*, size_t> recv_poll_indices_;
};

}  // namespace op
}  // namespace mxnet
#endif // MXNET_OPERATOR_NET_COMMON_H_
