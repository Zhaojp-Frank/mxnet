/*!
 * Copyright (c) 2017 by Contributors
 * \file p2pnet_common.cc
 * \brief
 * \author Chien-Chin Huang
*/
#include <sstream>
#include <string>
#include <thread>
#include <iterator>
#include "./p2pnet_common.h"

namespace mxnet {
namespace op {

P2PNet::P2PNet() {
  zmq_context_ =  zmq_ctx_new();
  server_ = zmq_socket(zmq_context_, ZMQ_ROUTER);
  internal_server_ = zmq_socket(zmq_context_, ZMQ_ROUTER);
  is_main_start_ = false;
  is_bind_ = false;
  main_thread_ = nullptr;
}

P2PNet::~P2PNet() {
  zmq_close(internal_server_);
  zmq_close(server_);
}

bool P2PNet::Init(const std::string& address) {
  request_queue_.clear();
  remote_request_queue_.clear();
  send_request_queue_.clear();
  recv_request_queue_.clear();
  recv_request_poll_indices.clear();
  request_index_mapping_.clear();
  // {recv_request_sockets_.clear();}
  // Note that we should not reinit recv_request_sockets_ because we don't want
  // to do connect every time.
  if (!is_bind_) {
    srand(time(nullptr));
    std::ostringstream address_with_proto;  
    address_with_proto << "tcp://" << address;
    int ret = zmq_bind(server_, address_with_proto.str().c_str());
    // TODO: Use LOG instead of std::cout and have better log output information
    // for net_common.cc.
    std::cout << "zmq_bind " << address_with_proto.str() << " ret = " << ret 
              << std::endl;
    if (ret == 0) {
      zmq_bind(internal_server_, "inproc://mxnet_local_request");
      is_bind_ = true;
      return true;
    } else {
      return false;
    }
  }

  return true;
};

static int SendWithIdentity(void* socket, const std::string& identity, 
                            const void* buffer, int len) {
  // Use zero-copy to send message
  zmq_msg_t msg1;
  zmq_msg_init_data(&msg1, (void *)identity.c_str(), identity.size(), NULL, NULL);
  zmq_msg_send(&msg1, socket, ZMQ_SNDMORE);
  zmq_msg_t msg2;
  zmq_msg_init_data(&msg2, (void *)buffer, 0, NULL, NULL);
  zmq_msg_send(&msg2, socket, ZMQ_SNDMORE);
  zmq_msg_t msg3;
  zmq_msg_init_data(&msg3, (void *)buffer, len, NULL, NULL);
  return zmq_msg_send(&msg3, socket, 0);
}

static int RecvWithIdentity(void* socket, std::string* identity,
                            void* buffer, int len) {
  // TODO: Use constant instead of the magic number 8.
  char identity_buffer[8];
  zmq_recv(socket, identity_buffer, 8, 0);
  *identity = std::string(identity_buffer, 8);
  zmq_recv(socket, buffer, 0, 0);
  return zmq_recv(socket, buffer, len, 0);
}

// Create a random identity with exact 8 characters.
static std::string CreateIdentity() {
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<std::mt19937::result_type> dist6(0, 9999999);
  return std::to_string(dist6(rng) + 90000000);
}

void P2PNet::FreeRequest(struct Request* request) {
  for (auto nd : request->ndptrs) {
    delete nd;
  }
  delete request;
}

void P2PNet::DoSend(std::string& receiver_identity, 
                    struct Request* request) {
    // TODO: Change to zero-copy send.
    SendWithIdentity(server_, receiver_identity, request->buffer,
                     request->buffer_size);
    std::cout << "Send on_complete " << receiver_identity << std::endl;
    request->on_complete();
    FreeRequest(request);
}

void P2PNet::DoInternalRequest(size_t index) {
  struct Request* request = request_queue_[index];
  if (request->type == SendRequest) {
    std::cout << "DoInternalRequest SendRequest " << request->tensor_id << std::endl;
    auto it = remote_request_queue_.find(request->tensor_id);
    if (it == remote_request_queue_.end()) {
      send_request_queue_[request->tensor_id] = index;
    } else {
      DoSend(it->second, request);
    }
  } else if (request->type == RecvRequest) {
    recv_request_queue_[request->tensor_id] = index;
    void* request_socket;
    auto it = recv_request_sockets_.find(request->tensor_id);
    if (it == recv_request_sockets_.end()) {
      request_socket = zmq_socket(zmq_context_, ZMQ_DEALER);
      std::ostringstream address;  
      address << "tcp://" << request->address;
      std::string identity = CreateIdentity();
      zmq_setsockopt(request_socket, ZMQ_IDENTITY, identity.c_str(), 
                     identity.size());
      zmq_connect(request_socket, address.str().c_str());
      recv_request_sockets_[request->tensor_id] = request_socket;
      // TODO: We can use a more efficient way to avoid copying everytime. 
      // But this may not be very critical.
      poll_items_count_++;
      zmq_pollitem_t* new_poll_items = new zmq_pollitem_t[poll_items_count_];
      std::copy(poll_items_, poll_items_ + poll_items_count_ - 1,
                new_poll_items);
      delete poll_items_;
      poll_items_ = new_poll_items;
      poll_items_[poll_items_count_ - 1] = {request_socket, 0, ZMQ_POLLIN, 0};
      recv_request_poll_indices[poll_items_count_ - 1] = request->tensor_id;
    } else {
      request_socket = it->second;
    }
    // TODO: Currently, we only have one and the only one request to the remote
    // worker. Therefore, we assume that the request content is the tensor_id. 
    // This may not rigorous enough when the communication becomes more
    // complicated.
    std::cout << "Before remote send." << std::endl;
    zmq_send(request_socket, "", 0, ZMQ_SNDMORE); // ZMQ delimiter message.
    zmq_send(request_socket, &request->tensor_id, sizeof(request->tensor_id), 
             0);
    std::cout << "After remote send." << std::endl;
  }
}

void P2PNet::DoExternalRequest() {
  std::string identity;
  unsigned tensor_id = 0;
  int ret = RecvWithIdentity(server_, &identity, &tensor_id, sizeof(tensor_id));
  std::cout << "DoExternalRequest " << identity << " " << tensor_id << " " 
            << ret << std::endl;
  auto it = send_request_queue_.find(tensor_id);
  if (it == send_request_queue_.end()) {
    remote_request_queue_[tensor_id] = identity;
  } else {
    struct Request* request = request_queue_[it->second];
    std::cout << "Before DoSend" << std::endl;
    DoSend(identity, request);
  }
}

void P2PNet::Main() {
  poll_items_count_ = 2;
  poll_items_ = new zmq_pollitem_t[poll_items_count_];
  poll_items_[0] = {internal_server_, 0, ZMQ_POLLIN, 0};
  poll_items_[1] = {server_, 0, ZMQ_POLLIN, 0};
  while (true) {
    zmq_poll(poll_items_, poll_items_count_, -1);
    if (poll_items_[0].revents & ZMQ_POLLIN) { // internal request
      std::string identity;
      RequestType type;
      size_t index = 0;
      RecvWithIdentity(internal_server_, &identity, &type, sizeof(type));
      if (type == NewIndexRequest) {
        index = request_queue_.size();
        request_queue_.resize(index + 1);
        request_index_mapping_[identity] = index;
        SendWithIdentity(internal_server_, identity, &index, sizeof(index));
      } else if (type == AddRequest) {
        index = request_index_mapping_[identity];
        DoInternalRequest(index);
      }
    }
    if (poll_items_[1].revents & ZMQ_POLLIN) {
      DoExternalRequest();
    }
    for (unsigned i = 2; i < poll_items_count_; i++) {
      if (poll_items_[i].revents & ZMQ_POLLIN) {
        unsigned tensor_id = recv_request_poll_indices[i];
        struct Request* request = 
            request_queue_[recv_request_queue_[tensor_id]];
        zmq_recv(poll_items_[i].socket, request->buffer, 0, 0);
        zmq_recv(poll_items_[i].socket, request->buffer, request->buffer_size,
                 0);
        std::cout << "Recv on_complete" << std::endl;
        request->on_complete();
      }
    }
  }
}

void P2PNet::Start() {
  if (!is_main_start_) {
    main_thread_ = new std::thread(&P2PNet::Main, this);
    is_main_start_ = true;
  }
}

void P2PNet::DoRequest(struct Request* request) {
  // I use a very tricky way to avoid lock.
  // 1. Send a message to ask the server to increase the size of request queue.
  // 2. The server returns the position that this request can be put to.
  // 3. Put the request to the request queue.
  // 4. Send a message to ask the server to process the new request.
  void* request_socket = zmq_socket(zmq_context_, ZMQ_REQ);
  std::string identity = CreateIdentity();
  zmq_setsockopt(request_socket, ZMQ_IDENTITY, identity.c_str(), 8);
  zmq_connect(request_socket, "inproc://mxnet_local_request");
  RequestType type = NewIndexRequest;
  zmq_send(request_socket, &type, sizeof(type), 0);
  size_t index;
  zmq_recv(request_socket, &index, sizeof(size_t), 0);
  type = AddRequest;
  request_queue_[index] = request;
  zmq_send(request_socket, &type, sizeof(type), 0);
  zmq_close(request_socket);
}

}  // namespace op
}  // namespace mxnet
