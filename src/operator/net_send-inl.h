/*!
 * Copyright (c) 2017 by Contributors
 * \file net_send-inl.h
 * \brief
 * \author Chien-Chin Huang
*/
#ifndef MXNET_OPERATOR_NET_SEND_INL_H_
#define MXNET_OPERATOR_NET_SEND_INL_H_
#include <cstring>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <map>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <string>
#include <utility>
#include <vector>
#include "./net_common.h"
#include "./operator_common.h"
namespace mxnet {
namespace op {

struct NetSendParam: public dmlc::Parameter<NetSendParam> {
  std::string address; 
  unsigned tensor_id;
  DMLC_DECLARE_PARAMETER(NetSendParam) {
    DMLC_DECLARE_FIELD(address).set_default("127.0.0.1:11111")
    .describe("The address and port (ip:port) this operator will send to.");
    DMLC_DECLARE_FIELD(tensor_id).set_default(0)
    .describe("An unique tensor id that both sender and receiver acknowledge."
              "This is not the same as the tensor id internally used by MXNet.");
  }
};  // struct NetSendParam

template<typename DType>
class NetSendOp : public Operator {
 public:
  explicit NetSendOp(NetSendParam param) 
    : address_(param.address), tensor_id_(param.tensor_id) {}

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data,
               const std::vector<TBlob> &aux_args) override {
    std::cout << "NetSend::Forward in" << std::endl;
    Context ndctx = Context::CPU();
    std::vector<NDArray*> ndptrs;
    std::vector<engine::VarHandle> read_vars;
    for (const auto input : in_data) {
      NDArray* nd = new NDArray(input, ndctx.dev_id);
      read_vars.push_back(nd->var());
      ndptrs.push_back(nd);
    }
    P2PNet::Request* request = new P2PNet::Request{
      P2PNet::SendRequest, address_, tensor_id_, in_data[0].dptr_,
      in_data[0].shape_.Size() * sizeof(DType), ndptrs};
    Engine::Get()->PushAsync(
      [request](RunContext rctx, Engine::CallbackOnComplete on_complete) {
        request->on_complete = on_complete;
        P2PNet::Get().DoRequest(request);
      }, ndctx, read_vars, {}, FnProperty::kNormal, 0,
      PROFILER_MESSAGE("NetSend"));
    std::cout << "NetSend::Forward out" << std::endl;
  }

  ExecType exec_type() const override {
    return kNetSend;
  }

 private:
  std::string address_;
  unsigned tensor_id_;
};  // class NetSendOp

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NET_SEND_INL_H_
