/*!
 * Copyright (c) 2017 by Contributors
 * \file p2pnet_send-inl.h
 * \brief
 * \author Chien-Chin Huang
*/
#ifndef MXNET_OPERATOR_P2PNET_SEND_INL_H_
#define MXNET_OPERATOR_P2PNET_SEND_INL_H_
#include <cstring>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <map>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <string>
#include <utility>
#include <vector>
#include <zmq.h>
#include "./p2pnet_common.h"
#include "./operator_common.h"
namespace mxnet {
namespace op {

struct P2PNetSendParam: public dmlc::Parameter<P2PNetSendParam> {
  std::string address;
  uint64_t tensor_id;
  DMLC_DECLARE_PARAMETER(P2PNetSendParam) {
    DMLC_DECLARE_FIELD(address).set_default("127.0.0.1:11111")
    .describe("The address and port (ip:port) this operator will send to.");
    DMLC_DECLARE_FIELD(tensor_id).set_default(0)
    .describe("An unique tensor id that both sender and receiver acknowledge."
              "This is not the same as the tensor id internally used by MXNet.");
  }
};  // struct P2PNetSendParam

template<typename DType>
class P2PNetSendOp : public Operator {
 public:
  explicit P2PNetSendOp(P2PNetSendParam param)
    : address_(param.address), tensor_id_(param.tensor_id) {}

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data,
               const std::vector<TBlob> &aux_args) override {
    P2PNet::Request* request = new P2PNet::Request{
      P2PNet::SendRequest, address_, tensor_id_, in_data[0].dptr_,
      in_data[0].shape_.Size() * sizeof(DType), ctx.async_on_complete};
    if (P2PNetDebugger::Get().Level() & P2PNetDebugger::kDebugNoCommunication) {
      ctx.async_on_complete();
    } else {
      P2PNet::Get().DoRequest(request);
    }
  }

  ExecType exec_type() const override {
    return kAsync;
  }

 private:
  std::string address_;
  unsigned tensor_id_;
};  // class P2PNetSendOp

//void P2PNetSendCompute(const nnvm::NodeAttrs& attrs,
                       //const OpContext& ctx,
                       //const std::vector<TBlob>& inputs,
                       //const std::vector<OpReqType>& req,
                       //const std::vector<TBlob>& outputs) {
  //const P2PNetSendParam& param = nnvm::get<P2PNetSendParam>(attrs.parsed);
  //std::cout << "P2PNetSendCompute in" << std::endl;
  //Context ndctx = Context::CPU();
  //std::vector<NDArray*> ndptrs;
  //std::vector<engine::VarHandle> read_vars;
  //for (const auto input : inputs) {
    //NDArray* nd = new NDArray(input, ndctx.dev_id);
    //read_vars.push_back(nd->var());
    //ndptrs.push_back(nd);
  //}
  //std::cout << "P2PNetSendCompute " << param.address << std::endl;
  //P2PNet::Request* request = new P2PNet::Request{
    //P2PNet::SendRequest, param.address, param.tensor_id, inputs[0].dptr_,
    //inputs[0].shape_.Size() * mshadow::mshadow_sizeof(inputs[0].type_flag_),
    //ndptrs};
  //Engine::Get()->PushAsync(
    //[request](RunContext rctx, Engine::CallbackOnComplete on_complete) {
      //request->on_complete = on_complete;
      //P2PNet::Get().DoRequest(request);
    //}, ndctx, read_vars, {}, FnProperty::kNormal, 0,
    //PROFILER_MESSAGE("P2PNetSendCompute"));
  //std::cout << "P2PNetSendCompute out" << std::endl;
//}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_P2PNET_SEND_INL_H_
