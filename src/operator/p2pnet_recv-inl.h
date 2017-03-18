/*!
 * Copyright (c) 2017 by Contributors
 * \file p2pnet_recv-inl.h
 * \brief
 * \author Chien-Chin Huang
*/
#ifndef MXNET_OPERATOR_P2PNET_RECV_INL_H_
#define MXNET_OPERATOR_P2PNET_RECV_INL_H_
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

struct P2PNetRecvParam: public dmlc::Parameter<P2PNetRecvParam> {
  std::string address;
  unsigned tensor_id;
  TShape shape;
  int dtype;
  DMLC_DECLARE_PARAMETER(P2PNetRecvParam) {
    DMLC_DECLARE_FIELD(address).set_default("127.0.0.1:11111")
    .describe("The address and port (ip:port) this worker should listen.");
    DMLC_DECLARE_FIELD(tensor_id).set_default(0)
    .describe("An unique tensor id that both sender and receiver acknowledge."
              "This is not the same as the tensor id internally used by MXNet.");
    DMLC_DECLARE_FIELD(shape).set_default(TShape())
    .describe("The shape of the tensor to be received.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .describe("Receive matrix data type.");
  }
};  // struct P2PNetRecvParam

template<typename DType>
class P2PNetRecvOp : public Operator {
 public:
  explicit P2PNetRecvOp(P2PNetRecvParam param)
    : address_(param.address), tensor_id_(param.tensor_id),
      tensor_shape_(param.shape), dtype_(param.dtype) {}

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data,
               const std::vector<TBlob> &aux_args) override {
    std::cout << "P2PNetRecv::Forward in" << std::endl;
    P2PNet::Request* request = new P2PNet::Request{
      P2PNet::RecvRequest, address_, tensor_id_, out_data[0].dptr_,
      out_data[0].shape_.Size() * sizeof(DType), ctx.async_on_complete};
    //P2PNet::Get().DoRequest(request);
    ctx.async_on_complete();
    std::cout << "P2PNetRecv::Forward out " << request << std::endl;
  }

  ExecType exec_type() const override {
    return kAsync;
  }

 private:
  std::string address_;
  unsigned tensor_id_;
  TShape tensor_shape_;
  int dtype_;
}; // class P2PNetRecvOp

void P2PNetRecvCompute(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  const P2PNetRecvParam& param = nnvm::get<P2PNetRecvParam>(attrs.parsed);
  std::cout << "P2PNetRecvCompute in" << std::endl;
  //Context ndctx = Context::CPU();
  //std::vector<NDArray*> ndptrs;
  //std::vector<engine::VarHandle> read_vars;
  //for (const auto input : inputs) {
    //NDArray* nd = new NDArray(input, ndctx.dev_id);
    //read_vars.push_back(nd->var());
    //ndptrs.push_back(nd);
  //}
  //std::vector<engine::VarHandle> write_vars;
  //for (const auto output : outputs) {
    //NDArray* nd = new NDArray(output, ndctx.dev_id);
    //write_vars.push_back(nd->var());
    //ndptrs.push_back(nd);
  //}
  //std::cout << "P2PNetRecv::Forward " << param.address << std::endl;
  //P2PNet::Request* request = new P2PNet::Request{
    //P2PNet::RecvRequest, param.address, param.tensor_id, outputs[0].dptr_,
    //outputs[0].shape_.Size() * mshadow::mshadow_sizeof(param.dtype), ndptrs};
  //Engine::Get()->PushAsync(
    //[request](RunContext rctx, Engine::CallbackOnComplete on_complete) {
      //request->on_complete = on_complete;
      //P2PNet::Get().DoRequest(request);
    //}, ndctx, read_vars, write_vars, FnProperty::kNormal, 0,
    //PROFILER_MESSAGE("P2PNetRecv"));

  std::cout << "P2PNetRecvCompute out" << std::endl;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NET_RECV_INL_H_
