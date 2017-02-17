/*!
 * Copyright (c) 2017 by Contributors
 * \file net_init.cc
 * \brief
 * \author Chien-Chin Huang
*/
#include <zmq.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include "./operator_common.h"
#include "./net_init-inl.h"
#include "./net_common.h"

namespace mxnet {
namespace op {

class NetInitOp : public Operator {
 public:
  explicit NetInitOp(NetInitParam param) : address_(param.address) {
    auto pos = address_.find(':');
    if (pos == std::string::npos) {
      LOG(FATAL) << "The address should be in the form ''ip:port''.";
    }
    ip_ = address_.substr(0, pos);
    port_ = atoi(address_.substr(pos + 1, address_.length()).c_str());
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data,
               const std::vector<TBlob> &aux_args) override {
    // NetInit is specially handled by graph executor.
    //LOG(FATAL) << "Not Reached";
    std::cout << "NetInit::Forward in" << std::endl;
    P2PNet::Get().Bind(ip_, port_);
    P2PNet::Get().Start();
    std::cout << "NetInit::Forward out" << std::endl;
  }

  ExecType exec_type() const override {
    return kNetInit;
  }

 private:
  std::string address_;
  std::string ip_;
  int port_;
};

class NetInitProperty : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    TShape outshape(1);
    outshape[0] = 1;
    out_shape->clear();
    out_shape->push_back(outshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new NetInitProperty();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "NetInit";
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override {
      Operator *op = NULL;
      MSHADOW_TYPE_SWITCH(in_type->at(0), DType, {
        op = new NetInitOp(param_);
      });
      return op;
  }

 private:
  NetInitParam param_;
};

DMLC_REGISTER_PARAMETER(NetInitParam);
MXNET_REGISTER_OP_PROPERTY(NetInit, NetInitProperty)
.add_argument("control", "Symbol", "Control matrix to the NetInitOp.")
.add_arguments(NetInitParam::__FIELDS__())
.describe("Special op to initialize network.");
}  // namespace op
}  // namespace mxnet
