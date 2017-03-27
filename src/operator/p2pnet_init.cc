/*!
 * Copyright (c) 2017 by Contributors
 * \file p2pnet_init.cc
 * \brief
 * \author Chien-Chin Huang
*/
#include <zmq.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include "./operator_common.h"
#include "./p2pnet_init-inl.h"
#include "./p2pnet_common.h"

namespace mxnet {
namespace op {

class P2PNetInitOp : public Operator {
 public:
  explicit P2PNetInitOp(P2PNetInitParam param) : address_(param.address) {
    auto pos = address_.find(':');
    if (pos == std::string::npos) {
      LOG(FATAL) << "The address should be in the form ''ip:port''.";
    }
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data,
               const std::vector<TBlob> &aux_args) override {
    // Avoid unused variable warnings.
    (void)ctx;(void)in_data;(void)req;(void)out_data;(void)aux_args;
    P2PNet::Get().Init(address_);
    P2PNet::Get().Start();
  }

  ExecType exec_type() const override {
    return kSync;
  }

 private:
  std::string address_;
};

class P2PNetInitProperty : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string>>& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    // Avoid unused variable warnings.
    (void)aux_shape;
    //CHECK_EQ(in_shape->size(), 1) << "Input:[control]";
    CHECK_EQ(in_shape->size(), 1) << "Input:[control]";
    TShape outshape(1);
    outshape[0] = 1;
    out_shape->clear();
    out_shape->push_back(outshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new P2PNetInitProperty();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "P2PNetInit";
  }

  Operator* CreateOperator(Context ctx) const override {
    // Avoid unused variable warnings.
    (void)ctx;
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override {
    // Avoid unused variable warnings.
    (void)ctx;(void)in_shape;
    Operator *op = NULL;
    MSHADOW_TYPE_SWITCH(in_type->at(0), DType, {
      op = new P2PNetInitOp(param_);
    });
    return op;
  }

 private:
  P2PNetInitParam param_;
};

DMLC_REGISTER_PARAMETER(P2PNetInitParam);
MXNET_REGISTER_OP_PROPERTY(P2PNetInit, P2PNetInitProperty)
.add_argument("control", "Symbol", "Control matrix to the P2PNetInitOp.")
.add_arguments(P2PNetInitParam::__FIELDS__())
.describe("Special op to initialize network.");
}  // namespace op
}  // namespace mxnet
