/*!
 * Copyright (c) 2017 by Contributors
 * \file p2pnet_recv.cc
 * \brief
 * \author Chien-Chin Huang
*/
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <zmq.h>
#include "./p2pnet_recv-inl.h"
#include "./p2pnet_common.h"
#include "./operator_common.h"

namespace mxnet {
namespace op {

class P2PNetRecvProperty : public OperatorProperty {
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
    out_shape->clear();
    out_shape->push_back(param_.shape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new P2PNetRecvProperty();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "P2PNetRecv";
  }

  std::vector<std::string> ListArguments() const override {
    return {"control"};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override {
      Operator *op = NULL;
      MSHADOW_TYPE_SWITCH(param_.dtype, DType, {
        op = new P2PNetRecvOp<DType>(param_);
      });
      return op;
  }

 private:
  P2PNetRecvParam param_;
};  // class P2PNetRecvProperty

DMLC_REGISTER_PARAMETER(P2PNetRecvParam);
MXNET_REGISTER_OP_PROPERTY(P2PNetRecv, P2PNetRecvProperty)
.add_argument("control", "Symbol", "Control matrix to the P2PNetRecvOp.")
.add_arguments(P2PNetRecvParam::__FIELDS__())
.describe("Special op to receive a matrix.");
}  // namespace op
}  // namespace mxnet
