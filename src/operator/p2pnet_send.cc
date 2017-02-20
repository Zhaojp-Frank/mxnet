/*!
 * Copyright (c) 2017 by Contributors
 * \file p2pnet_send.cc
 * \brief
 * \author Chien-Chin Huang
*/
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <zmq.h>
#include "./p2pnet_send-inl.h"
#include "./p2pnet_common.h"
#include "./operator_common.h"

namespace mxnet {
namespace op {

class P2PNetSendProperty : public OperatorProperty {
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
    // Avoid unused variable warnings.
    (void)aux_shape;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data]";
    TShape outshape(1);
    outshape[0] = 1;
    out_shape->clear();
    out_shape->push_back(outshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new P2PNetSendProperty();
    ptr->param_ = param_;
    return ptr;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "control"};
  }

  std::string TypeString() const override {
    return "P2PNetSend";
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
      op = new P2PNetSendOp<DType>(param_);
    });
    return op;
  }

 private:
  P2PNetSendParam param_;
};  // class P2PNetSendProperty

DMLC_REGISTER_PARAMETER(P2PNetSendParam);

MXNET_REGISTER_OP_PROPERTY(P2PNetSend, P2PNetSendProperty)
.add_argument("data", "Symbol", "Input matrix to the P2PNetSendOp.")
.add_argument("control", "Symbol", "Control matrix to the P2PNetSendOp.")
.add_arguments(P2PNetSendParam::__FIELDS__())
.describe("Special op to send a matrix.");

}  // namespace op
}  // namespace mxnet
