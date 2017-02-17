/*!
 * Copyright (c) 2017 by Contributors
 * \file net_send.cc
 * \brief
 * \author Chien-Chin Huang
*/
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <zmq.h>
#include "./net_send-inl.h"
#include "./net_common.h"
#include "./operator_common.h"

namespace mxnet {
namespace op {

class NetSendProperty : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 2) << "Input:[data]";
    TShape outshape(1);
    outshape[0] = 1;
    out_shape->clear();
    out_shape->push_back(outshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new NetSendProperty();
    ptr->param_ = param_;
    return ptr;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "control"};
  }

  std::string TypeString() const override {
    return "NetSend";
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override {
      Operator *op = NULL;
      MSHADOW_TYPE_SWITCH(in_type->at(0), DType, {
        op = new NetSendOp<DType>(param_);
      });
      return op;
  }

 private:
  NetSendParam param_;
};  // class NetSendProperty

DMLC_REGISTER_PARAMETER(NetSendParam);

MXNET_REGISTER_OP_PROPERTY(NetSend, NetSendProperty)
.add_argument("data", "Symbol", "Input matrix to the NetSendOp.")
.add_argument("control", "Symbol", "Control matrix to the NetSendOp.")
.add_arguments(NetSendParam::__FIELDS__())
.describe("Special op to send a matrix.");

}  // namespace op
}  // namespace mxnet
