/*!
 * Copyright (c) 2017 by Contributors
 * \file p2pnet_send.cc
 * \brief
 * \author Chien-Chin Huang
*/
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
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

 //private:
  P2PNetSendParam param_;
};  // class P2PNetSendProperty

DMLC_REGISTER_PARAMETER(P2PNetSendParam);

MXNET_REGISTER_OP_PROPERTY(P2PNetSend, P2PNetSendProperty)
.add_argument("data", "Symbol", "Input matrix to the P2PNetSendOp.")
.add_argument("control", "Symbol", "Control matrix to the P2PNetSendOp.")
.add_arguments(P2PNetSendParam::__FIELDS__())
.describe("Special op to send a matrix.");

inline bool P2PNetSendSinkInferShape(const nnvm::NodeAttrs& attrs,
                                       std::vector<TShape> *in_shapes,
                                       std::vector<TShape> *out_shapes) {
  // Avoid unused variable warnings.
  (void)attrs;
  CHECK_EQ(in_shapes->size(), 0);
  CHECK_EQ(out_shapes->size(), 1);
  TShape outshape(1);
  outshape[0] = 1;
  SHAPE_ASSIGN_CHECK(*out_shapes, 0, outshape);
  return true;
}

inline bool P2PNetSendSinkInferType(const nnvm::NodeAttrs& attrs,
                                      std::vector<int> *in_types,
                                      std::vector<int> *out_types) {
  (void)attrs;
  CHECK_EQ(in_types->size(), 0);
  CHECK_EQ(out_types->size(), 1);
  TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
  return true;
}

void P2PNetSendSinkCompute(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  (void)attrs;
  (void)ctx;
  (void)inputs;
  (void)req;
  (void)outputs;
}

NNVM_REGISTER_OP(P2PNetSendSink)
  .set_num_inputs(0)
  .set_num_outputs(1)
  .set_attr<FCompute>("FCompute<cpu>", P2PNetSendSinkCompute)
  .set_attr<nnvm::FInferShape>("FInferShape", P2PNetSendSinkInferShape)
  .set_attr<nnvm::FInferType>("FInferType", P2PNetSendSinkInferType)
  .add_arguments(P2PNetSendParam::__FIELDS__())
  .describe("Special op to be P2PNetSend sink.");

inline void P2PNetSendAttrParser(NodeAttrs* attrs) {
  P2PNetSendParam param;
  std::vector<std::pair<std::string, std::string> > kwargs(
    attrs->dict.begin(), attrs->dict.end());
  try {
    param.Init(kwargs);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  attrs->parsed = std::move(param);
}

inline bool P2PNetSendInferShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_shapes,
                                 std::vector<TShape> *out_shapes) {
  // Avoid unused variable warnings.
  (void)attrs;
  CHECK_EQ(in_shapes->size(), 2);
  CHECK_EQ(out_shapes->size(), 1);
  TShape outshape(1);
  outshape[0] = 1;
  SHAPE_ASSIGN_CHECK(*out_shapes, 0, outshape);
  return true;
}

inline bool P2PNetSendInferType(const nnvm::NodeAttrs& attrs,
                                std::vector<int> *in_types,
                                std::vector<int> *out_types) {
  (void)attrs;
  CHECK_EQ(in_types->size(), 2);
  CHECK_EQ(out_types->size(), 1);
  TYPE_ASSIGN_CHECK(*out_types, 0, (*in_types)[0]);
  return true;
}

//NNVM_REGISTER_OP(P2PNetSend)
  //.set_num_inputs(2)
  //.set_num_outputs(1)
  //.set_attr_parser(P2PNetSendAttrParser)
  //.set_attr<FCompute>("FCompute<cpu>", P2PNetSendCompute)
  //.set_attr<nnvm::FInferShape>("FInferShape", P2PNetSendInferShape)
  //.set_attr<nnvm::FInferType>("FInferType", P2PNetSendInferType)
  //.set_attr<nnvm::FListInputNames>("FListInputNames",
      //[](const NodeAttrs& attrs) {
        //(void) attrs;
        //return std::vector<std::string>{"data", "control"};
      //})
  //.add_argument("data", "NDArray", "Input matrix to the P2PNetSendOp.")
  //.add_argument("control", "NDArray", "Control matrix to the P2PNetSendOp.")
  //.add_arguments(P2PNetSendParam::__FIELDS__())
  //.describe("Special op to send a matrix.");

}  // namespace op
}  // namespace mxnet
