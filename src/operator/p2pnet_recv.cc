/*!
 * Copyright (c) 2017 by Contributors
 * \file p2pnet_recv.cc
 * \brief
 * \author Chien-Chin Huang
*/
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
    // Avoid unused variable warnings.
    (void)aux_shape;
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
    //return {"data", "control"};
    return {"control"};
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
    (void)ctx;(void)in_shape;(void)in_type;
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
//.add_argument("data", "Symbol", "Control matrix to the P2PNetRecvOp.")
.add_argument("control", "Symbol", "Control matrix to the P2PNetRecvOp.")
.add_arguments(P2PNetRecvParam::__FIELDS__())
.describe("Special op to receive a matrix.");

inline void P2PNetRecvAttrParser(NodeAttrs* attrs) {
  P2PNetRecvParam param;
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

inline bool P2PNetRecvInferShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_shapes,
                                 std::vector<TShape> *out_shapes) {
  CHECK_EQ(in_shapes->size(), 1);
  CHECK_EQ(out_shapes->size(), 1);
  const P2PNetRecvParam& param = nnvm::get<P2PNetRecvParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_shapes, 0, param.shape);
  return true;
}

inline bool P2PNetRecvInferType(const nnvm::NodeAttrs& attrs,
                                std::vector<int> *in_types,
                                std::vector<int> *out_types) {
  CHECK_EQ(in_types->size(), 1);
  CHECK_EQ(out_types->size(), 1);
  const P2PNetRecvParam& param = nnvm::get<P2PNetRecvParam>(attrs.parsed);
  TYPE_ASSIGN_CHECK(*out_types, 0, param.dtype);
  return true;
}

//NNVM_REGISTER_OP(P2PNetRecv)
  //.set_num_inputs(2)
  //.set_num_outputs(1)
  //.set_attr_parser(P2PNetRecvAttrParser)
  //.set_attr<FCompute>("FCompute<cpu>", P2PNetRecvCompute)
  //.set_attr<nnvm::FInferShape>("FInferShape", P2PNetRecvInferShape)
  //.set_attr<nnvm::FInferType>("FInferType", P2PNetRecvInferType)
  //.set_attr<nnvm::FListInputNames>("FListInputNames",
      //[](const NodeAttrs& attrs) {
        //(void) attrs;
        //return std::vector<std::string>{"data", "control"};
      //})
  //.add_argument("data", "NDArray", "Input matrix to the P2PNetRecvOp.")
  //.add_argument("control", "NDArray", "Control matrix to the P2PNetRecvOp.")
  //.add_arguments(P2PNetRecvParam::__FIELDS__())
  //.describe("Special op to receive a matrix.");

}  // namespace op
}  // namespace mxnet
