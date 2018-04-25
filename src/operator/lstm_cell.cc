/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cc
 * \brief activation op
 * \author Bing Xu
*/
#include "./lstm_cell-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new LSTMCellOp<cpu, DType>();
  })
  return op;
}

template<>
Operator* CreateBackwardOp<cpu>(int dtype) {
  return CreateOp<cpu>(dtype);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* LSTMCellProp::CreateOperatorEx(
    Context ctx,
    std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, (*in_type)[0]);
}

Operator* LSTMCellProp::CreateBackwardOperatorEx(
    const Context& ctx,
    const std::vector<TShape>& in_shape,
    const std::vector<int>& in_type,
    const std::vector<TShape>& out_shape,
    const std::vector<int>& out_type) const {
  DO_BIND_DISPATCH(CreateBackwardOp, in_type[0]);
}

//DMLC_REGISTER_PARAMETER(ActivationParam);

MXNET_REGISTER_OP_PROPERTY(LSTMCell, LSTMCellProp)
.describe("Fused element-wise op for LSTMCell")
.add_argument("prev_c", "Symbol", "Previous cell state.")
.add_argument("in_gate", "Symbol", "Input gate.")
.add_argument("trans_gate", "Symbol", "Transform gate.")
.add_argument("forget_gate", "Symbol", "Forget gate.")
.add_argument("out_gate", "Symbol", "Output gate.");
//.add_arguments(ActivationParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

