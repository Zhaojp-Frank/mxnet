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
Operator *CreateOp<gpu>(int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new LSTMCellOp<gpu, DType>();
  })
  return op;
}

template<>
Operator* CreateBackwardOp<gpu>(int dtype) {
  return CreateOp<gpu>(dtype);
}

}  // namespace op
}  // namespace mxnet

