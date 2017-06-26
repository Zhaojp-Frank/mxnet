/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cc
 * \brief
 * \author Bing Xu
*/

#include "./convolution-inl.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_convolution-inl.h"
#endif  // MXNET_USE_MKL2017
#if MXNET_USE_NNPACK == 1
#include "./nnpack/nnpack_convolution-inl.h"
#endif  // MXNET_USE_NNPACK
#include "./operator_common.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(ConvolutionParam);

template<>
Operator* CreateOp<cpu>(ConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
#if MXNET_USE_MKL2017 == 1
  if ((param.dilate[0] == 1 && param.dilate[1] == 1)
      && param.kernel.ndim() == 2) {
    switch (dtype) {
    case mshadow::kFloat32:
      return new MKLConvolutionOp<cpu, float>(param, *in_shape, *out_shape);
    case mshadow::kFloat64:
      return new MKLConvolutionOp<cpu, double>(param, *in_shape, *out_shape);
    default:
      break;
    }
  }
  LOG(INFO) << MKLConvolutionOp<cpu, float>::getName() << " Skip MKL optimization";
#endif
#if MXNET_USE_NNPACK == 1
  const size_t batch_size = (*in_shape)[0][0];
  if ((param.dilate[0] == 1 && param.dilate[1] == 1)
      && param.kernel.ndim() == 2 && (!param.no_bias)
      && param.num_group == 1 && (batch_size == 1 ||
      ((batch_size > 1) && (param.stride[0] == 1) &&
      (param.stride[1] == 1)))) {
    switch (dtype) {
    case mshadow::kFloat32:
      return new NNPACKConvolutionOp<cpu, float>(param);
    default:
      break;
    }
  }
#endif
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ConvolutionOp<cpu, DType>(param);
  })
  return op;
}

template<>
Operator* CreateBackwardOp<cpu>(
    const ConvolutionParam& param,
    int dtype,
    const std::vector<TShape>& out_grad_shape,
    const std::vector<TShape>& in_data_shape,
    const std::vector<TShape>& out_data_shape,
    const std::vector<TShape>& in_grad_shape,
    const Context& ctx) {
#if MXNET_USE_MKL2017 == 1
  if ((param.dilate[0] == 1 && param.dilate[1] == 1)
      && param.kernel.ndim() == 2) {
    switch (dtype) {
    case mshadow::kFloat32:
      return new MKLConvolutionOp<cpu, float>(param, in_grad_shape, out_grad_shape);
    case mshadow::kFloat64:
      return new MKLConvolutionOp<cpu, double>(param, in_grad_shape, out_grad_shape);
    default:
      break;
    }
  }
  LOG(INFO) << MKLConvolutionOp<cpu, float>::getName() << " Skip MKL optimization";
#endif
#if MXNET_USE_NNPACK == 1
  LOG(INFO) << "Skip NNPACK for backward operator.";
#endif
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ConvolutionOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ConvolutionProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

Operator* ConvolutionProp::CreateBackwardOperatorEx(
    const Context& ctx,
    const std::vector<TShape>& in_shape,
    const std::vector<int>& in_type,
    const std::vector<TShape>& out_shape,
    const std::vector<int>& out_type) const {
  std::vector<TShape> out_grad_shape, in_data_shape, out_data_shape;
  ParseBackwardInputs(*this, in_shape, &out_grad_shape, &in_data_shape, &out_data_shape);
  DO_BIND_DISPATCH(CreateBackwardOp, param_, in_type[0],
                   out_grad_shape, in_data_shape, out_data_shape, out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionProp)
.add_argument("data", "Symbol", "Input data to the ConvolutionOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(ConvolutionParam::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet
