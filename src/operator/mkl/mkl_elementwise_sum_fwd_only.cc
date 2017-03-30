#include "./mkl_elementwise_sum_fwd_only-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(
    const ElementWiseSumOnlyFwdParam& param,
    int dtype,
    const std::vector<TShape>& in_shapes,
    const std::vector<TShape>& out_shapes) {
#if MXNET_USE_MKL2017 == 1
  switch (dtype) {
  case mshadow::kFloat32:
    return new MKLElementWiseSumOnlyFwdOp<cpu, float>(param, in_shapes, out_shapes);
  case mshadow::kFloat64:
    return new MKLElementWiseSumOnlyFwdOp<cpu, double>(param, in_shapes, out_shapes);
  }
#endif
  return nullptr;
}
  
Operator* ElementWiseSumOnlyFwdProp::CreateOperatorEx(
    Context ctx,
    std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const {
  std::vector<TShape> out_shape(1, TShape()), aux_shape;
  std::vector<int> out_type(1, -1), aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0), *in_shape, out_shape);
}

DMLC_REGISTER_PARAMETER(ElementWiseSumOnlyFwdParam);

MXNET_REGISTER_OP_PROPERTY(ElementWiseSumOnlyFwd, ElementWiseSumOnlyFwdProp)
  .describe("Element wise summation only forward")
  .add_argument("data", "Symbol[]", "List of tensors to sum")
  .add_arguments(ElementWiseSumOnlyFwdParam::__FIELDS__())
  .set_key_var_num_args("num_args")
;
}  // namespace op
}  // namespace mxnet
