/*!
 * Copyright (c) 2015 by Contributors
 * \file slice_channel.cc
 * \brief
 * \author Bing Xu
*/

#include "./slice_channel-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(SliceChannelParam param) {
  return new SliceChannelOp<cpu>(param);
}
template<>
Operator* CreateBackwardOp<cpu>(const SliceChannelParam& param) {
  return CreateOp<cpu>(param);
}

Operator* SliceChannelProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

Operator* SliceChannelProp::CreateBackwardOperatorEx(
    const Context& ctx,
    const std::vector<TShape>& in_shape,
    const std::vector<int>& in_type,
    const std::vector<TShape>& out_shape,
    const std::vector<int>& out_type) const {
  DO_BIND_DISPATCH(CreateBackwardOp, param_);
}


DMLC_REGISTER_PARAMETER(SliceChannelParam);

MXNET_REGISTER_OP_PROPERTY(SliceChannel, SliceChannelProp)
.describe("Slice input equally along specified axis")
.set_return_type("Symbol[]")
.add_arguments(SliceChannelParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

