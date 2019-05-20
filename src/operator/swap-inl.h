
#ifndef MXNET_OPERATOR_SWAP_INL_H_
#define MXNET_OPERATOR_SWAP_INL_H_

#include <mxnet/operator_util.h>
#include "./mxnet_op.h"

namespace mxnet {
namespace op {

struct SwapOpParam : public dmlc::Parameter<SwapOpParam> {
  uint32_t src_tensor_nid;
  uint32_t src_tensor_idx;
  uint32_t is_noop;
  DMLC_DECLARE_PARAMETER(SwapOpParam) {
    DMLC_DECLARE_FIELD(src_tensor_nid)
    .describe("The node id of the source tensor.");
    DMLC_DECLARE_FIELD(src_tensor_idx)
    .describe("The index of the source tensor.");
    DMLC_DECLARE_FIELD(is_noop)
    .describe("Whether this node is an nope operator.");
  }
};

}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_SWAP_INL_H_
