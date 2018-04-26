#ifndef MXNET_EXECUTOR_TOFU_OP_H_
#define MXNET_EXECUTOR_TOFU_OP_H_

#include <nnvm/node.h>
#include <mxnet/ndarray.h>

namespace mxnet {
namespace op {
void TofuCopyFromTo(const nnvm::NodeAttrs& attrs,
                    const std::vector<NDArray>& from,
                    NDArray* to,
                    int priority = 0);
void TofuCopyFromToNoComm(const nnvm::NodeAttrs& attrs,
                          const std::vector<NDArray>& from,
                          NDArray* to,
                          int priority = 0);
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_EXECUTOR_TOFU_OP_H_
