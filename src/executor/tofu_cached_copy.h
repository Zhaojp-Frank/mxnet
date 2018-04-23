#ifndef MXNET_EXECUTOR_TOFU_CACHED_COPY_H_
#define MXNET_EXECUTOR_TOFU_CACHED_COPY_H_

#include <nnvm/node.h>
#include <mxnet/ndarray.h>

namespace mxnet {
namespace exec {

void TofuCachedCopy(const nnvm::NodeAttrs& attrs,
                    const NDArray& from,
                    NDArray* to,
                    int priority = 0);

}  // namespace exec
}  // namespace mxnet

#endif  // MXNET_EXECUTOR_TOFU_CACHED_COPY_H_
