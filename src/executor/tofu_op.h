#ifndef MXNET_EXECUTOR_TOFU_OP_H_
#define MXNET_EXECUTOR_TOFU_OP_H_

#include <nnvm/node.h>
#include <mxnet/ndarray.h>
#include <mxnet/engine.h>
#include "./exec_pass.h"

namespace mxnet {
namespace op {

void TofuCopyFromTo(const nnvm::NodeAttrs& attrs,
                    std::shared_ptr<exec::OpExecutor> op_exec,
                    Engine::VarHandle finish_var,
                    int priority = 0,
                    bool ignore_comm = false);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_EXECUTOR_TOFU_OP_H_
