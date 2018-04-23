#include "./tofu_cached_copy.h"

#include <mxnet/resource.h>
#include "../ndarray/ndarray_function.h"

namespace mxnet {
namespace exec {

void TofuCachedCopy(const nnvm::NodeAttrs& attrs,
                    const NDArray& from,
                    NDArray* to,
                    int priority) {
  NDArray ret = *to;
  int a = from.ctx().dev_mask();
  int b = to->ctx().dev_mask();
  CHECK(a == gpu::kDevMask && b == gpu::kDevMask);

  std::vector<Engine::VarHandle> const_vars;
  if (from.var() != ret.var()) const_vars.push_back(from.var());
  Engine::Get()->PushSync([from, ret](RunContext ctx) {
      ret.CheckAndAlloc();
      LOG(INFO) << "Called tofu copy op!!!";
      TBlob tmp = ret.data();
      ndarray::Copy<gpu, gpu>(from.data(), &tmp,
                              from.ctx(), ret.ctx(), ctx);
      // Wait GPU kernel to complete
      ctx.get_stream<gpu>()->Wait();
    }, from.ctx(), const_vars, {ret.var()},
    from.dtype() != ret.dtype() ? FnProperty::kNormal : FnProperty::kCopyFromGPU,
    priority, PROFILER_MESSAGE("CopyGPU2GPU"));
}

}  // namespace exec
}  // namespace mxnet
