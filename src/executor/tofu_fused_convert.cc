#include <nnvm/op.h>
#include <nnvm/node.h>
#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/scheme.h>

using namespace std;

namespace mxnet {
namespace op {

void TofuFusedConvert(
    const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<TBlob>& outputs) {
  LOG(FATAL) << "Unreachable";
  const auto& info = nnvm::get<pair<vector<TShape>, vector<TShape>>>(attrs.parsed);
  const vector<TShape>& offsets = info.first;
  const vector<TShape>& sizes = info.second;
  CHECK_EQ(offsets.size(), inputs.size());
  CHECK_EQ(sizes.size(), inputs.size());
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(outputs[0].dev_mask_, gpu::kDevMask);
  for (size_t i = 0; i < inputs.size(); ++i) {
    CHECK_EQ(inputs[i].dev_mask_, gpu::kDevMask);
    cudaError_t err = cudaMemcpy(
        outputs[0].dptr_,
        inputs[i].dptr_,
        sizes[i].Size() * sizeof(float),
        cudaMemcpyDeviceToDevice);
    CHECK(err == cudaSuccess)
      << "Error in tofu fused convert: " << err;
  }
}

void TofuDoNothing(
    const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<TBlob>& outputs) {
  // Do nothing. 
}

void TofuCopyFromTo(const nnvm::NodeAttrs& attrs,
                    const vector<NDArray>& from,
                    NDArray* to,
                    int priority) {
  NDArray ret = *to;
  CHECK_GT(from.size(), 0);
  int a = from[0].ctx().dev_mask();
  int b = to->ctx().dev_mask();
  CHECK(a == gpu::kDevMask && b == gpu::kDevMask);
  std::vector<Engine::VarHandle> const_vars;
  for (const auto& f : from) {
    const_vars.push_back(f.var());
  }
  const auto& param = nnvm::get<nnvm::pass::TofuConvertParam>(attrs.parsed);
  const vector<TShape>& offsets = param.offsets;
  const vector<TShape>& sizes = param.sizes;
  CHECK_EQ(offsets.size(), from.size());
  CHECK_EQ(sizes.size(), from.size());
  Engine::Get()->PushSync([from, ret, param](RunContext ctx) {
    ret.CheckAndAlloc();
    for (size_t i = 0; i < from.size(); ++i) {
      const auto& f = from[i];
      if (f.ctx().dev_id == ret.ctx().dev_id) {
        // Ignored.
      } else {
        mshadow::Stream<gpu> *s = static_cast<mshadow::Stream<gpu>*>(ctx.stream);
        CHECK(s != NULL) << "need stream in GPU context";
        size_t copy_size = param.sizes[i].Size() * mshadow::mshadow_sizeof(ret.data().type_flag_);
        cudaMemcpyPeerAsync(ret.data().dptr_,
                            ret.ctx().dev_id,
                            f.data().dptr_,
                            f.ctx().dev_id,
                            copy_size,
                        s->stream_);
      }
    }
    // Wait GPU kernel to complete
    ctx.get_stream<gpu>()->Wait();
  }, ret.ctx(), const_vars, {ret.var()},
  FnProperty::kCopyFromGPU,
  priority,
  PROFILER_MESSAGE("TofuFusedConvert"));
}

void TofuCopyFromToNoComm(const nnvm::NodeAttrs& attrs,
                          const std::vector<NDArray>& from,
                          NDArray* to,
                          int priority = 0) {
  // Do nothing.
}

NNVM_REGISTER_OP(_TofuFusedConvert)
.set_num_outputs(1)
.describe("Operator used for tensor scheme conversion.")
.set_attr<FCompute>("FCompute<gpu>", TofuFusedConvert);

NNVM_REGISTER_OP(_TofuFusedConvertNoComm)
.set_num_outputs(1)
.describe("Operator used for tensor scheme conversion.")
.set_attr<FCompute>("FCompute<gpu>", TofuDoNothing);

NNVM_REGISTER_OP(_TofuFakeVar)
 .set_num_inputs(0)
 .set_num_outputs(1)
 .set_attr<FCompute>("FCompute<cpu>", TofuDoNothing)
 .set_attr<FCompute>("FCompute<gpu>", TofuDoNothing) ;

NNVM_REGISTER_OP(_TofuFakeOut)
 .set_num_outputs(1)
 .set_attr<FCompute>("FCompute<cpu>", TofuDoNothing)
 .set_attr<FCompute>("FCompute<gpu>", TofuDoNothing) ;

}  // namespace op
}  // namespace mxnet
