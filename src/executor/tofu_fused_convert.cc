#include <nnvm/op.h>
#include <nnvm/node.h>
#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/scheme.h>
#include "./tofu_op.h"

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

void TofuFakeOut(
    const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<TBlob>& outputs) {
  // Do nothing. 
  //LOG(INFO) << "FAKE OUT";
}

void TofuCopyFromTo(const nnvm::NodeAttrs& attrs,
                    std::shared_ptr<exec::OpExecutor> op_exec,
                    Engine::VarHandle finish_var,
                    int priority,
                    bool ignore_comm) {
  const auto& from = op_exec->in_array;
  const auto& to = op_exec->out_array[0];
  CHECK_GT(from.size(), 0);
  int a = from[0].ctx().dev_mask();
  int b = to.ctx().dev_mask();
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
  Engine::Get()->PushSync([op_exec, param, ignore_comm] (RunContext ctx) {
    auto& ret = op_exec->out_array[0];
    ret.CheckAndAlloc();
    if (ignore_comm ||
        (param.is_reduction && param.ignore_reduction)) {
      // Do nothing.
      //LOG(INFO) << "Comm ignored!!!";
    } else {
      for (size_t i = 0; i < op_exec->in_array.size(); ++i) {
        const auto& f = op_exec->in_array[i];
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
    }

    // Clear all temp input ndarrays.
    for (size_t i = 0; i < op_exec->in_array.size(); ++i) {
      if (op_exec->in_array_is_temp[i]) {
        op_exec->in_array[i] = NDArray();
      }
    }
  }, to.ctx(), const_vars, {to.var(), finish_var},
  FnProperty::kNormal,
  //FnProperty::kCopyFromGPU,
  priority,
  PROFILER_MESSAGE("TofuFusedConvert"));

  //std::ostringstream oss;
  //oss << "use=[";
  //for (auto& v : const_vars) oss << v << " ";
  //oss << "] mutate=[" << to.var() << " " << finish_var;
  //oss << "]";
  //LOG(INFO) << "TofuFusedConvert " << oss.str();

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
 .set_attr<FCompute>("FCompute<cpu>", TofuFakeOut)
 .set_attr<FCompute>("FCompute<gpu>", TofuFakeOut) ;

}  // namespace op
}  // namespace mxnet
