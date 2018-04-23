#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/resource.h>
#include "../ndarray/ndarray_function.h"
#include "../operator/operator_common.h"

namespace mxnet {
namespace exec {

struct TofuCachedCopyParam : public dmlc::Parameter<TofuCachedCopyParam> {
  int tensor_id;
  TShape offset;
  TShape size;
  DMLC_DECLARE_PARAMETER(TofuCachedCopyParam) {
    DMLC_DECLARE_FIELD(tensor_id).set_default(-1)
      .describe("The global unique tensor id.");
    DMLC_DECLARE_FIELD(offset).set_default(TShape())
      .describe("The offset of the copy region.");
    DMLC_DECLARE_FIELD(size).set_default(TShape())
      .describe("The size of the copy region.");
  }
};

DMLC_REGISTER_PARAMETER(TofuCachedCopyParam);

void TofuCachedCopy(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), 1);
  const auto& param = nnvm::get<TofuCachedCopyParam>(attrs.parsed);
  LOG(INFO) << "Called tofu copy op!!! tensor_id="
    << param.tensor_id << " offset=" << param.offset
    << " size=" << param.size;
}

NNVM_REGISTER_OP(_TofuCachedCopy)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(op::ParamParser<TofuCachedCopyParam>)
.set_attr<FCompute>("FCompute<cpu>", TofuCachedCopy)
.set_attr<FCompute>("FCompute<gpu>", TofuCachedCopy);

}  // namespace exec
}  // namespace mxnet
