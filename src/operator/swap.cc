/*!
 * Copyright (c) 2017 by Contributors
 * \file p2pnet_send.cc
 * \brief
 * \author Chien-Chin Huang
*/
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/sa_util.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <zmq.h>
#include "./operator_common.h"
#include "./swap-inl.h"
#include "../storage/swapadv_mm_dptr.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SwapOpParam);

// Ugly but efficient
bool swap_doit = false;

inline bool SwapEntryInferShape(const nnvm::NodeAttrs& attrs,
                                std::vector<TShape> *in_shapes,
                                std::vector<TShape> *out_shapes) {
  // Avoid unused variable warnings.
  (void)attrs;
  CHECK_EQ(in_shapes->size(), 1);
  CHECK_EQ(out_shapes->size(), 1);
  TShape outshape(1);
  outshape[0] = 1;
  SHAPE_ASSIGN_CHECK(*out_shapes, 0, outshape);
  return true;
}

inline bool SwapEntryInferType(const nnvm::NodeAttrs& attrs,
                               std::vector<int> *in_types,
                               std::vector<int> *out_types) {
  (void)attrs;
  CHECK_EQ(in_types->size(), 1);
  CHECK_EQ(out_types->size(), 1);
  TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
  return true;
}

void SwapEntryCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  (void)attrs;
  (void)ctx;
  (void)inputs;
  (void)req;
  (void)outputs;
  const char *type = getenv("MXNET_GPU_MEM_POOL_TYPE");
  if (type != nullptr && strcmp(type, "SwapAdv") == 0) {
    swap_doit = true;
  }
  std::cout << "SwapEntryCompute " << swap_doit << std::endl;
}

NNVM_REGISTER_OP(SwapEntry)
  .set_num_inputs(0)
  .set_num_outputs(1)
  .set_attr<FCompute>("FCompute<gpu>", SwapEntryCompute)
  .set_attr<nnvm::FInferShape>("FInferShape", SwapEntryInferShape)
  .set_attr<nnvm::FInferType>("FInferType", SwapEntryInferType)
  .describe("Special op to be swap nodes entry.");

inline bool SwapoutSinkInferShape(const nnvm::NodeAttrs& attrs,
                                  std::vector<TShape> *in_shapes,
                                  std::vector<TShape> *out_shapes) {
  // Avoid unused variable warnings.
  (void)attrs;
  CHECK_EQ(in_shapes->size(), 0);
  CHECK_EQ(out_shapes->size(), 1);
  TShape outshape(1);
  outshape[0] = 1;
  SHAPE_ASSIGN_CHECK(*out_shapes, 0, outshape);
  return true;
}

inline bool SwapoutSinkInferType(const nnvm::NodeAttrs& attrs,
                                 std::vector<int> *in_types,
                                 std::vector<int> *out_types) {
  (void)attrs;
  CHECK_EQ(in_types->size(), 0);
  CHECK_EQ(out_types->size(), 1);
  TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
  return true;
}

void SwapoutSinkCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  (void)attrs;
  (void)ctx;
  (void)inputs;
  (void)req;
  (void)outputs;
  std::cout << "SwapoutSinkCompute" << std::endl;
}

NNVM_REGISTER_OP(SwapoutSink)
  .set_num_inputs(0)
  .set_num_outputs(1)
  .set_attr<FCompute>("FCompute<gpu>", SwapoutSinkCompute)
  .set_attr<nnvm::FInferShape>("FInferShape", SwapoutSinkInferShape)
  .set_attr<nnvm::FInferType>("FInferType", SwapoutSinkInferType)
  .describe("Special op to be swapout sink.");
  //.add_arguments(P2PNetSendParam::__FIELDS__())

inline bool SwapoutInferShape(const nnvm::NodeAttrs& attrs,
                              std::vector<TShape> *in_shapes,
                              std::vector<TShape> *out_shapes) {
  // Avoid unused variable warnings.
  (void)attrs;
  CHECK_EQ(in_shapes->size(), 0);
  CHECK_EQ(out_shapes->size(), 1);
  TShape outshape(1);
  outshape[0] = 1;
  SHAPE_ASSIGN_CHECK(*out_shapes, 0, outshape);
  return true;
}

inline bool SwapoutInferType(const nnvm::NodeAttrs& attrs,
                               std::vector<int> *in_types,
                               std::vector<int> *out_types) {
  (void)attrs;
  CHECK_EQ(in_types->size(), 0);
  CHECK_EQ(out_types->size(), 1);
  TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
  return true;
}

void SwapoutCompute(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  const SwapOpParam& param = nnvm::get<SwapOpParam>(attrs.parsed);
  (void)ctx;
  (void)inputs;
  (void)req;
  (void)outputs;
  sa_log << "SwapoutCompute src = (" << param.src_tensor_nid << ", "
         << param.src_tensor_idx << ")" << std::endl;
  if (sa_likely(swap_doit)) {
    storage::SA_MM_DPTR()->Swapout(param.src_tensor_nid, param.src_tensor_idx);
  }
}

NNVM_REGISTER_OP(Swapout)
  .set_num_inputs(0)
  .set_num_outputs(1)
  .set_attr_parser(ParamParser<SwapOpParam>)
  .set_attr<FCompute>("FCompute<gpu>", SwapoutCompute)
  .set_attr<nnvm::FInferShape>("FInferShape", SwapoutInferShape)
  .set_attr<nnvm::FInferType>("FInferType", SwapoutInferType)
  .describe("Swapout operator.");

inline bool SwapinInferShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_shapes,
                             std::vector<TShape> *out_shapes) {
  // Avoid unused variable warnings.
  (void)attrs;
  CHECK_EQ(in_shapes->size(), 0);
  CHECK_EQ(out_shapes->size(), 1);
  TShape outshape(1);
  outshape[0] = 1;
  SHAPE_ASSIGN_CHECK(*out_shapes, 0, outshape);
  return true;
}

inline bool SwapinInferType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_types,
                            std::vector<int> *out_types) {
  (void)attrs;
  CHECK_EQ(in_types->size(), 0);
  CHECK_EQ(out_types->size(), 1);
  TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
  return true;
}

void SwapinCompute(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  const SwapOpParam& param = nnvm::get<SwapOpParam>(attrs.parsed);
  (void)ctx;
  (void)inputs;
  (void)req;
  (void)outputs;
  sa_log << "SwapinCompute src = (" << param.src_tensor_nid << ", "
         << param.src_tensor_idx << ")" << std::endl;
  if (sa_likely(swap_doit)) {
    storage::SA_MM_DPTR()->Swapin(param.src_tensor_nid, param.src_tensor_idx);
  }
}

NNVM_REGISTER_OP(Swapin)
  .set_num_inputs(0)
  .set_num_outputs(1)
  .set_attr_parser(ParamParser<SwapOpParam>)
  .set_attr<FCompute>("FCompute<gpu>", SwapinCompute)
  .set_attr<nnvm::FInferShape>("FInferShape", SwapinInferShape)
  .set_attr<nnvm::FInferType>("FInferType", SwapinInferType)
  .describe("Swapin operator.");

}  // namespace op
}  // namespace mxnet
