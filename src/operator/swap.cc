/*!
 * Copyright (c) 2017 by Contributors
 * \file p2pnet_send.cc
 * \brief
 * \author Chien-Chin Huang
*/
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <zmq.h>
#include "./operator_common.h"

namespace mxnet {
namespace op {
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
  std::cout << "SwapEntryCompute" << std::endl;
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
  std::cout << "SwapSinkCompute" << std::endl;
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
  (void)attrs;
  (void)ctx;
  (void)inputs;
  (void)req;
  (void)outputs;
#if SWAP_ADVISOR_FLOW_TRACE
  std::cout << "SwapoutCompute" << std::endl;
#endif
}

NNVM_REGISTER_OP(Swapout)
  .set_num_inputs(0)
  .set_num_outputs(1)
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
  (void)attrs;
  (void)ctx;
  (void)inputs;
  (void)req;
  (void)outputs;
#if SWAP_ADVISOR_FLOW_TRACE
  std::cout << "SwapinCompute" << std::endl;
#endif
}

NNVM_REGISTER_OP(Swapin)
  .set_num_inputs(0)
  .set_num_outputs(1)
  .set_attr<FCompute>("FCompute<gpu>", SwapinCompute)
  .set_attr<nnvm::FInferShape>("FInferShape", SwapinInferShape)
  .set_attr<nnvm::FInferType>("FInferType", SwapinInferType)
  .describe("Swapin operator.");

}  // namespace op
}  // namespace mxnet
