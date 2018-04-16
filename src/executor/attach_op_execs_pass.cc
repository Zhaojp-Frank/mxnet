/*!
 * Copyright (c) 2016 by Contributors
 * \file attach_op_execs_pass.cc
 * \brief Operator executor to execute each operator.
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include "./exec_pass.h"
#include "../nnvm/legacy_op_util.h"

using namespace std;

namespace mxnet {
namespace exec {

namespace {
void DoNothingFCompute(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  // DO nothing.
}
inline bool StartsWith(const string& value, const string& starting) {
  if (starting.size() > value.size()) return false;
  return std::equal(starting.begin(), starting.end(), value.begin());
}
}

// forward executor
class ForwardOpExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx) override {
    this->Setup();
    op_ctx.run_ctx = rctx;
    op_->Forward(op_ctx, in_data_, req, out_data_, aux_data_);
  }

  void Setup() override {
    in_data_.clear(); aux_data_.clear();
    for (size_t i = 0; i < in_array.size(); ++i) {
      if (!std::binary_search(aux_index_.begin(), aux_index_.end(), i)) {
        in_data_.push_back(in_array[i].data());
      } else {
        aux_data_.push_back(in_array[i].data());
      }
    }
    out_data_.resize(out_array.size());
    std::transform(out_array.begin(), out_array.end(), out_data_.begin(), [](const NDArray& nd) {
        return nd.data();
      });
  }
  Operator::ExecType exec_type() const override {
    return op_->exec_type();
  }
  explicit ForwardOpExecutor(shared_ptr<Operator> op, const vector<uint32_t>& aux_index)
      : op_(op), aux_index_(aux_index) {
    std::sort(aux_index_.begin(), aux_index_.end());
  }

 private:
  friend Graph AttachOpExecs(Graph g);
  shared_ptr<Operator> op_;
  vector<uint32_t> aux_index_;
  vector<TBlob> in_data_, out_data_, aux_data_;
};

// backward executor
class BackwardOpExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx) override {
    op_ctx.run_ctx = rctx;
    this->Setup();
    op_->Backward(op_ctx, out_grad_, in_data_, out_data_,
                  req, in_grad_, aux_data_);
  }
  void Setup() override {
    size_t arg_top = 0, aux_top = 0;
    aux_data_.resize(aux_index_.size());
    for (size_t i = 0; i < in_array.size(); ++i) {
      if (!std::binary_search(aux_index_.begin(), aux_index_.end(), i)) {
        CHECK_GT(arg_data_ptr_.size(), arg_top);
        *arg_data_ptr_[arg_top++] = in_array[i].data();
      } else {
        aux_data_.at(aux_top++) = in_array[i].data();
      }
    }
    CHECK_EQ(out_array.size(), in_grad_.size());
    std::transform(out_array.begin(), out_array.end(),
                   in_grad_.begin(), [](const NDArray& nd) {
        return nd.data();
      });
  }
  Operator::ExecType exec_type() const override {
    return op_->exec_type();
  }
  explicit BackwardOpExecutor(shared_ptr<Operator> op,
                              const OperatorProperty& prop,
                              const vector<uint32_t>& aux_index)
      : op_(op), aux_index_(aux_index) {
    std::sort(aux_index_.begin(), aux_index_.end());
    out_grad_.resize(prop.NumVisibleOutputs());
    in_data_.resize(prop.ListArguments().size());
    in_grad_.resize(in_data_.size());
    out_data_.resize(prop.NumOutputs());

    vector<TBlob*> out_grad_ptr(out_grad_.size());
    for (size_t i = 0; i < out_grad_.size(); ++i) {
      out_grad_ptr[i] = &out_grad_[i];
    }
    vector<TBlob*> in_data_ptr(in_data_.size());
    for (size_t i = 0; i < in_data_.size(); ++i) {
      in_data_ptr[i] = &in_data_[i];
    }
    vector<TBlob*> out_data_ptr(out_data_.size());
    for (size_t i = 0; i < out_data_.size(); ++i) {
      out_data_ptr[i] = &out_data_[i];
    }
    arg_data_ptr_ = prop.BackwardInputs(
        out_grad_ptr, in_data_ptr, out_data_ptr);
  }

 private:
  shared_ptr<Operator> op_;
  vector<uint32_t> aux_index_;
  vector<TBlob> out_grad_, in_grad_, in_data_, out_data_, aux_data_;
  vector<TBlob*> arg_data_ptr_;
};

// fcompute executor executor
class FComputeExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx) override {
    op_ctx.run_ctx = rctx;
    this->Setup();
    fcompute_(attrs_, op_ctx, in_data_, req, out_data_);
  }
  void Setup() override {
    in_data_.resize(in_array.size());
    out_data_.resize(out_array.size());
    auto get_blob =  [](const NDArray& nd) {
      return nd.data();
    };
    std::transform(in_array.begin(), in_array.end(), in_data_.begin(), get_blob);
    std::transform(out_array.begin(), out_array.end(), out_data_.begin(), get_blob);
  }
  Operator::ExecType exec_type() const override {
    return Operator::kSync;
  }
  explicit FComputeExecutor(FCompute fcompute, const NodeAttrs& attrs)
      : fcompute_(fcompute), attrs_(attrs) {
  }

  static FCompute GetFCompute(const Op* op, Context ctx) {
    static auto& fcompute_cpu = nnvm::Op::GetAttr<FCompute>("FCompute<cpu>");
    static auto& fcompute_gpu = nnvm::Op::GetAttr<FCompute>("FCompute<gpu>");
    if (ctx.dev_mask() == cpu::kDevMask) {
      return fcompute_cpu.get(op, nullptr);
    } else if (ctx.dev_mask() == gpu::kDevMask) {
      return fcompute_gpu.get(op, nullptr);
    } else {
      LOG(FATAL) << "Unknown device mask";
      return nullptr;
    }
  }

 private:
  FCompute fcompute_;
  NodeAttrs attrs_;
  vector<TBlob> in_data_, out_data_;
};

const OperatorProperty& ParseOpProp(const NodeAttrs& attrs) {
  return *nnvm::get<mxnet::op::ParsedOpProp>(attrs.parsed).ptr.get();
}

inline vector<uint32_t> OpPropMutateInputs(const OperatorProperty& prop) {
  vector<uint32_t> ret;
  for (uint32_t i = 0; i < prop.ListAuxiliaryStates().size(); ++i) {
    ret.push_back(static_cast<uint32_t>(i + prop.ListArguments().size()));
  }
  return ret;
}

inline std::vector<uint32_t> OpPropBackMutateInputs(const OperatorProperty& prop) {
  if (prop.ListAuxiliaryStates().size() == 0) {
    return {};
  }
  std::vector<int> out_grad_index(prop.NumVisibleOutputs());
  std::vector<int> in_data_index(prop.ListArguments().size());
  std::vector<int> out_data_index(prop.ListOutputs().size());
  size_t arg_size = prop.DeclareBackwardDependency(
      out_grad_index, in_data_index, out_data_index).size();
  std::vector<uint32_t> ret;
  for (uint32_t i = 0; i < prop.ListAuxiliaryStates().size(); ++i) {
    ret.push_back(static_cast<uint32_t>(i + arg_size));
  }
  return ret;
}

inline Operator* OpPropCreateLayerOp(
    const OperatorProperty& prop,
    const Context& ctx,
    const vector<TShape>& ishape,
    const vector<int>& itype) {
  const size_t num_args = prop.ListArguments().size();
  vector<TShape> is(ishape.begin(), ishape.begin() + num_args);
  vector<int> it(itype.begin(), itype.begin() + num_args);
  return prop.CreateOperatorEx(ctx, &is, &it);
}

inline Operator* OpPropCreateBackwardLayerOp(
    const OperatorProperty& prop,
    const Context& ctx,
    const vector<TShape>& ishape,
    const vector<int>& itype,
    const vector<TShape>& oshape,
    const vector<int>& otype) {
  return prop.CreateBackwardOperatorEx(ctx, ishape, itype, oshape, otype);
}

template<typename T>
inline vector<T> ParseInAttrs(
    const nnvm::IndexedGraph& idx,
    const vector<T>& entry_attrs,
    const nnvm::IndexedGraph::Node& inode) {
  vector<T> ret;
  for (const auto& e : inode.inputs) {
    ret.emplace_back(entry_attrs[idx.entry_id(e)]);
  }
  return ret;
}

template<typename T>
inline vector<T> ParseOutAttrs(
    const nnvm::IndexedGraph& idx,
    const vector<T>& entry_attrs,
    const nnvm::IndexedGraph::Node& inode) {
  const uint32_t nid = idx.node_id(inode.source);
  vector<T> ret;
  for (size_t i = 0; i < inode.source->num_outputs(); ++i) {
    ret.emplace_back(entry_attrs[idx.entry_id(nid, i)]);
  }
  return ret;
}

inline bool ExistForwardNode(const nnvm::IndexedGraph::Node& inode) {
  // TODO(minjie): Currently, the first control dependency of a backward
  // node must be its corresponding forward node. A better way is to use
  // some graph attribute to specify that.
  if (inode.source->attrs.dict.count("OriginalControlSize") > 0) {
    return false;
  } 
  return inode.control_deps.size() > 0 ;
}

// pass to attach operator executors
Graph AttachOpExecs(Graph g) {
  using nnvm::DTypeVector;
  using nnvm::ShapeVector;
  using nnvm::FMutateInputs;

  //auto& fcreate_layer_op = nnvm::Op::GetAttr<FCreateLayerOp>("FCreateLayerOp");
  //auto& fmutate_inputs = nnvm::Op::GetAttr<FMutateInputs>("FMutateInputs");
  auto& is_layer_forward = nnvm::Op::GetAttr<bool>("TIsLayerOpForward");
  auto& is_layer_backward = nnvm::Op::GetAttr<bool>("TIsLayerOpBackward");

  const auto& vdtype = g.GetAttr<DTypeVector>("dtype");
  const auto& vshape = g.GetAttr<ShapeVector>("shape");
  const auto& vctx = g.GetAttr<ContextVector>("context");

  // get the graph
  const auto& idx = g.indexed_graph();
  vector<shared_ptr<OpExecutor> > ret(idx.num_nodes());

  const int no_comp_flag = dmlc::GetEnv("TOFU_NO_COMPUTATION", 0);
  if (no_comp_flag) {
    LOG(INFO) << "Enable No Computation Mode.";
  }

  // initialize the nodes
  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    const auto& inode = idx[i];
    if (inode.source->is_variable()) {
      // No need to generate OpExecutor for variable node.
      continue;
    }
    const nnvm::Op* op = CHECK_NOTNULL(inode.source->op());
    if (no_comp_flag && op->name != "P2PNetRecv" && op->name != "P2PNetSend"
        && op->name != "P2PNetInit" && op->name != "P2PNetSendSink") {
	    LOG(INFO) << "Ignore: " << op->name;
      ret[i] = std::make_shared<FComputeExecutor>(DoNothingFCompute, inode.source->attrs);
      continue;
    }

    if (dmlc::GetEnv("MXNET_P2PNET_DEBUG", 0) & 2) {
        if (false
            || op->name == "P2PNetRecv"
            || op->name == "P2PNetSend"
            || op->name == "P2PNetInit"
            || op->name == "P2PNetSendSink"
        ) {
            ret[i] = std::make_shared<FComputeExecutor>(DoNothingFCompute, inode.source->attrs);
            continue;
        }
    }

    //FCompute f_zero_compute = FComputeExecutor::GetFCompute(
        //nnvm::Op::Get("_zeros"), vctx[i]);
    if (dmlc::GetEnv("TOFU_IGNORE_GPU_COMM", 0)) {
      if (op->name == "_CrossDeviceCopy") {
        ret[i] = std::make_shared<FComputeExecutor>(
            DoNothingFCompute, inode.source->attrs);
        continue;
      }
    }
    if (dmlc::GetEnv("TOFU_IGNORE_CONVERSION", 0)) {
      if (StartsWith(inode.source->attrs.name, "_TOFU")) {
        ret[i] = std::make_shared<FComputeExecutor>(
            DoNothingFCompute, inode.source->attrs);
        continue;
      }
    }
    if (dmlc::GetEnv("TOFU_ONLY_GPU0", 0)) {
      if (vctx[i].dev_type != Context::kGPU || vctx[i].dev_id != 0) {
        ret[i] = std::make_shared<FComputeExecutor>(
            DoNothingFCompute, inode.source->attrs);
        continue;
      }
    }
    if (false
        //|| op->name == "_CrossDeviceCopy"
        //|| op->name == "Concat"
        //|| op->name == "_backward_Concat"
        //|| op->name == "ElementWiseSum"
        //|| op->name == "_backward_FullyConnected"
        //|| op->name == "SoftmaxOutput"
        //|| op->name == "_backward_SoftmaxOutput"
        ) {
      ret[i] = std::make_shared<FComputeExecutor>(DoNothingFCompute, inode.source->attrs);
      continue;
    }

    if (is_layer_forward.count(op) || is_layer_backward.count(op)) {
      // Layer operator.
      const OperatorProperty& prop = ParseOpProp(inode.source->attrs);
      const vector<TShape>& ishape = ParseInAttrs(idx, vshape, inode);
      const vector<int>& itype = ParseInAttrs(idx, vdtype, inode);
      const vector<TShape>& oshape = ParseOutAttrs(idx, vshape, inode);
      const vector<int>& otype = ParseOutAttrs(idx, vdtype, inode);
      if (is_layer_forward.count(op)) {
        // Forward operator.
        const vector<uint32_t>& mutate_index = OpPropMutateInputs(prop);
        shared_ptr<Operator> layer_fwd_op(OpPropCreateLayerOp(prop, vctx[i], ishape, itype));
        ret[i] = std::make_shared<ForwardOpExecutor>(layer_fwd_op, mutate_index);
/*
      } else if (ExistForwardNode(inode)) {
        // Backward operator that has corresponding forward operator. Reuse the already created
        // forward OpExecutor.
        // TODO(minjie): Currently, the first control dependency of a backward
        // node must be its corresponding forward node. A better way is to use
        // some graph attribute to specify that.
        const vector<uint32_t>& mutate_index = OpPropBackMutateInputs(prop);
        const uint32_t fwd_id = inode.control_deps[0];
        CHECK(vctx[fwd_id] == vctx[i])
          << "Stateful backward node requires to have the same device context with the forward node.";
        CHECK(ret[fwd_id] != nullptr)
          << "Stateful backward node requires its forward OpExecutor to be created.";
        ret[i] = std::make_shared<BackwardOpExecutor>(
            std::dynamic_pointer_cast<ForwardOpExecutor>(ret[fwd_id])->op_,
            prop,
            mutate_index);
*/
      } else {
        // Backward operator that has no corresponding forward operator. Try to create the OpExecutor
        // by its own.
        const vector<uint32_t>& mutate_index = OpPropBackMutateInputs(prop);
        shared_ptr<Operator> layer_bwd_op(OpPropCreateBackwardLayerOp(
            prop, vctx[i], ishape, itype, oshape, otype));
        CHECK(layer_bwd_op != nullptr)
          << "Failed to create executor for the backward node (op: " << op->name << ").";
        ret[i] = std::make_shared<BackwardOpExecutor>(layer_bwd_op, prop, mutate_index);
      }
    } else {
      FCompute fcompute = FComputeExecutor::GetFCompute(inode.source->op(), vctx[i]);
      CHECK(fcompute != nullptr) << "FCompute not registered for op: " << inode.source->op()->name;
      ret[i] = std::make_shared<FComputeExecutor>(fcompute, inode.source->attrs);
    }
  }
  g.attrs["op_execs"] = std::make_shared<nnvm::any>(ret);
  return g;
}

}  // namespace exec
}  // namespace mxnet
