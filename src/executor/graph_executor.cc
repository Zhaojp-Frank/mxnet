/*!
 *  Copyright (c) 2015 by Contributors
 * \file graph_executor.cc
 * \brief graph executor
 */
#include <mxnet/base.h>
#include <nnvm/graph.h>
#include <nnvm/pass_functions.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>

#include "./dfge_profiling.h"
#include "./exec_pass.h"
#include "./graph_executor.h"
#include "../engine/profiler.h"
#include "../operator/p2pnet_common.h"
#include "./tofu_op.h"

//#define SPLIT_GRADIENT_TEST

namespace mxnet {
namespace exec {
namespace {
void EnableP2P(const std::vector<Context>& devs) {
#if MXNET_USE_CUDA
  std::vector<int> gpus;
  for (const auto& d : devs) {
    if (d.dev_mask() == gpu::kDevMask) {
      gpus.push_back(d.dev_id);
    }
  }
  int n = static_cast<int>(gpus.size());
  int enabled = 0;
  std::vector<int> p2p(n*n);
  for (int i = 0; i < n; ++i) {
    cudaSetDevice(gpus[i]);
    for (int j = 0; j < n; j++) {
      int access;
      cudaDeviceCanAccessPeer(&access, gpus[i], gpus[j]);
      if (access) {
        cudaError_t e = cudaDeviceEnablePeerAccess(gpus[j], 0);
        if (e == cudaSuccess) {
          ++enabled;
          p2p[i*n+j] = 1;
        }
      }
    }
  }
  if (enabled != n*(n-1)) {
    // print warning info if not fully enabled
    LOG(WARNING) << "only " << enabled <<  " out of "
      << n*(n-1) << " GPU pairs are enabled direct access. "
      << "It may affect the performance. "
      << "You can set MXNET_ENABLE_GPU_P2P=0 to turn it off";
  }
  std::string access(n, '.');
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      int succ;
      cudaDeviceCanAccessPeer(&succ, gpus[i], gpus[j]);
      access[j] = succ ? 'v' : '.';
    }
    LOG(INFO) << access;
  }
#endif
}
}  // namespace

GraphExecutor::~GraphExecutor() {
  for (auto& n : op_nodes_) {
    if (n.cached_opr != nullptr) {
      Engine::Get()->DeleteOperator(n.cached_opr);
    }
  }
}

void GraphExecutor::Forward(bool is_train) {
  DFGEProfiler::Get().Begin();
  DFGEProfiler::Get().Write(-1, false, false);
  RunOps(is_train, 0, num_forward_nodes_);
}

void GraphExecutor::PartialForward(bool is_train, int step, int *step_left) {
  size_t sstep = static_cast<size_t>(step);
  if (sstep >= num_forward_nodes_) {
    *step_left = 0; return;
  }
  RunOps(is_train, sstep, sstep + 1);
  *step_left = static_cast<int>(num_forward_nodes_ - sstep - 1);
}

void GraphExecutor::Backward(const std::vector<NDArray>& head_grads) {
  const auto& idx = graph_.indexed_graph();
  if (num_forward_inputs_ != idx.input_nodes().size()) {
    for (size_t i = 0; i < head_grad_array_.size(); ++i) {
      if (!head_grad_array_[i].is_none()) {
        //CHECK(i < head_grads.size() && !head_grads[i].is_none())
            //<< "Because the last operator is not Loss function, "
            //<< "head_gradient is required in calling backward.";
        //LOG(INFO) << "[WARNING!!!] Copy is turned off for header grad";
        // TODO: hack
        //CopyFromTo(head_grads[i], &(head_grad_array_[i]));
      }
    }
  }
  RunOps(true, num_forward_nodes_, idx.num_nodes());
}

void GraphExecutor::Print(std::ostream &os) const {  // NOLINT(*)
  nnvm::Symbol s; s.outputs = graph_.outputs;
  s.Print(os);
  // message to be backward compatible with the memonger
  size_t total_bytes = graph_.GetAttr<size_t>("storage_allocated_bytes");
  os << "Total " << (total_bytes >> 20UL) <<" MB allocated\n";
  os << "Total " << 11 << " TempSpace resource requested\n";
}

void GraphExecutor::SetMonitorCallback(const MonitorCallback& callback) {
  CHECK(callback) << "invalid callback";
  monitor_callback_ = callback;
}

const std::vector<NDArray>& GraphExecutor::outputs() const {
  return output_arrays_;
}

nnvm::NodeEntry AttrHint(nnvm::NodeEntry src, nnvm::NodeEntry like) {
  //static const Op* id_like = Op::Get("_identity_with_attr_like_rhs");
  //nnvm::NodePtr n = nnvm::Node::Create();
  //n->attrs.op = id_like;
  //n->attrs.name = src.node->attrs.name + "_id";
  //n->inputs = {src, like};
  //return nnvm::NodeEntry{n, 0, 0};
  return src;
}

nnvm::NodeEntry AggregateGradient(const std::vector<nnvm::NodeEntry>& vv) {
  using nnvm::Op;
  static size_t inplace_sum_cap = dmlc::GetEnv("MXNET_EXEC_INPLACE_GRAD_SUM_CAP", 8);
  static const Op* ewise_plus_op = Op::Get("_grad_add");
  static const Op* ewise_sum_op = Op::Get("ElementWiseSum");
  static const Op* identity_op = Op::Get("identity");
  static const Op* zeros_op = Op::Get("_zeros");
  std::vector<nnvm::NodeEntry> v = vv; // copy the list.
  // remove zero in the sum.
  size_t begin = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    if (v[i].node->op() != zeros_op) {
      if (begin != i) {
        v[begin] = std::move(v[i]);
      }
      ++begin;
    }
  }
  v.resize(begin);

  if (v.size() == 1) {
    return std::move(v[0]);
  } else if (v.size() == 0) {
    nnvm::NodePtr ng = nnvm::Node::Create();
    ng->attrs.op = zeros_op;
    ng->attrs.name = "zeros";
    ng->attrs.op->attr_parser(&(ng->attrs));
    return nnvm::NodeEntry{ng, 0, 0};
  } else {
    if (v.size() < inplace_sum_cap) {
      nnvm::NodePtr sum_node = nnvm::Node::Create();
      sum_node->attrs.op = ewise_sum_op;
      sum_node->attrs.name = "sum_grad";
      sum_node->attrs.dict["num_args"] = std::to_string(v.size());
      sum_node->attrs.op->attr_parser(&(sum_node->attrs));
      sum_node->inputs = std::move(v);
      return nnvm::NodeEntry{sum_node, 0, 0};
    } else {
      // use a stream line of plus instead
      nnvm::NodeEntry ret = v[0];
      for (size_t i = 1; i < v.size(); ++i) {
        // Add control flow dependency from to previous node
        // This enforces the gradient sum order will be in the inverse
        // order of forward traversal
        // NOTE: adding control dependency can be dangerous and cause cycle in the dep.
        // The curent usage is correct, because of the following invariant:
        // assert: v[i-1] do not depend on v[i]
        // To put in plain text: v is gradient vector that get pushed in the order
        // that can generate them, which means if v[i] is not yet pushed,
        // all previous gradient cannot depend on it.
        v[i].node->control_deps.push_back(ret.node);

        std::ostringstream os;
        os << "sum_grad_" << i;
        nnvm::NodePtr x = nnvm::Node::Create();
        x->attrs.op = ewise_plus_op;
        x->attrs.name = os.str();
        x->inputs = {ret, v[i]};
        ret = nnvm::NodeEntry{x, 0, 0};
      }
      // identity node is used to avoid exposure of dummy plus node
      // when its output get assigned to another space.
      nnvm::NodePtr id_node = nnvm::Node::Create();
      id_node->attrs.op = identity_op;
      id_node->attrs.name = "sum_grad_final";
      id_node->inputs = {ret};
      return nnvm::NodeEntry{id_node, 0, 0};
    }
  }
}

template<typename ValueType>
inline ValueType get_node_attr(
    const nnvm::Node& node,
    const std::string& key, ValueType default_value) {
  auto it = node.attrs.dict.find(key);
  if (it == node.attrs.dict.end()) {
    return default_value;
  } else {
    ValueType ret;
    dmlc::parameter::FieldEntry<ValueType> e;
    e.Init(key, &ret, ret);
    e.Set(&ret, it->second);
    return ret;
  }
}

Graph GraphExecutor::SplitDistributedGraph(Graph& g, const Context& default_ctx)
{

  const auto& idx = g.indexed_graph();
  nnvm::AddressVector address_vec;
  const ContextVector& context_vec = g.GetAttr<ContextVector>("context");
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    address_vec.push_back(context_vec[nid].dev_address);
  }
  std::set<std::string> address_set(address_vec.begin(), address_vec.end());
  if (address_set.size() == 1) {
    LOG(INFO) << "Only one machine is involved. No need to split graph.";
    return g;
  }
  //std::cout << "==================== Original Graph ====================" << std::endl;
  //DFSVisit(g.outputs, [&g, &idx, & address_vec] (const nnvm::NodePtr& n) {
    //std::cout << n->attrs.name << "(" << idx.node_id(n.get()) << ")"
              //<< " " << address_vec[idx.node_id(n.get())]
              //<< " : ";
    //for (const auto e : n->inputs) {
      //std::cout << e.node->attrs.name << "(" << idx.node_id(e.node.get()) << ")" << "_" << e.index << " : ";
      //std::cout << g.GetAttr<nnvm::ShapeVector>("shape")[idx.entry_id(e)]
                //<< ", ";
    //}
    //std::cout << std::endl;
    ////std::cout << n->attrs.name << " : ";
    ////for (const auto dep_n : n->control_deps) {
      ////std::cout << dep_n->attrs.name << ", ";
    ////}
    ////std::cout << std::endl;
  //});

  g = nnvm::pass::SplitDistributedGraph(g, address_vec, default_ctx.dev_address,
                                        g.GetAttr<nnvm::ShapeVector>("shape"),
                                        g.GetAttr<nnvm::DTypeVector>("dtype"),
                                        copy_op_name_, "P2PNetInit",
                                        "P2PNetSend", "P2PNetRecv", "P2PNetSendSink",
                                        &num_forward_inputs_, &num_forward_outputs_);
  // Renews everything
  ContextVector new_context_vec;
  const nnvm::NodeIdMap& node_id_map =
      g.GetAttr<nnvm::NodeIdMap>("node_id_map");
  const auto& new_idx = g.indexed_graph();
  for (uint32_t nid = 0; nid < new_idx.num_nodes(); ++nid) {
    const auto it = node_id_map.find(nid);
    if (it == node_id_map.end()) {
      // TODO: If this is a new node, we simply assign the default context to
      // it. A better way is to find out which node is using it as input and
      // set the context to be the same as that node.
      new_context_vec.push_back(default_ctx);
    } else {
      new_context_vec.push_back(context_vec[it->second]);
    }
  }
  g.attrs["context"] = std::make_shared<dmlc::any>(std::move(new_context_vec));

  const nnvm::EntryIdMap& entry_id_map =
      g.GetAttr<nnvm::EntryIdMap>("entry_id_map");
  const auto& dtype_vec = g.GetAttr<nnvm::DTypeVector>("dtype");
  const auto& shape_vec = g.GetAttr<nnvm::ShapeVector>("shape");
  std::vector<NDArray> new_data_entry(new_idx.num_node_entries());
  for (uint32_t nid = 0; nid < new_idx.num_nodes(); ++nid) {
    const size_t num_outputs = new_idx[nid].source->num_outputs();
    for (size_t output_idx = 0; output_idx < num_outputs; output_idx++) {
      const uint32_t eid = new_idx.entry_id(nid, output_idx);
      const auto old_eid_it = entry_id_map.find(eid);
      if (old_eid_it != entry_id_map.end()) {
        new_data_entry[eid] = data_entry_[old_eid_it->second];
      } else {
        new_data_entry[eid] = NDArray(shape_vec[eid], default_ctx,
                                      false, dtype_vec[eid]);
      }
    }
  }
  data_entry_ = new_data_entry;

  const nnvm::OutputIdxMap& output_idx_reverse_map =
      g.GetAttr<nnvm::OutputIdxMap>("output_idx_reverse_map");
  std::unordered_map<const nnvm::Node*, size_t> new_head_grad_map;
  for (auto kv: head_grad_map_) {
    auto it = output_idx_reverse_map.find(kv.second);
    if (it != output_idx_reverse_map.end()) {
      new_head_grad_map[kv.first] = it->second;
    }
  }
  head_grad_map_ = new_head_grad_map;

  const uint32_t grad_output_size =
      new_idx.outputs().size() - num_forward_outputs_;
  std::vector<std::pair<OpReqType, NDArray> > new_grad_store(grad_output_size);
  for (uint32_t j = num_forward_outputs_, k = 0; j < new_idx.outputs().size(); j++) {
      const uint32_t eid = new_idx.entry_id(new_idx.outputs()[j]);
      const auto old_eid_it = entry_id_map.find(eid);
    if (old_eid_it != entry_id_map.end()) {
      new_grad_store[j - num_forward_outputs_] = grad_store_[k++];
    }
  }
  grad_store_ = new_grad_store;
  std::cout << "SplitDistributedGraph finished" << std::endl;

  //std::cout << "==================== New Graph ====================" << std::endl;
  //DFSVisit(g.outputs, [&g, &new_idx] (const nnvm::NodePtr& n) {
    //std::cout << n->attrs.name << " : ";
    //for (const auto e : n->inputs) {
      //std::cout << e.node->attrs.name << "_" << e.index << " : ";
      //std::cout << g.GetAttr<nnvm::ShapeVector>("shape")[new_idx.entry_id(e)]
                //<< ", ";
    //}
    //std::cout << std::endl;
    ////std::cout << n->attrs.name << " : ";
    ////for (const auto dep_n : n->control_deps) {
      ////std::cout << dep_n->attrs.name << ", ";
    ////}
    ////std::cout << std::endl;
  //});
  //std::cout << std::endl;
  // head_grad_entry_ will not be used anymore.
  // head_grad_array will be initialized later.
  return g;
}

nnvm::Graph GraphExecutor::InitFullGraph(
    nnvm::Symbol symbol,
    const std::vector<OpReqType>& grad_req_type,
    const std::vector<NDArray>& arg_grad_store) {
  using nnvm::NodePtr;
  using nnvm::NodeEntry;
  // initial information
  num_forward_outputs_ = symbol.outputs.size();
  num_forward_inputs_ = symbol.ListInputs(nnvm::Symbol::kAll).size();

  nnvm::Graph g;
  g.outputs = symbol.outputs;

  bool need_grad = false;
  for (OpReqType req : grad_req_type) {
    if (req != kNullOp) need_grad = true;
  }
  if (!need_grad) return g;
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    NodeEntry ngrad{nnvm::Node::Create(), 0, 0};
    ngrad.node->attrs.name = std::string("ngrad") + std::to_string(i);
    head_grad_entry_.emplace_back(AttrHint(ngrad, g.outputs[i]));
    head_grad_map_[ngrad.node.get()] = i;
  }
  std::vector<NodePtr> args = symbol.ListInputs(nnvm::Symbol::kReadOnlyArgs);
  std::vector<NodeEntry> xs;
  for (size_t i = 0; i < grad_req_type.size(); ++i) {
    if (grad_req_type[i] != kNullOp) {
      grad_store_.emplace_back(
          std::make_pair(grad_req_type[i], arg_grad_store[i]));
      xs.emplace_back(NodeEntry{args[i], 0, 0});
    }
  }

  int do_mirror = dmlc::GetEnv("MXNET_BACKWARD_DO_MIRROR", 0);
  auto need_mirror = [do_mirror](const nnvm::Node& node) -> int {
    if (node.is_variable()) return 0;
    const std::string& type = node.attrs.op->name;
    if (type == "Dropout") return false;
    if (get_node_attr(node, "__force_mirroring__", false)) return true;
    if (do_mirror == 0) return false;
    if (type == "Convolution") return false;
    if (type == "FullyConnected") return false;
    if (type == "Concat") return false;
    if (type == "SoftmaxOutput") return false;
    if (type == "CuDNNBatchNorm") return false;
    return true;
  };
  // take gradient
  nnvm::Graph g_grad = nnvm::pass::Gradient(
      g, symbol.outputs, xs, head_grad_entry_,
      AggregateGradient, need_mirror);
  return g_grad;
}

// pass to assign context to the graph
Graph AssignContext(Graph g,
                    const Context& default_ctx,
                    const std::map<std::string, Context>& ctx_map,
                    const std::vector<NDArray>& in_args,
                    const std::vector<std::pair<OpReqType, NDArray> >& grad_store,
                    const std::vector<NDArray>& aux_states,
                    size_t num_forward_inputs,
                    size_t num_forward_outputs,
                    const std::string& copy_op_name) {
  const auto& idx = g.indexed_graph();
  const auto& mutable_nodes = idx.mutable_input_nodes();
  // default use default context.
  if (ctx_map.size() == 0) {
    g.attrs["context"] = std::make_shared<nnvm::any>(
        ContextVector(idx.num_nodes(), default_ctx));
    for (const auto& x : in_args) {
      CHECK(x.ctx() == default_ctx)
        << "Input array is in " << x.ctx() << " while binding with ctx=" << default_ctx
        << ". All arguments must be in global context (" << default_ctx
        << ") unless group2ctx is specified for cross-device graph.";
    }
    for (const auto& x : grad_store) {
      CHECK(x.second.ctx() == default_ctx)
        << "Gradient array is in " << x.second.ctx() << " while binding with ctx="
        << default_ctx << ". All gradients must be in global context (" << default_ctx
        << ") unless group2ctx is specified for cross-device graph.";
    }
    return g;
  }
  // otherwise, use context assignment.
  std::map<Context, int> ctx2id;
  std::vector<Context> ctx_list;
  nnvm::DeviceVector device(idx.num_nodes(), -1);
  nnvm::DeviceAssignMap device_map;

  for (auto &kv : ctx_map) {
    if (ctx2id.count(kv.second) == 0) {
      ctx2id[kv.second] = static_cast<int>(ctx_list.size());
      ctx_list.push_back(kv.second);
    }
    device_map[kv.first] = ctx2id.at(kv.second);
  }

  // Enable P2P connection.
  EnableP2P(ctx_list);

#ifdef SPLIT_GRADIENT_TEST
  g = nnvm::pass::SplitGradientTest(g, "__ctx_group__", num_forward_outputs);
#endif
  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < num_forward_inputs; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    Context ctx;
    if (mutable_nodes.count(nid)) {
      CHECK_LT(aux_top, aux_states.size());
      ctx = aux_states[aux_top].ctx();
      ++aux_top;
    } else {
      CHECK_LT(arg_top, in_args.size());
      ctx = in_args[arg_top].ctx();
      ++arg_top;
    }
    if (ctx2id.count(ctx) == 0) {
      ctx2id[ctx] = static_cast<int>(ctx_list.size());
      ctx_list.push_back(ctx);
    }
    device[nid] = ctx2id.at(ctx);
  }
  for (size_t i = num_forward_outputs; i < g.outputs.size(); ++i) {
    const uint32_t nid = idx.outputs()[i].node_id;
    Context ctx = grad_store[i - num_forward_outputs].second.ctx();
    if (ctx2id.count(ctx) == 0) {
      ctx2id[ctx] = static_cast<int>(ctx_list.size());
      ctx_list.push_back(ctx);
    }
    int devid = ctx2id.at(ctx);
    if (device[nid] != -1) {
      CHECK_EQ(device[nid], devid) << "device of same output not equal to each other";
    } else {
      device[nid] = devid;
    }
  }
  g.attrs["device"] = std::make_shared<dmlc::any>(std::move(device));
  g = nnvm::pass::PlaceDevice(g, "__ctx_group__", device_map, copy_op_name);
  LOG(INFO) << "Place device pass finished.";
  const auto& assigned_device = g.GetAttr<nnvm::DeviceVector>("device");

  ContextVector vcontext;
  for (size_t i = 0; i < assigned_device.size(); ++i) {
    if (assigned_device[i] == -1) {
      vcontext.push_back(default_ctx);
    } else {
      vcontext.push_back(ctx_list[assigned_device[i]]);
    }
  }
  g.attrs["context"] = std::make_shared<nnvm::any>(std::move(vcontext));
  return g;
}

void GraphExecutor::Init(nnvm::Symbol symbol,
                         const Context& default_ctx,
                         const std::map<std::string, Context>& ctx_map,
                         const std::vector<NDArray>& in_args,
                         const std::vector<NDArray>& arg_grad_store,
                         const std::vector<OpReqType>& grad_req_type,
                         const std::vector<NDArray>& aux_states,
                         Executor* shared_exec) {
  nnvm::Graph g = InitGraph(symbol, default_ctx,
                            ctx_map, in_args, arg_grad_store,
                            grad_req_type, aux_states);
  g = AttachOpExecs(g);
  g = AttachOpResources(g);
  graph_ = std::move(g);
  if (shared_exec != nullptr) {
    this->InitDataEntryMemory(dynamic_cast<GraphExecutor*>(shared_exec)->data_pool_);
  } else {
    this->InitDataEntryMemory({});
  }
  {
    // initialize output arrays
    auto& idx = graph_.indexed_graph();
    for (size_t i = 0; i < num_forward_outputs_; ++i) {
      auto& e = idx.outputs()[i];
      output_arrays_.push_back(data_entry_[idx.entry_id(e)]);
    }
    // initialize head gradient array
    head_grad_array_.resize(symbol.outputs.size());
    for (size_t i = num_forward_inputs_; i < idx.input_nodes().size(); ++i) {
      uint32_t nid = idx.input_nodes().at(i);
      auto it = head_grad_map_.find(idx[nid].source);
      if (it != head_grad_map_.end()) {
        uint32_t oid = it->second;
        head_grad_array_[oid] = data_entry_[idx.entry_id(nid, 0)];
      }
    }
  }
  this->InitCachedOps();
}

Graph GraphExecutor::InferShapeType(
    Graph g,
    const std::vector<NDArray>& in_args,
    const std::vector<NDArray>& aux_states) {
  const auto& idx = g.indexed_graph();
  // Setup argument shape and type.
  const std::unordered_set<uint32_t>& mutable_nodes = idx.mutable_input_nodes();
  nnvm::ShapeVector arg_shapes;
  nnvm::DTypeVector arg_types;
  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    if (mutable_nodes.count(nid)) {
      CHECK_LT(aux_top, aux_states.size());
      arg_shapes.push_back(aux_states[aux_top].shape());
      arg_types.push_back(aux_states[aux_top].dtype());
      ++aux_top;
    } else {
      CHECK_LT(arg_top, in_args.size());
      arg_shapes.push_back(in_args[arg_top].shape());
      arg_types.push_back(in_args[arg_top].dtype());
      ++arg_top;
    }
  }
  arg_shapes.resize(idx.input_nodes().size(), TShape());
  arg_types.resize(idx.input_nodes().size(), 0);
  g = nnvm::pass::InferShape(g, arg_shapes, "__shape__");
  g = nnvm::pass::InferType(g, arg_types);
  return g;
}


Graph GraphExecutor::InitGraph(nnvm::Symbol symbol,
                               const Context& default_ctx,
                               const std::map<std::string, Context>& ctx_map,
                               const std::vector<NDArray>& in_args,
                               const std::vector<NDArray>& arg_grad_store,
                               const std::vector<OpReqType>& grad_req_type,
                               const std::vector<NDArray>& aux_states) {
  // setup gradient
  nnvm::Graph g = InitFullGraph(symbol, grad_req_type, arg_grad_store);
  g = InferShapeType(g, in_args, aux_states);
  std::string json = nnvm::pass::SaveJSON(g);
  std::ofstream json_file;
  json_file.open("graph.json");
  json_file << json;
  json_file.close();
  // Call partition pass here.
  bool need_grad = false;
  for (OpReqType req : grad_req_type) {
    if (req != kNullOp) need_grad = true;
  }
  const int num_devices = ctx_map.size();
  // TODO(minjie): Here has an implicit assumption.
  // The ctx_map is of form {"group:%d" % id : context object}
  // The default group name is "group:default".
  std::map<std::string, Context> ctx_map_with_default = ctx_map;
  ctx_map_with_default["group:default"] = default_ctx;
  const int tofu_enabled = dmlc::GetEnv("TOFU_ENABLED", 0);
  if (num_devices > 1 && tofu_enabled && need_grad) {
    LOG(INFO) << "Num devices: " << num_devices;
    g.attrs["num_devices"] = std::make_shared<nnvm::any>(num_devices);
    g.attrs["default_group"] = std::make_shared<nnvm::any>(std::string("group:default"));
    g.attrs["user_tiling_json"] = std::make_shared<nnvm::any>("");
    g.attrs["copy_op_name"] = std::make_shared<nnvm::any>(copy_op_name_);
    const std::string& tiling_type = dmlc::GetEnv("TOFU_TILING_TYPE",
                                                  std::string("kcuts"));
    if (tiling_type == "usertiling") {
      const std::string& user_tiling_json =
        dmlc::GetEnv("TOFU_TILING_JSON", std::string("user_tiling.json"));
      std::ifstream json_file;
      json_file.open(user_tiling_json);
      std::stringstream json;
      json << json_file.rdbuf();
      json_file.close();
      g.attrs["user_tiling_json"] = std::make_shared<nnvm::any>(json.str());
    }
    g = nnvm::ApplyPass(g, "PartitionPass");
  }
  // Assign contexts to the graph.
  g = AssignContext(g, default_ctx, ctx_map_with_default,
                    in_args,
                    grad_store_,
                    aux_states,
                    num_forward_inputs_,
                    num_forward_outputs_,
                    copy_op_name_);
  const auto& idx = g.indexed_graph();
  const std::unordered_set<uint32_t>& mutable_nodes = idx.mutable_input_nodes();
  // Setup data entry and point input/output to proper arguments.
  data_entry_.resize(idx.num_node_entries());
  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    if (mutable_nodes.count(nid)) {
      CHECK_LT(aux_top, aux_states.size());
      data_entry_[idx.entry_id(nid, 0)] = aux_states[aux_top];
      ++aux_top;
    } else {
      CHECK_LT(arg_top, in_args.size());
      data_entry_[idx.entry_id(nid, 0)] = in_args[arg_top];
      ++arg_top;
    }
  }
  for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) {
    data_entry_[idx.entry_id(idx.outputs()[j])]
        = grad_store_[j - num_forward_outputs_].second;
  }

  // We wait until the last momoent, right before allocating memory, to split
  // the graph.
  g = SplitDistributedGraph(g, default_ctx);

  {
    // Can't reuse the previous idx because of SplitDistributedGraph.
    const auto& idx = g.indexed_graph();
    // get number of nodes used in forward pass
    num_forward_nodes_ = 0;
    for (size_t i = 0; i < num_forward_outputs_; ++i) {
      num_forward_nodes_ = std::max(
          num_forward_nodes_, static_cast<size_t>(idx.outputs()[i].node_id + 1));
    }

    // memory allocator
    const int kBadStorageID = -1;
    const int kExternalStorageID = -2;
    nnvm::StorageVector arg_storage_id(idx.num_node_entries(), kBadStorageID);
    for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) {
      arg_storage_id[idx.entry_id(idx.outputs()[j])] = kExternalStorageID;
    }
    g.attrs["storage"] = std::make_shared<dmlc::any>(std::move(arg_storage_id));
    g = nnvm::ApplyPass(g, "PlanMemory");
  }
  g = DetectInplaceAddTo(g);
  return g;
}

// initialize the memory of each entries
void GraphExecutor::InitDataEntryMemory(const std::vector<NDArray>& shared_pool) {
  using nnvm::DTypeVector;
  using nnvm::ShapeVector;
  using nnvm::StorageVector;
  // get the graph
  const auto& idx = graph_.indexed_graph();
  // get the storage
  const auto& vdtype = graph_.GetAttr<DTypeVector>("dtype");
  const auto& vshape = graph_.GetAttr<ShapeVector>("shape");
  const auto& vstorage = graph_.GetAttr<StorageVector>("storage_id");
  const auto& vctx = graph_.GetAttr<ContextVector>("context");
  CHECK_EQ(idx.num_node_entries(), vshape.size());
  CHECK_EQ(idx.num_node_entries(), vdtype.size());
  CHECK_EQ(idx.num_node_entries(), vstorage.size());
  CHECK_EQ(data_entry_.size(), vshape.size());
  std::vector<Context> data_context(idx.num_node_entries());
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    for (uint32_t i = 0; i < idx[nid].source->num_outputs(); ++i) {
      //LOG(INFO) << "Node " << idx[nid].source->attrs.name << " output#" << i << ": "
        //<< " entryid=" << idx.entry_id(nid, i) << " shape=" << vshape[idx.entry_id(nid, i)];
      data_context[idx.entry_id(nid, i)] = vctx[nid];
    }
  }

  // information about the pool
  using PoolEntry = std::pair<Context, size_t>;
  std::vector<PoolEntry> pool_info;

  for (size_t i = num_forward_inputs_; i < idx.input_nodes().size(); ++i) {
    uint32_t nid = idx.input_nodes().at(i);
    const auto& it = head_grad_map_.find(idx[nid].source);
    if (it == head_grad_map_.end()) {
      uint32_t grad_eid = idx.entry_id(nid, 0);
      data_entry_[grad_eid] = NDArray(vshape[grad_eid], data_context[grad_eid],
                                      false, vdtype[grad_eid]);
    } else {
      //uint32_t oid = head_grad_map_.at(idx[nid].source);
      uint32_t oid = it->second;
      uint32_t eid = idx.entry_id(idx.outputs()[oid]);
      CHECK_NE(vshape[eid].ndim(), 0);
      CHECK_NE(vdtype[eid], -1);
      data_entry_[idx.entry_id(nid, 0)] =
          NDArray(vshape[eid], data_context[eid], false, vdtype[eid]);
    }
  }
  // get maximum bytes in each pool
  for (size_t i = 0; i < vshape.size(); ++i) {
    if (!data_entry_[i].is_none()) continue;
    size_t bytes = vshape[i].Size() * mshadow::mshadow_sizeof(vdtype[i]);
    int storage_id = vstorage[i];
    if (storage_id < 0) continue;
    size_t sid = static_cast<size_t>(storage_id);
    if (sid >= pool_info.size()) {
      pool_info.resize(sid + 1, PoolEntry{Context::CPU(), size_t(0)});
    }
    PoolEntry& info = pool_info[sid];
    if (info.second == 0) {
      info = PoolEntry{data_context[i], bytes};
    } else {
      info.second = std::max(info.second, bytes);
    }
  }
  // construct the re-use pool, if needed
  std::multimap<size_t, NDArray> free_pool;
  for (const NDArray& nd : shared_pool) {
    size_t bytes = nd.shape().Size() * mshadow::mshadow_sizeof(nd.dtype());
    free_pool.insert(std::make_pair(bytes, nd));
  }
  // remake the data pool
  data_pool_.clear();
  for (size_t i = 0; i < pool_info.size(); ++i) {
    const Context& ctx = pool_info[i].first;
    size_t bytes = pool_info[i].second;
    bool allocated = false;
    for (auto it = free_pool.lower_bound(bytes); it != free_pool.end(); ++it) {
      if (it->second.ctx() == ctx && it->first >= bytes) {
        data_pool_.push_back(it->second);
        free_pool.erase(it);
        allocated = true;
        break;
      }
    }
    if (!allocated) {
      size_t nword = (bytes + 3) / 4;
      CHECK_LE(nword, std::numeric_limits<index_t>::max());
      // allocate float arrays
      TShape shape{index_t(nword)};
      data_pool_.emplace_back(NDArray(shape, ctx));
    }
  }
  CHECK_EQ(data_pool_.size(), pool_info.size());
  // assign the data entries
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    // avoid pre-allocated arrays
    if (!data_entry_[i].is_none()) continue;
    // assign allocated array by storage id
    int storage_id = vstorage[i];
    CHECK_GE(storage_id, 0)
        << "Do not support runtime shape op yet. Node's name is"
        << " " << idx[i].source->attrs.name << ". Entryid="
        << i << " DataEntrySize=" << data_entry_.size()
        << " Shape=" << vshape[i];
    const NDArray& src = data_pool_.at(storage_id);
    data_entry_[i] = src.AsArray(vshape[i], vdtype[i]);
  }

  LOG(INFO) << "[WARNING!!!] Init fake data for all entries (for more stable benchmark).";
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    SetValueOp(0.0, &(data_entry_[i]));
  }

}

std::vector<int> GraphExecutor::CalcPriority() {
  const auto& idx = graph_.indexed_graph();
  std::vector<int> ret(idx.num_nodes(), 0);
  DFSVisit(graph_.outputs, [&ret, &idx] (const nnvm::NodePtr& n) {
      const uint32_t nid = idx.node_id(n.get());
      int priority = 0;
      for (const auto& in_ent : n->inputs) {
        const uint32_t pnid = idx.node_id(in_ent.node.get());
        priority = std::max(priority, ret[pnid]);
      }
      ret[nid] = priority + 1;
      //LOG(INFO) << "Visit node #" << idx.node_id(n.get()) << " Priority=" << ret[nid];
    });
  return ret;
}

void GraphExecutor::InitCachedOps() {
  // get the graph
  const auto& idx = graph_.indexed_graph();
  const auto& vstorage_inplace =
      graph_.GetAttr<std::vector<int> >("storage_inplace_index");
  const auto& op_execs =
      graph_.GetAttr<OpExecVector>("op_execs");
  const auto& vctx = graph_.GetAttr<ContextVector>("context");
  const auto& addto_entry = graph_.GetAttr<std::vector<int> >("addto_entry");
  const auto& skip_plus_node = graph_.GetAttr<std::vector<int> >("skip_plus_node");

  // Get priorities.
  const auto& node_priorities = CalcPriority();

  op_nodes_.resize(idx.num_nodes());
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
#if MXNET_USE_PROFILER
    op_nodes_[nid].opr_name = inode.source->op()->name.c_str();
#else
    op_nodes_[nid].opr_name = nullptr;
#endif
    if (skip_plus_node.at(nid)) {
      op_nodes_[nid].skip_exec_node = true; continue;
    }
    if (inode.source->op() == nnvm::Op::Get("_TofuFakeVar")) {
      op_nodes_[nid].skip_exec_node = true; continue;
    }

    op_nodes_[nid].exec = op_execs[nid];
    op_nodes_[nid].ctx = vctx[nid];
    auto& exec = op_nodes_[nid].exec;
    CHECK_EQ(exec->in_array.size(), 0);
    CHECK_EQ(exec->out_array.size(), 0);
    for (const auto& e : inode.inputs) {
      exec->in_array.push_back(data_entry_[idx.entry_id(e)]);
    }
    // detect inplace requirement
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      exec->out_array.push_back(data_entry_[eid]);
      if (addto_entry.at(eid) != 0) {
        exec->req.push_back(kAddTo);
      } else if (vstorage_inplace[eid] >= 0) {
        exec->req.push_back(kWriteInplace);
      } else if (vstorage_inplace[eid] == -2) {
        // -2 indicate that the entry is never referenced.
        exec->req.push_back(kNullOp);
      } else {
        exec->req.push_back(kWriteTo);
      }
    }
  }
  // Note that this modifies the requirment of kWriteInplace
  for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) {
    auto& e = idx.outputs()[j];
    op_nodes_[e.node_id].exec->req[e.index] =
        grad_store_[j - num_forward_outputs_].first;
  }
  std::vector<Engine::VarHandle> finish_vars(idx.num_nodes());
  DFGEProfiler::Get().Begin();
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    finish_vars[nid] = Engine::Get()->NewVariable();
    if (inode.source->is_variable()) continue;
    if (op_nodes_[nid].skip_exec_node) continue;
    auto& exec = op_nodes_[nid].exec;

    std::vector<uint32_t> inplace_inputs;
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      // must check exec->req because, vstorage_inplace is only a hint.
      if (vstorage_inplace[eid] >= 0 && exec->req.at(index) == kWriteInplace) {
        inplace_inputs.push_back(vstorage_inplace[eid]);
      }
    }
    std::sort(inplace_inputs.begin(), inplace_inputs.end());

    bool is_async = op_nodes_[nid].exec->exec_type() == Operator::kAsync;
    bool is_gpu = op_nodes_[nid].ctx.dev_mask() == gpu::kDevMask;
    // the variable
    std::vector<Engine::VarHandle> use_vars, mutate_vars, all_vars;
    for (size_t i = 0; i < exec->in_array.size(); ++i) {
      if (!std::binary_search(inplace_inputs.begin(), inplace_inputs.end(), i)) {
        auto& nd = exec->in_array[i];
        all_vars.push_back(nd.var());
        use_vars.push_back(nd.var());
      }
    }
    // Handle control dependencies.
    for (auto & depend_node : inode.source->control_deps) {
      const uint32_t depend_nid = idx.node_id(depend_node.get());
      CHECK_LT(depend_nid, nid);
      use_vars.push_back(finish_vars[depend_nid]);
    }
    for (auto& r : exec->op_ctx.requested) {
      all_vars.push_back(r.var);
      mutate_vars.push_back(r.var);
    }
    for (auto& nd : exec->out_array) {
      all_vars.push_back(nd.var());
      mutate_vars.push_back(nd.var());
    }
    mutate_vars.push_back(finish_vars[nid]);
    auto dedup = [] (std::vector<Engine::VarHandle>& vars) {  // NOLINT(*)
      std::sort(vars.begin(), vars.end());
      vars.resize(std::unique(vars.begin(), vars.end()) - vars.begin());
    };
    dedup(use_vars);
    for (auto v : use_vars) {
      if (std::binary_search(mutate_vars.begin(), mutate_vars.end(), v)) {
        LOG(FATAL) << "var duplication happens for op " << inode.source->attrs.name;
      }
    }
    dedup(mutate_vars);
    dedup(all_vars);
    const int oversharding = dmlc::GetEnv("TOFU_OVERSHARDING", 0);
    if (oversharding) {
      Engine::Get()->PushSync([exec](RunContext rctx) {
          exec->Setup();
      }, Context::CPU(), {}, all_vars, FnProperty::kNormal, node_priorities[nid],
      PROFILER_MESSAGE("SetupExec"));
    } else {
      Engine::Get()->PushSync([exec](RunContext rctx) {
          exec->Setup();
      }, Context::CPU(), {}, all_vars, FnProperty::kNormal, 0,
      PROFILER_MESSAGE("SetupExec"));
    }
    auto& name = idx[nid].source->attrs.name;
    auto exec_fun = [exec, is_async, is_gpu, name, nid, this] (
        RunContext ctx, Engine::CallbackOnComplete on_complete) {
      if (is_async) {
        //exec->op_ctx.async_on_complete = on_complete;
        struct Capture {
          Engine::CallbackOnComplete on_complete;
          uint32_t nid;
          bool is_gpu;
        };
        Capture* capture = new Capture{on_complete, nid, is_gpu};
        exec->op_ctx.async_on_complete =
          Engine::Get()->CreateCallback(
              [](Engine* engine, void *param) {
                  Capture* cpt = static_cast<Capture*>(param);
                  cpt->on_complete();
                  DFGEProfiler::Get().Write(cpt->nid, cpt->is_gpu, true);
                  delete cpt;
              }, static_cast<void*>(capture));
      }
      op::P2PNetDebugger::Get().PrintTime("Begin executing %s", name.c_str());
      DFGEProfiler::Get().Write(nid, is_gpu, is_async);
      //timeval st, ed;
      //gettimeofday(&st, NULL);
      exec->Run(ctx);
      op::P2PNetDebugger::Get().PrintTime("Finish executing %s", name.c_str());
      // call on complete only if it is async op
      if (!is_async) {
        if (is_gpu) {
        #if MXNET_USE_CUDA
          // Wait GPU kernel to finish.
          ctx.get_stream<gpu>()->Wait();
        #else
          LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
        #endif
        }
        //gettimeofday(&ed, NULL);
        //{
          //std::lock_guard<std::mutex> guard(time_mutex_);
          //LOG(INFO) << "Node: " << name << " time: " << ((ed.tv_sec - st.tv_sec) * 1000.0 + (ed.tv_usec - st.tv_usec) / 1000.0);
        //}
        on_complete();
        DFGEProfiler::Get().Write(nid, is_gpu, is_async);
      }
    };
    // setup the vars
    FnProperty prop;
    const auto op_name = std::string(idx[nid].source->op()->name);
    if (op_name == "P2PNetInit" || op_name == "P2PNetRecv" ||
        op_name == "P2PNetSend") {
      prop = FnProperty::kCPUPrioritized;
    } else {
      prop = FnProperty::kNormal;
    }
    op_nodes_[nid].cached_opr = Engine::Get()->NewOperator(
        exec_fun, use_vars, mutate_vars, prop,
        PROFILER_MESSAGE(op_nodes_[nid].opr_name));
  }
}

void GraphExecutor::RunOps(bool is_train, size_t topo_start, size_t topo_end) {
  static const auto& flist_outputs =
      nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  const auto& idx = graph_.indexed_graph();
  for (size_t nid = topo_start; nid < topo_end; ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    OpNode& opnode = op_nodes_[nid];
    if (op_nodes_[nid].skip_exec_node) continue;
    opnode.exec->op_ctx.is_train = is_train;
    if (inode.source->op() == nnvm::Op::Get("_TofuFusedConvert")) {
      op::TofuCopyFromTo(inode.source->attrs,
                         opnode.exec->in_array,
                         &(opnode.exec->out_array[0]));
    } else if (inode.source->op() == nnvm::Op::Get("_TofuFusedConvertNoComm")) {
      op::TofuCopyFromToNoComm(inode.source->attrs,
                               opnode.exec->in_array,
                               &(opnode.exec->out_array[0]));
    } else if (opnode.exec->exec_type() == Operator::kCrossDeviceCopy) {
      CHECK_EQ(inode.inputs.size(), 1);
      CHECK_EQ(opnode.exec->in_array.size(), 1);
      CHECK_EQ(opnode.exec->out_array.size(), 1);
      CopyFromTo(opnode.exec->in_array[0], &(opnode.exec->out_array[0]));
    } else if (opnode.cached_opr != nullptr) {
#if MXNET_USE_PROFILER
      bool profiling = engine::Profiler::Get()->GetState() == engine::Profiler::kRunning;
#else
      bool profiling = false;
#endif
      Engine::Get()->Push(opnode.cached_opr, opnode.ctx, 0, profiling);
    } else {
      LOG(FATAL) << "Not accessed";
    }

    if (monitor_callback_) {
      std::vector<std::string> output_names;
      const auto& node = idx[nid].source;
      if (flist_outputs.count(node->op())) {
        output_names = flist_outputs[node->op()](node->attrs);
      } else {
        for (size_t i = 0; i < node->num_outputs(); ++i) {
          output_names.emplace_back(std::to_string(i));
        }
      }
      for (index_t i = 0; i < opnode.exec->out_array.size(); ++i) {
        NDArray *cpy = new NDArray(opnode.exec->out_array[i]);
        std::string name = inode.source->attrs.name + "_" + output_names[i];
        this->monitor_callback_(name.c_str(), reinterpret_cast<void*>(cpy));
      }
    }
  }
}
  
GraphExecutor::GraphExecutor() {
  const int use_cached_copy = dmlc::GetEnv("TOFU_USE_CACHED_COPY", 0);
  if (use_cached_copy) {
    copy_op_name_ = "_TofuCachedCopy";
  }
}

}  // namespace exec

Executor *Executor::Bind(nnvm::Symbol symbol,
                         const Context& default_ctx,
                         const std::map<std::string, Context>& group2ctx,
                         const std::vector<NDArray> &in_args,
                         const std::vector<NDArray> &arg_grad_store,
                         const std::vector<OpReqType> &grad_req_type,
                         const std::vector<NDArray> &aux_states,
                         Executor* shared_exec) {
  auto exec = new exec::GraphExecutor();
  exec->Init(symbol, default_ctx, group2ctx,
             in_args, arg_grad_store, grad_req_type, aux_states,
             reinterpret_cast<Executor*>(shared_exec));
  return exec;
}
}  // namespace mxnet
