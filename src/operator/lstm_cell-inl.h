/*!
 * Copyright (c) 2015 by Contributors
 * \file lstm_cell-inl.h
 * \brief Activation operator
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_LSTM_CELL_INL_H_
#define MXNET_OPERATOR_LSTM_CELL_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
namespace lstm {
enum LSTMCellInputs {kPrevC=0, kInGate, kTransGate, kForgetGate, kOutGate, kNumInputs};
enum LSTMCellOutputs {kC=0, kH, kNumOutputs};
}  // lstm

/**
 * \brief This is the implementation of lstm operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename DType>
class LSTMCellOp : public Operator {
 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), lstm::kNumInputs);
    CHECK_EQ(out_data.size(), lstm::kNumOutputs);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> prev_c = in_data[lstm::kPrevC].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> in_gate = in_data[lstm::kInGate].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> trans_gate = in_data[lstm::kTransGate].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> forget_gate = in_data[lstm::kForgetGate].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out_gate = in_data[lstm::kOutGate].FlatTo2D<xpu, DType>(s);

    Tensor<xpu, 2, DType> c = out_data[lstm::kC].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> h = out_data[lstm::kH].FlatTo2D<xpu, DType>(s);

    auto i = F<mshadow_op::sigmoid>(in_gate);
    auto t = F<mshadow_op::tanh>(trans_gate);
    auto f = F<mshadow_op::sigmoid>(forget_gate);
    auto o = F<mshadow_op::sigmoid>(out_gate);

    Assign(c, req[lstm::kC], f * prev_c + i * t);
    Assign(h, req[lstm::kH], o * F<mshadow_op::tanh>(c));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    return;
    LOG(INFO) << "LSTM Cell backward";
    CHECK_EQ(out_grad.size(), lstm::kNumOutputs);
    CHECK_EQ(in_data.size(), lstm::kNumInputs);
    CHECK_EQ(in_grad.size(), lstm::kNumInputs);
    CHECK_EQ(req.size(), lstm::kNumInputs);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> dC = out_grad[lstm::kC].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> dH = out_grad[lstm::kH].FlatTo2D<xpu, DType>(s);

    Tensor<xpu, 2, DType> prev_c = in_data[lstm::kPrevC].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> in_gate = in_data[lstm::kInGate].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> trans_gate = in_data[lstm::kTransGate].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> forget_gate = in_data[lstm::kForgetGate].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out_gate = in_data[lstm::kOutGate].FlatTo2D<xpu, DType>(s);

    Tensor<xpu, 2, DType> d_prev_c = in_grad[lstm::kPrevC].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> d_in_gate = in_grad[lstm::kInGate].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> d_trans_gate = in_grad[lstm::kTransGate].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> d_forget_gate = in_grad[lstm::kForgetGate].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> d_out_gate = in_grad[lstm::kOutGate].FlatTo2D<xpu, DType>(s);
    LOG(INFO) << "LSTM Cell backward2";

    auto i = F<mshadow_op::sigmoid>(in_gate);
    auto t = F<mshadow_op::tanh>(trans_gate);
    auto f = F<mshadow_op::sigmoid>(forget_gate);
    auto o = F<mshadow_op::sigmoid>(out_gate);
    auto ct = f * prev_c + i * t;
    auto x2 = F<mshadow_op::tanh>(ct);
    LOG(INFO) << "LSTM Cell backward3";

    auto dx1 = dH * x2;
    auto dx2 = o * dH;
    auto dx3 = dx2 * F<mshadow_op::tanh_grad>(x2);
    auto dx4 = dC + dx3;
    auto df = dx4 * prev_c;
    LOG(INFO) << "LSTM Cell backward3.5";
    Assign(d_prev_c, req[lstm::kPrevC], f * dx4);
    LOG(INFO) << "LSTM Cell backward4";
    auto di = dx4 * t;
    auto dt = i * dx4;
    Assign(d_in_gate, req[lstm::kInGate], F<mshadow_op::sigmoid_grad>(i) * di);
    Assign(d_trans_gate, req[lstm::kTransGate], F<mshadow_op::tanh_grad>(t) * dt);
    Assign(d_forget_gate, req[lstm::kForgetGate], F<mshadow_op::sigmoid_grad>(f) * df);
    Assign(d_out_gate, req[lstm::kOutGate], F<mshadow_op::sigmoid_grad>(o) * dx1);
    LOG(INFO) << "LSTM Cell backward finished";
  }
};  // class LSTMCellOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(int dtype);

template<typename xpu>
Operator* CreateBackwardOp(int dtype);

#if DMLC_USE_CXX11
class LSTMCellProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"prev_c", "in_gate", "trans_gate", "forget_gate", "out_gate"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"next_c", "next_h"};
  }

  int NumOutputs() const override {
    return lstm::kNumOutputs;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
  }

  std::map<std::string, std::string> GetParams() const override {
    return std::map<std::string, std::string>();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), lstm::kNumInputs);
    //CHECK_EQ(out_shape->size(), lstm::kNumOutputs);
    TShape shape;
    for (const auto& shp : (*in_shape)) {
      if (shp.ndim() != 0) {
        shape = shp;
        break;
      }
    }
    for (const auto& shp : (*out_shape)) {
      if (shp.ndim() != 0) {
        shape = shp;
        break;
      }
    }
    if (shape.ndim() == 0) return false;
    in_shape->clear();
    in_shape->resize(lstm::kNumInputs, shape);
    out_shape->clear();
    out_shape->resize(lstm::kNumOutputs, shape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), lstm::kNumInputs);
    //CHECK_EQ(out_type->size(), lstm::kNumOutputs);
    int type;
    for (const auto& ty : (*in_type)) {
      if (ty != -1) {
        type = ty;
        break;
      }
    }
    for (const auto& ty : (*out_type)) {
      if (ty != -1) {
        type = ty;
        break;
      }
    }
    if (type == -1) return false;
    in_type->clear();
    in_type->resize(lstm::kNumInputs, type);
    out_type->clear();
    out_type->resize(lstm::kNumOutputs, type);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new LSTMCellProp();
    return ptr;
  }

  std::string TypeString() const override {
    return "LSTMCell";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[lstm::kC], out_grad[lstm::kH],
            in_data[lstm::kPrevC], in_data[lstm::kInGate],
            in_data[lstm::kTransGate], in_data[lstm::kForgetGate],
            in_data[lstm::kOutGate]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[lstm::kC], in_grad[lstm::kPrevC]},
            {out_grad[lstm::kH], in_grad[lstm::kPrevC]},
            {out_grad[lstm::kC], in_grad[lstm::kInGate]},
            {out_grad[lstm::kH], in_grad[lstm::kInGate]},
            {out_grad[lstm::kC], in_grad[lstm::kTransGate]},
            {out_grad[lstm::kH], in_grad[lstm::kTransGate]},
            {out_grad[lstm::kC], in_grad[lstm::kForgetGate]},
            {out_grad[lstm::kH], in_grad[lstm::kForgetGate]},
            {out_grad[lstm::kC], in_grad[lstm::kOutGate]},
            {out_grad[lstm::kH], in_grad[lstm::kOutGate]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[lstm::kPrevC], out_data[lstm::kC]},
            {in_data[lstm::kPrevC], out_data[lstm::kH]},
            {in_data[lstm::kInGate], out_data[lstm::kC]},
            {in_data[lstm::kInGate], out_data[lstm::kH]},
            {in_data[lstm::kTransGate], out_data[lstm::kC]},
            {in_data[lstm::kTransGate], out_data[lstm::kH]},
            {in_data[lstm::kForgetGate], out_data[lstm::kC]},
            {in_data[lstm::kForgetGate], out_data[lstm::kH]},
            {in_data[lstm::kOutGate], out_data[lstm::kC]},
            {in_data[lstm::kOutGate], out_data[lstm::kH]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

  Operator* CreateBackwardOperatorEx(
      const Context& ctx,
      const std::vector<TShape>& in_shape,
      const std::vector<int>& in_type,
      const std::vector<TShape>& out_shape,
      const std::vector<int>& out_type) const override;

};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_LSTM_CELL_INL_H_
