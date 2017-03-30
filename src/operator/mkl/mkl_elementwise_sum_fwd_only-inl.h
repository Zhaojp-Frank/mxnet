#ifndef MXNET_OPERATOR_MKL_MKL_ELEMENTWISE_FWD_ONLY_INL_H_
#define MXNET_OPERATOR_MKL_MKL_ELEMENTWISE_FWD_ONLY_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "./mkl_util-inl.h"

namespace mxnet {
namespace op {

namespace elemsum {
enum ElementWiseSumOnlyFwdOutputs {kOut};
}

struct ElementWiseSumOnlyFwdParam : public dmlc::Parameter<ElementWiseSumOnlyFwdParam> {
  // use int for enumeration
  int num_args;
  DMLC_DECLARE_PARAMETER(ElementWiseSumOnlyFwdParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs to be summed.");
  }
};

#if MKL_EXPERIMENTAL == 1
template<typename DType>
inline std::shared_ptr<MKLData<DType>> GetMemDescr(const TBlob& data) {
  if (mkl_prv_data<DType>(data) != nullptr) {
    std::shared_ptr<MKLMemHolder> data_mem = data.Mkl_mem_;
    std::shared_ptr<PrvMemDescr> prv_descriptor = data_mem->get_prv_descriptor();
    CHECK_EQ(prv_descriptor->get_descr_type(), PrvMemDescr::PRV_DESCR_MKL2017);
    return CHECK_NOTNULL(std::static_pointer_cast<MKLData<DType>>(prv_descriptor));
  }
  return nullptr;
}
#endif

template<typename xpu, typename DType>
class MKLElementWiseSumOnlyFwdOp : public Operator {
 public:
  explicit MKLElementWiseSumOnlyFwdOp(
      const ElementWiseSumOnlyFwdParam& param,
      const std::vector<TShape>& in_shapes,
      const std::vector<TShape>& out_shapes)
    : num_args_(param.num_args) {
    fwd_top_data_ = MKLData<DType>::create();
    LayerSetUp(in_shapes, out_shapes);
  }
  virtual ~MKLElementWiseSumOnlyFwdOp() {
    dnnDelete<DType>(sum_primitive_);
  }

 private:
  void SanityCheck(
      const std::vector<TShape>& in_shapes,
      const std::vector<TShape>& out_shapes) const {
    CHECK_EQ(in_shapes.size(), num_args_);
    CHECK_EQ(out_shapes.size(), 1);
    for (const TShape& shp : in_shapes) {
      CHECK_EQ(shp, out_shapes[0]);
    }
  }
  void LayerSetUp(
      const std::vector<TShape>& in_shapes,
      const std::vector<TShape>& out_shapes) {
    SanityCheck(in_shapes, out_shapes);
    padded_dshape_ = ConvertTo4DShape(in_shapes[0]);
    coeffs_ = std::vector<DType>(padded_dshape_.Size(), 1);

    const size_t dim_src = 4;
    std::vector<size_t> sizes_src(dim_src), strides_src(dim_src);
    for (size_t d = 0; d < dim_src; ++d) {
      sizes_src[d] = in_shapes[0][dim_src - d - 1];
      strides_src[d] = (d == 0) ? 1 : strides_src[d - 1] * sizes_src[d - 1];
    }

    // Create user layout.
    for (size_t i = 0; i < num_args_; ++i) {
      fwd_bottom_data_.push_back(MKLData<DType>::create());
      fwd_bottom_data_[i]->create_user_layout(
          dim_src, &sizes_src[0], &strides_src[0]);
    }
    fwd_top_data_->create_user_layout(
        dim_src, &sizes_src[0], &strides_src[0]);

#if MKL_EXPERIMENTAL == 0
    // Only create SumPrimitive when NOT using this flag. Otherwise, layout information
    // is required.
    MKLDNN_CALL(dnnSumCreate<DType>(
          &sum_primitive_, nullptr, num_args_, fwd_top_data_->layout_usr, &coeffs_[0]));
#endif
  }

 public:
  void CreateSumPrimitiveIfNeeded(const std::vector<TBlob>& in_data) {
#if MKL_EXPERIMENTAL == 1
    if (sum_primitive_) {
      // Primitive already been created.
      return;
    }
    bool has_mkl_prv_data = false;
    for (size_t i = 0; i < num_args_; i++) {
      if (mkl_prv_data<DType>(in_data[i]) != nullptr) {
        has_mkl_prv_data = true;
        break;
      }
    }
    if (has_mkl_prv_data) {
      dnnLayout_t int_layout = nullptr;
      for (size_t i = 0; i < num_args_; ++i) {
        std::shared_ptr<MKLData<DType>> mem_descr = GetMemDescr<DType>(in_data[i]);
        if (!mem_descr) {
          fwd_bottom_data_[i] = mem_descr;
          if (!int_layout) {
            int_layout = mem_descr->layout_int;
          }
        }
      }
      MKLDNN_CALL(dnnSumCreate<DType>(
            &sum_primitive_, nullptr, num_args_, int_layout, &coeffs_[0]));

      fwd_top_data_->create_internal_layout(sum_primitive_, dnnResourceDst);

      for (size_t i = 0; i < num_args_; ++i) {
        if (mkl_prv_data<DType>(in_data[i]) == nullptr) {
          fwd_bottom_data_[i]->create_internal_layout(sum_primitive_,
              (dnnResourceType_t)(dnnResourceMultipleSrc + i));
        }
      }
    }
#endif
  }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) override {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), num_args_);
    CHECK_EQ(out_data.size(), 1);

    if (req[elemsum::kOut] == kNullOp) {
      // Do nothing.
      return;
    }

    CreateSumPrimitiveIfNeeded(in_data);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 4, DType> > data(num_args_);
    for (size_t i = 0; i < num_args_; ++i) {
      data[i] = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
          in_data[i], padded_dshape_, s);
    }
    Tensor<xpu, 4, DType> out = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_data[elemsum::kOut], padded_dshape_, s);

    std::vector<void*> bottom_data;
    for (size_t i = 0; i < num_args_; i++) {
#if MKL_EXPERIMENTAL == 1
      void* i_data = mkl_prv_data<DType>(in_data[i]);
      if (i_data != nullptr) {
        bottom_data.push_back(i_data);
      } else {
        bottom_data.push_back(in_data[i].dptr_);
      }
#else
      bottom_data.push_back(in_data[i].dptr_);
#endif
    }

    void *eltwise_res[dnnResourceNumber];
    // Convert input layouts.
    for (size_t i = 0; i < num_args_; ++i) {
      if (fwd_bottom_data_[i]->conversion_needed()) {
        std::shared_ptr<MKLMemHolder> in_data_mem =
#if MKL_EXPERIMENTAL == 1
          in_data[i].Mkl_mem_;
#else
          nullptr;
#endif
        eltwise_res[dnnResourceMultipleSrc + i] =
          fwd_bottom_data_[i]->get_converted_prv(data[i].dptr_, false, in_data_mem);
      } else {
        eltwise_res[dnnResourceMultipleSrc + i] = bottom_data[i];
      }
    }

    // Convert output layouts.
    if (fwd_top_data_->conversion_needed()) {
#if MKL_EXPERIMENTAL == 1
      std::shared_ptr<MKLMemHolder> top_mem = out_data[elemsum::kOut].Mkl_mem_;
      top_mem->set_prv_descriptor(fwd_top_data_);
#endif
      eltwise_res[dnnResourceDst] = fwd_top_data_->prv_ptr();
    } else {
      eltwise_res[dnnResourceDst] = const_cast<DType*>(out.dptr_);
    }

    MKLDNN_CALL(dnnExecute<DType>(sum_primitive_, eltwise_res));

    // Convert output layouts.
    if (fwd_top_data_->conversion_needed()) {
      fwd_top_data_->convert_from_prv(out.dptr_);
    }
  }

 private:
  const size_t num_args_;
  mshadow::Shape<4> padded_dshape_;
  std::vector<DType> coeffs_;

  std::shared_ptr<MKLData<DType> > fwd_top_data_;
  std::vector< std::shared_ptr<MKLData<DType> > > fwd_bottom_data_;

  dnnPrimitive_t sum_primitive_{nullptr};
};  // class ElementWiseSumOp


template<typename xpu>
Operator *CreateOp(
    const ElementWiseSumOnlyFwdParam& param,
    int dtype,
    const std::vector<TShape>& in_shapes,
    const std::vector<TShape>& out_shapes);

#if DMLC_USE_CXX11
class ElementWiseSumOnlyFwdProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> ret;
    for (int i = 0; i < param_.num_args; ++i) {
      ret.push_back(std::string("arg") + static_cast<char>('0' + i));
    }
    return ret;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), param_.num_args);
    CHECK_EQ(out_shape->size(), 1);
    for (const TShape& shp : *in_shape) {
      CHECK_EQ(shp, (*in_shape)[0]);
    }
    SHAPE_ASSIGN_CHECK(*out_shape, elemsum::kOut, (*in_shape)[0]);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), param_.num_args);
    CHECK_EQ(out_type->size(), 1);
    for (const int& ty : *in_type) {
      CHECK_EQ(ty, (*in_type)[0]);
    }
    out_type->clear();
    out_type->push_back((*in_type)[0]);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ElementWiseSumOnlyFwdProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "ElementWiseSumOnlyFwd";
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  ElementWiseSumOnlyFwdParam param_;
};  // class ElementWiseSumOnlyFwdProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_ELEMENTWISE_FWD_ONLY_INL_H_
