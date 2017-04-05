/*******************************************************************************
* Copyright 2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* \file mkl_convolution-inl.h
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_MKL_MKL_CONVOLUTION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "./mkl_util-inl.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class MKLConvolutionOp : public Operator {
 public:
  static std::string getName() {
    return "MKLConvolutionOp";
  }
  

  explicit MKLConvolutionOp(ConvolutionParam p,
                            const std::vector<TShape>& in_shapes,
                            const std::vector<TShape>& out_shapes):
    param_(p) {
    SetupBuffer();
    LayerSetUp(in_shapes[conv::kData], out_shapes[conv::kOut]);
  }
  void ReleaseBuffer() {
    if (convolutionFwd != NULL) {
     dnnDelete<DType>(convolutionFwd);
     convolutionFwd = NULL;
    }
    if (convolutionBwdData != NULL) {
     dnnDelete<DType>(convolutionBwdData);
     convolutionBwdData = NULL;
    }
    if (convolutionBwdFilter != NULL) {
     dnnDelete<DType>(convolutionBwdFilter);
     convolutionBwdFilter = NULL;
    }
    if (!param_.no_bias && convolutionBwdBias != NULL) {
     dnnDelete<DType>(convolutionBwdBias);
     convolutionBwdBias = NULL;
    }
  }
  virtual ~MKLConvolutionOp() {
    ReleaseBuffer();
  }

 private:
  void SetupBuffer() {
    fwd_bottom_data = MKLData<DType>::create();
    fwd_top_data = MKLData<DType>::create();
    fwd_filter_data = MKLData<DType>::create();
    fwd_bias_data = MKLData<DType>::create();
    bwdd_top_diff = MKLData<DType>::create();
    bwdd_bottom_diff = MKLData<DType>::create();
    bwdd_filter_data = MKLData<DType>::create();
    bwdf_top_diff = MKLData<DType>::create();
    bwdf_filter_diff = MKLData<DType>::create();
    bwdf_bottom_data = MKLData<DType>::create();
    bwdb_top_diff = MKLData<DType>::create();
    bwdb_bias_diff = MKLData<DType>::create();
  }

  void LayerSetUp(
      const TShape& data_shape,
      const TShape& out_shape) {
    const size_t dimension = 4;
    const size_t g = std::max(param_.num_group, (uint32_t)1);
    const size_t n = data_shape[0];
    const size_t iw = data_shape[3];
    const size_t ih = data_shape[2];
    const size_t ic = data_shape[1];
    const size_t ow = out_shape[3];
    const size_t oh = out_shape[2];
    const size_t oc = out_shape[1];
    const size_t kw = param_.kernel[1];
    const size_t kh = param_.kernel[0];
    size_t bdata_sizes[4] = { iw, ih, ic, n };
    size_t bdata_strides[4] = { 1, iw, iw*ih, iw*ih*ic };
    /* starting with MKL 2017 Gold in case of groups filter layout
    * becomes 5D, i.e. groups become a separate dimension */
    size_t g_mkl2017 = g;
    size_t f_dimension = dimension + (g != 1);
    if (getMKLBuildDate() < 20160701) {
     g_mkl2017 = 1;
     f_dimension = dimension;
    }
    size_t fdata_sizes[5] = { kw, kh, ic / g, oc / g_mkl2017, g_mkl2017 };
    size_t fdata_strides[5] = { 1, kw, kw*kh, kw*kh*ic / g, kw*kh*ic / g*oc / g };
    size_t bias_sizes[1] = { oc };
    size_t bias_strides[1] = { 1 };
    size_t tdata_sizes[4] = { ow, oh, oc, n };
    size_t tdata_strides[4] = { 1, ow, ow*oh, ow*oh*oc };
    size_t convolutionStrides[2] = { param_.stride[1], param_.stride[0] };
    int    inputOffset[2] = { -param_.pad[1], -param_.pad[0] };
    // Names are for debugging purposes only.
    /*** convolution section ***/
    if (!param_.no_bias) {
      MKLDNN_CALL(dnnGroupsConvolutionCreateForwardBias<DType>(
            &convolutionFwd,
            nullptr,
            dnnAlgorithmConvolutionDirect,
            g,
            dimension,
            bdata_sizes,
            tdata_sizes,
            fdata_sizes,
            convolutionStrides,
            inputOffset,
            dnnBorderZeros));
    } else {
      MKLDNN_CALL(dnnGroupsConvolutionCreateForward<DType>(
            &convolutionFwd,
            nullptr,
            dnnAlgorithmConvolutionDirect,
            g,
            dimension,
            bdata_sizes,
            tdata_sizes,
            fdata_sizes,
            convolutionStrides,
            inputOffset,
            dnnBorderZeros));
    }
    fwd_bottom_data->create_layouts(convolutionFwd, dnnResourceSrc, dimension,
                                    bdata_sizes, bdata_strides);
    fwd_top_data->create_layouts(convolutionFwd, dnnResourceDst, dimension,
                                 tdata_sizes, tdata_strides);
    fwd_filter_data->create_layouts(convolutionFwd, dnnResourceFilter,
                                    f_dimension, fdata_sizes, fdata_strides);
    if (!param_.no_bias)
      fwd_bias_data->create_layouts(convolutionFwd, dnnResourceBias, 1,
                                    bias_sizes, bias_strides);
    /*
    * Backward by data layer setup
    */
    MKLDNN_CALL(dnnGroupsConvolutionCreateBackwardData<DType>(
          &convolutionBwdData,
          nullptr,
          dnnAlgorithmConvolutionDirect,
          g,
          dimension,
          bdata_sizes,
          tdata_sizes,
          fdata_sizes,
          convolutionStrides,
          inputOffset,
          dnnBorderZeros));
    bwdd_bottom_diff->create_layouts(convolutionBwdData, dnnResourceDiffSrc,
                                     dimension, bdata_sizes, bdata_strides);
    bwdd_top_diff->create_layouts(convolutionBwdData, dnnResourceDiffDst,
                                  dimension, tdata_sizes, tdata_strides);
    bwdd_filter_data->create_layouts(convolutionBwdData, dnnResourceFilter,
                                     f_dimension, fdata_sizes, fdata_strides);
    /*
    * Backward by filter layer setup
    */
    MKLDNN_CALL(dnnGroupsConvolutionCreateBackwardFilter<DType>(
          &convolutionBwdFilter,
          nullptr,
          dnnAlgorithmConvolutionDirect,
          g,
          dimension,
          bdata_sizes,
          tdata_sizes,
          fdata_sizes,
          convolutionStrides,
          inputOffset,
          dnnBorderZeros));
    bwdf_bottom_data->create_layouts(convolutionBwdFilter, dnnResourceSrc,
                                     dimension, bdata_sizes, bdata_strides);
    bwdf_top_diff->create_layouts(convolutionBwdFilter, dnnResourceDiffDst,
                                  dimension, tdata_sizes, tdata_strides);
    bwdf_filter_diff->create_layouts(convolutionBwdFilter, dnnResourceDiffFilter,
                                     f_dimension, fdata_sizes, fdata_strides);
    /*
    * Backward by bias layer setup
    */
    if (!param_.no_bias) {
      MKLDNN_CALL(dnnGroupsConvolutionCreateBackwardBias<DType>(
            &convolutionBwdBias,
            nullptr,
            dnnAlgorithmConvolutionDirect,
            g,
            dimension,
            tdata_sizes));
     bwdb_top_diff->create_layouts(convolutionBwdBias, dnnResourceDiffDst,
                                   dimension, tdata_sizes, tdata_strides);
     bwdb_bias_diff->create_layouts(convolutionBwdBias, dnnResourceDiffBias, 1,
                                    bias_sizes, bias_strides);
    }
  }

 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    DType *data_ptr = NULL;
    DType *wmat_ptr = NULL;
    DType *out_ptr = NULL;
    Tensor<xpu, 4, DType> data =
      mkl_experimental_direct_get<xpu, 4, DType>(in_data[conv::kData], s);
    Tensor<xpu, 4, DType> out =
      mkl_experimental_direct_get<xpu, 4, DType>(out_data[conv::kOut], s);
    Tensor<xpu, 4, DType> wmat =
      mkl_experimental_direct_get<xpu, 4, DType>(in_data[conv::kWeight], s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(wmat.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    data_ptr = data.dptr_;
    wmat_ptr = wmat.dptr_;
    out_ptr = out.dptr_;
    int status;
    void *res_convolutionFwd[dnnResourceNumber];
    std::shared_ptr<MKLMemHolder> in_data_mem =
#if MKL_EXPERIMENTAL == 1
      in_data[conv::kData].Mkl_mem_;
#else
      NULL;
#endif
    res_convolutionFwd[dnnResourceSrc] =
      fwd_bottom_data->get_converted_prv(data_ptr, false, in_data_mem);
    std::shared_ptr<MKLMemHolder> in_weight_mem =
#if MKL_EXPERIMENTAL == 1
      in_data[conv::kWeight].Mkl_mem_;
#else
      NULL;
#endif
    res_convolutionFwd[dnnResourceFilter] =
      fwd_filter_data->get_converted_prv(wmat_ptr, true, in_weight_mem);
    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> bias =
        mkl_experimental_direct_get<xpu, 1, DType>(in_data[conv::kBias], s);
      std::shared_ptr<MKLMemHolder> in_bias_mem =
#if MKL_EXPERIMENTAL == 1
       in_data[conv::kBias].Mkl_mem_;
#else
       NULL;
#endif
      res_convolutionFwd[dnnResourceBias] =
        fwd_bias_data->get_converted_prv(bias.dptr_, true, in_bias_mem);
    }

    std::shared_ptr<MKLMemHolder> top_mem =
#if MKL_EXPERIMENTAL == 1
     out_data[conv::kOut].Mkl_mem_;
#else
     NULL;
#endif
    if (fwd_top_data->conversion_needed()) {
      res_convolutionFwd[dnnResourceDst] =
        reinterpret_cast<void *>(fwd_top_data->prv_ptr());
#if MKL_EXPERIMENTAL == 1
      top_mem->set_prv_descriptor(fwd_top_data);
#endif
    } else {
      res_convolutionFwd[dnnResourceDst] = out_ptr;
    }
    status = dnnExecute<DType>(convolutionFwd, res_convolutionFwd);
    CHECK_EQ(status, 0) << "Forward convolution failed with status " << status;
#if MKL_EXPERIMENTAL == 0
    if (fwd_top_data->conversion_needed()) {
        fwd_top_data->convert_from_prv(out_ptr);
    }
#endif
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    if (param_.kernel.ndim() > 2) {
      LOG(FATAL) << "Volume convolution is not implmented in mshadow";
    }
    CHECK_EQ(out_grad.size(), 1);
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[conv::kWeight].CheckContiguous(), true);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data =
      mkl_experimental_direct_get<xpu, 4, DType>(in_data[conv::kData], s);
    Shape<3> wmat_shape =
      Shape3(param_.num_group,
             param_.num_filter / param_.num_group,
             data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    Tensor<xpu, 3, DType> wmat =
      mkl_experimental_direct_get_with_shape<xpu, 3, DType>(
      in_data[conv::kWeight], wmat_shape, s);
    Tensor<xpu, 4, DType> grad =
      mkl_experimental_direct_get<xpu, 4, DType>(out_grad[conv::kOut], s);
    Tensor<xpu, 4, DType> gdata =
      mkl_experimental_direct_get<xpu, 4, DType>(in_grad[conv::kData], s);
    Tensor<xpu, 3, DType> gwmat =
      mkl_experimental_direct_get_with_shape<xpu, 3, DType>(
      in_grad[conv::kWeight], wmat_shape, s);

    int status;
    if (req[0]) {
      void *res_convolutionBwdData[dnnResourceNumber];
      std::shared_ptr<MKLMemHolder> out_grad_mem =
#if MKL_EXPERIMENTAL == 1
       out_grad[conv::kOut].Mkl_mem_;
#else
       NULL;
#endif
      res_convolutionBwdData[dnnResourceDiffDst] =
        bwdd_top_diff->get_converted_prv(grad.dptr_, true, out_grad_mem);
      std::shared_ptr<MKLMemHolder> in_weight_mem =
#if MKL_EXPERIMENTAL == 1
        in_data[conv::kWeight].Mkl_mem_;
#else
        NULL;
#endif
      res_convolutionBwdData[dnnResourceFilter] =
        bwdd_filter_data->get_converted_prv(wmat.dptr_, false, in_weight_mem);
     if (bwdd_bottom_diff->conversion_needed()) {
       res_convolutionBwdData[dnnResourceDiffSrc] =
         reinterpret_cast<void *>(bwdd_bottom_diff->prv_ptr());
#if MKL_EXPERIMENTAL == 1
       std::shared_ptr<MKLMemHolder> bottom_diff_mem =
         in_grad[conv::kData].Mkl_mem_;
       bottom_diff_mem->set_prv_descriptor(bwdd_bottom_diff);
#endif
     } else {
       res_convolutionBwdData[dnnResourceDiffSrc] = gdata.dptr_;
     }
     status = dnnExecute<DType>(convolutionBwdData, res_convolutionBwdData);
     CHECK_EQ(status, 0) << "Backward Data conv failed with status " << status;
#if MKL_EXPERIMENTAL == 0
     if (bwdd_bottom_diff->conversion_needed()) {
       bwdd_bottom_diff->convert_from_prv(gdata.dptr_);
     }
#endif
    }
    if (req[1]) {
      void *res_convolutionBwdFilter[dnnResourceNumber];
      std::shared_ptr<MKLMemHolder> out_bias_mem =
#if MKL_EXPERIMENTAL == 1
        out_grad[conv::kOut].Mkl_mem_;
#else
        NULL;
#endif
      res_convolutionBwdFilter[dnnResourceDiffDst] =
        bwdf_top_diff->get_converted_prv(grad.dptr_, true, out_bias_mem);
      MKLMemoryDescriptor<DType>* fwd_bottom_data_desc = NULL;
#if MKL_EXPERIMENTAL == 1
      std::shared_ptr<MKLMemHolder> in_data_mem = in_data[conv::kData].Mkl_mem_;
      fwd_bottom_data_desc = fwd_bottom_data.get();
#else
      std::shared_ptr<MKLMemHolder> in_data_mem = NULL;
#endif
      res_convolutionBwdFilter[dnnResourceSrc] =
        bwdf_bottom_data->get_converted_prv(data.dptr_, false,
                                            in_data_mem,
                                            fwd_bottom_data_desc);
     if (bwdf_filter_diff->conversion_needed()) {
#if MKL_EXPERIMENTAL == 1
       std::shared_ptr<MKLMemHolder> gwamt_mem =
         in_grad[conv::kWeight].Mkl_mem_;
       gwamt_mem->set_prv_descriptor(bwdf_filter_diff);
#endif
       res_convolutionBwdFilter[dnnResourceDiffFilter] =
         reinterpret_cast<void *>(bwdf_filter_diff->prv_ptr());
     } else {
       res_convolutionBwdFilter[dnnResourceDiffFilter] = gwmat.dptr_;
     }
     status = dnnExecute<DType>(convolutionBwdFilter, res_convolutionBwdFilter);
     CHECK_EQ(status, 0) << "Backward Filter conv failed with status " << status;
#if MKL_EXPERIMENTAL == 0
     if (bwdf_filter_diff->conversion_needed()) {
       bwdf_filter_diff->convert_from_prv(gwmat.dptr_);
     }
#endif
    }
    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> gbias =
        mkl_experimental_direct_get<xpu, 1, DType>(in_grad[conv::kBias], s);
      void *res_convolutionBwdBias[dnnResourceNumber];
      std::shared_ptr<MKLMemHolder> out_grad_mem =
#if MKL_EXPERIMENTAL == 1
        out_grad[conv::kOut].Mkl_mem_;
#else
        NULL;
#endif
      res_convolutionBwdBias[dnnResourceDiffDst] =
        bwdb_top_diff->get_converted_prv(grad.dptr_, true, out_grad_mem);
      if (bwdb_bias_diff->conversion_needed()) {
#if MKL_EXPERIMENTAL == 1
        std::shared_ptr<MKLMemHolder> gbias_mem = in_grad[conv::kBias].Mkl_mem_;
        gbias_mem->set_prv_descriptor(bwdb_bias_diff);
#endif
        res_convolutionBwdBias[dnnResourceDiffBias] =
          bwdb_bias_diff->prv_ptr();
      } else {
        res_convolutionBwdBias[dnnResourceDiffBias] =
          reinterpret_cast<void *>(gbias.dptr_);
      }
      status = dnnExecute<DType>(convolutionBwdBias, res_convolutionBwdBias);
      CHECK_EQ(status, 0) << "Backward Bias failed with status " << status;
#if MKL_EXPERIMENTAL == 0
      if (bwdb_bias_diff->conversion_needed()) {
        bwdb_bias_diff->convert_from_prv(gbias.dptr_);
      }
#endif
    }
  }

 private:
  const ConvolutionParam param_;

  dnnPrimitive_t convolutionFwd{nullptr};
  dnnPrimitive_t convolutionBwdData{nullptr};
  dnnPrimitive_t convolutionBwdFilter{nullptr};
  dnnPrimitive_t convolutionBwdBias{nullptr};
  /* Fwd step */
  std::shared_ptr<MKLData<DType> > fwd_bottom_data, fwd_top_data, fwd_filter_data,
                                   fwd_bias_data;
  /* Bwd data step */
  std::shared_ptr<MKLData<DType> > bwdd_top_diff, bwdd_bottom_diff;
  std::shared_ptr<MKLData<DType> > bwdd_filter_data;
  /* Bwd filter step */
  std::shared_ptr<MKLData<DType> > bwdf_top_diff, bwdf_filter_diff;
  std::shared_ptr<MKLData<DType> > bwdf_bottom_data;
  std::shared_ptr<MKLData<DType> > bwdf_filter_diff_iter, bwdf2fwd_filter_diff,
                                   bwdb_bias_diff_iter;
  /* Bwd bias step */
  std::shared_ptr<MKLData<DType> > bwdb_top_diff, bwdb_bias_diff;
};  // class ConvolutionOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_CONVOLUTION_INL_H_
