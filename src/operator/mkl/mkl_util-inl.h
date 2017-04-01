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
* \file mkl_util-inl.h
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_UTIL_INL_H_
#define MXNET_OPERATOR_MKL_MKL_UTIL_INL_H_
#include "../operator_common.h"

namespace mxnet {
namespace op {

#if MKL_EXPERIMENTAL == 1
template<typename DType>
inline DType * mkl_prv_data(const TBlob &b) {
  std::shared_ptr<MKLMemHolder> bottom_data_mem = b.Mkl_mem_;
  bool mem_valid = (bottom_data_mem != nullptr) && bottom_data_mem->head_at_prv();
  if (mem_valid) {
    return reinterpret_cast<DType*>(bottom_data_mem->prv_data());
  }
  return NULL;
}

template<typename DType>
inline int mkl_prv_count(const TBlob &b) {
  std::shared_ptr<MKLMemHolder> bottom_data_mem = b.Mkl_mem_;
  bool mem_valid = (bottom_data_mem != nullptr) && bottom_data_mem->head_at_prv();
  if (mem_valid) {
    return bottom_data_mem->prv_count();
  }
  return 0;
}
#endif
inline void mkl_set_priv_flag(const TBlob &b) {
#if MKL_EXPERIMENTAL == 1
  std::shared_ptr<MKLMemHolder> bottom_data_mem = b.Mkl_mem_;
  bool mem_valid = (bottom_data_mem != nullptr) && bottom_data_mem->head_at_prv();
  if (mem_valid) {
    bottom_data_mem->disable_prv_2_cpu(true);
  }
#endif
}
template<typename xpu, int dim, typename DType>
inline  mshadow::Tensor<xpu, dim, DType> mkl_experimental_direct_get(
  const TBlob &b, mshadow::Stream<xpu> *s) {
  mkl_set_priv_flag(b);
  return b.get<xpu, dim, DType>(s);
}
template<typename xpu, int dim, typename DType>
inline  mshadow::Tensor<xpu, dim, DType> mkl_experimental_direct_get_with_shape(
  const TBlob &b, const mshadow::Shape<dim> &shape, mshadow::Stream<xpu> *s) {
  mkl_set_priv_flag(b);
  return b.get_with_shape<xpu, dim, DType>(shape, s);
}

// If shape has less than four dimensions, left pad one.
// Else, flatten dimensions larger than the forth dimension.
inline TShape ConvertTo4DShape(const TShape& shape) {
  switch (shape.ndim()) {
  case 1:
    return mshadow::Shape4(shape[0], 1, 1, 1);
  case 2:
    return mshadow::Shape4(shape[0], shape[1], 1, 1);
  case 3:
    return mshadow::Shape4(shape[0], shape[1], shape[2], 1);
  default:
    return mshadow::Shape4(shape[0], shape[1], shape[2], shape.ProdShape(3, shape.ndim()));
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_UTIL_INL_H_
