/*!
 * Copyright (c) 2017 by Contributors
 * \file net_init-inl.h
 * \brief
 * \author Chien-Chin Huang
*/
#ifndef MXNET_OPERATOR_NET_INIT_INL_H_
#define MXNET_OPERATOR_NET_INIT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
namespace mxnet {
namespace op {

struct NetInitParam: public dmlc::Parameter<NetInitParam> {
  std::string address; 
  DMLC_DECLARE_PARAMETER(NetInitParam) {
    DMLC_DECLARE_FIELD(address).set_default("127.0.0.1:11111")
    .describe("The address and port (ip:port) this worker should listen.");
  }
};  // struct NetInitParam

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NET_INIT_INL_H_
