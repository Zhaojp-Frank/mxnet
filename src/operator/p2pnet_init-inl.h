/*!
 * Copyright (c) 2017 by Contributors
 * \file p2pnet_init-inl.h
 * \brief
 * \author Chien-Chin Huang
*/
#ifndef MXNET_OPERATOR_P2PNET_INIT_INL_H_
#define MXNET_OPERATOR_P2PNET_INIT_INL_H_
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

struct P2PNetInitParam: public dmlc::Parameter<P2PNetInitParam> {
  std::string address;
  DMLC_DECLARE_PARAMETER(P2PNetInitParam) {
    DMLC_DECLARE_FIELD(address).set_default("127.0.0.1:11111")
    .describe("The address and port (ip:port) this worker should listen.");
  }
};  // struct P2PNetInitParam

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_P2PNET_INIT_INL_H_
