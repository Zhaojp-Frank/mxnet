#ifndef MXNET_NNVM_LEGACY_OP_UTIL_H_
#define MXNET_NNVM_LEGACY_OP_UTIL_H_

#include <dmlc/base.h>
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <nnvm/node.h>

namespace mxnet {
namespace op {

class ParsedOpProp {
 public:
  std::shared_ptr<OperatorProperty> ptr;
  std::vector<std::string> arguments;
  std::vector<std::string> aux_states;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  // initializer
  void Init(const nnvm::NodeAttrs& attrs) {
    std::vector<std::pair<std::string, std::string> > kwargs(
        attrs.dict.begin(), attrs.dict.end());
    try {
      ptr->Init(kwargs);
    } catch (const dmlc::ParamError& e) {
      std::ostringstream os;
      os << e.what();
      os << ", in operator " << attrs.op->name << "("
         << "name=\"" << attrs.name << "\"";
      for (const auto& k : attrs.dict) {
        os << ", " << k.first << "=\"" << k.second << "\"";
      }
      os << ")";
      throw dmlc::ParamError(os.str());
    }
    arguments = ptr->ListArguments();
    aux_states = ptr->ListAuxiliaryStates();
    outputs = ptr->ListOutputs();
    inputs = arguments;
    inputs.insert(
        inputs.end(), aux_states.begin(), aux_states.end());
  }
};

void RegisterOpAlignedSchemes();

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_NNVM_LEGACY_OP_UTIL_H_
