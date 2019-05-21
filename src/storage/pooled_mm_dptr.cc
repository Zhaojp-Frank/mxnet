#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <unordered_map>
#include <mxnet/mm_dptr.h>
#include <mxnet/sa_util.h>
#include "./pooled_mm_dptr.h"

namespace mxnet {
namespace storage {

Pooled_MM_Dptr* POOLED_MM_DPTR() {
  return dynamic_cast<Pooled_MM_Dptr*>(MM_DPTR());
}

}   // namespace storage
}   // namespace mxnet
