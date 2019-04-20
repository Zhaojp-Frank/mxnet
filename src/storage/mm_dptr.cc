#include <unordered_map>
#include <mxnet/mm_dptr.h>
#include "./pooled_storage_manager.h"
#include "./swapadvisor_storage_manager.h"

namespace mxnet {
MM_Dptr* MM_DPTR() {
  static MM_Dptr *mm_dptr = nullptr;
  if (mm_dptr != nullptr) {
    return mm_dptr;
  }
  const string type = dmlc::GetEnv("MXNET_GPU_MEM_POOL_TYPE", "Naive");
  if (mm_type == "Round" || mm_type == "Naive") {
    mm_dptr = new Pooled_MM_Dptr();
  } else if (mm_type == "SwapAdv") {
    mm_dptr = new SA_MM_Dptr();
  } else if (mm_type == "SwapOnDemand") {
  } else {
    CHECK(0) << "Unknown mm type.";
  }
}
}   // namespace mxnet

