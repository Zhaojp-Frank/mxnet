#include <unordered_map>
#include <mxnet/mm_dptr.h>
#include "./pooled_storage_manager.h"
#include "./swapadv_storage_manager.h"

namespace mxnet {
namespace storage {
MM_Dptr* MM_DPTR() {
  static MM_Dptr *mm_dptr = nullptr;
  if (mm_dptr != nullptr) {
    return mm_dptr;
  }
  const char *type = getenv("MXNET_GPU_MEM_POOL_TYPE");
  std::string mm_type = "Naive";
  if (type != nullptr) {
    mm_type = type;
  }
  if (mm_type == "Round" || mm_type == "Naive") {
    mm_dptr = new Pooled_MM_Dptr();
  } else if (mm_type == "SwapAdv") {
    mm_dptr = new SA_MM_Dptr();
  } else if (mm_type == "SwapOnDemand") {
  } else {
    CHECK(0) << "Unknown mm type.";
  }
  return mm_dptr;
}
}   // namespace storage
}   // namespace mxnet
