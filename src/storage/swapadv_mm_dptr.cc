#include <unordered_map>
#include <mxnet/mm_dptr.h>
#include "./swapadv_mm_dptr.h"

namespace mxnet {
namespace storage {

SA_MM_Dptr::SA_MM_Dptr() {
  // TODO(fegin): Determine this dynamically.
  memory_size_ = 3L * 1024 * 1024 * 1024;
  temp_size_ = 0.5L * 1024 * 1024 * 1024;
  cudaError_t e = cudaMalloc(&memory_, memory_size_);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    LOG(FATAL) << "Can't allocate the memory in the initialization of "
               << "the memory manager. The required size = " << memory_size_
               <<cudaGetErrorString(e);
  }
  e = cudaMalloc(&temp_memory_, temp_size_);
  if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
    LOG(FATAL) << "Can't allocate the memory in the initialization of "
               << "the memory manager. The required size = " << temp_size_
               <<cudaGetErrorString(e);
  }

  ReadAllocationRst();
  ReadInitialHandlesRst();

  void* address = memory_;
  for (size_t i = 0; i < mempool_counts_.size(); i++) {
    mempools_.emplace_back(std::vector<void*>());
    used_mempools_.emplace_back(std::unordered_map<void*, uint32_t>());
    for (size_t j = 0; j < mempool_counts_[i]; j++) {
      mempools_[i].push_back(address);
      address = (void*)((size_t)address + mempool_to_size_[i]);
    }
  }
  regular_finalized_ = false;
  used_memory_ = 0;
  temp_user_ = 0;
  iteration_started = false;
  curr_iteration = -1;
  std::cout << "SA_MM_Dptr initialized" << std::endl;
}

void* SA_MM_Dptr::Alloc(handle_id_t id, size_t size, void* ptr=nullptr) {
  std::cout << "SA_MM_Dptr Alloc " << id << ", size " << size << std::endl;
  //std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  if (regular_finalized_) {
    CHECK_EQ(temp_user_, 0);
    temp_user_ = id;
    hdl_dptr_mapping_[id] = temp_memory_;
    std::cout << "SA_MM_Dptr Alloc temporary memory done" << std::endl;
    return temp_memory_;
  } else {
    size_t mempool = rsize_to_mempool_.at(size);
    hdl_size_mapping_[id] = size;
    hdl_to_mempool_[id]= mempool;
    void* address = nullptr;
    if (mempools_.at(mempool).size() == 0) {
      address = FreeAlloc_(id);
    } else {
      address = Alloc_(id);
    }
    std::cout << "SA_MM_Dptr Alloc done " << address << std::endl;
    return address;
  }
}

void SA_MM_Dptr::ReadAllocationRst() {
  std::cout << "ReadAllocationRst" << std::endl;
  std::ifstream ifs("memalloc.rst");
  std::string line;

  size_t next = 0, last = 0;
  std::getline(ifs, line);
  while ((next = line.find(",", last)) != std::string::npos) {
    next = line.find(",", last);
    size_t size = std::stol(line.substr(last, next - last));
    last = next + 1;
    mempool_to_size_.emplace_back(size);
  }

  std::getline(ifs, line);
  next = last = 0;
  while ((next = line.find(",", last)) != std::string::npos) {
    next = line.find(",", last);
    uint32_t count = std::stoi(line.substr(last, next - last));
    last = next + 1;
    mempool_counts_.emplace_back(count);
  }

  std::getline(ifs, line);
  next = last = 0;
  while ((next = line.find(",", last)) != std::string::npos) {
    next = line.find(",", last);
    size_t rsize = std::stol(line.substr(last, next - last));
    last = next + 1;
    next = line.find(",", last);
    uint32_t pool = std::stoi(line.substr(last, next - last));
    last = next + 1;
    rsize_to_mempool_[rsize] = pool;
  }
}

void SA_MM_Dptr::ReadInitialHandlesRst() {
  std::cout << "ReadInitialHandlesRst" << std::endl;
  std::ifstream ifs("initial_handles.rst");
  std::string line;

  size_t next = 0, last = 0;
  std::getline(ifs, line);
  while ((next = line.find(",", last)) != std::string::npos) {
    uint32_t hid = std::stoi(line.substr(last, next - last));
    last = next + 1;
    initial_handles_.push_back(hid);
  }
}

void SA_MM_Dptr::StartIteration() {
  std::cout << "StartIteration " << std::endl;
  iteration_started = true;
  curr_iteration += 1;
  if (curr_iteration == 0) {
    // Release all the memory.
    std::unordered_map<handle_id_t, void*> hdl_dptr_mapping(hdl_dptr_mapping_);
    for (auto &it : hdl_dptr_mapping) {
      Free_(it.first);
    }
    for (uint32_t i = 0; i < used_mempools_.size(); i++) {
      CHECK_EQ(used_mempools_[i].size(), 0);
    }
    // Swapin all weights should be in the memory in the beginning.
    for (auto hid : initial_handles_) {
      void* address = Alloc_(hid);
      std::cout << "Initial handles " << address << std::endl;
    }
  } else {
    size_t size_in_memory = 0;
    for (uint32_t i = 0; i < used_mempools_.size(); i++) {
      size_in_memory += used_mempools_[i].size();
    }
    CHECK_EQ(size_in_memory, hdl_dptr_mapping_.size());
    CHECK_EQ(size_in_memory, initial_handles_.size());
  }
  return;
}

void* SA_MM_Dptr::Swapin(handle_id_t id) {
  std::cout << "Swapin " << id << std::endl;
  void* address = nullptr;
  uint32_t mempool = hdl_to_mempool_.at(id);
  if (mempools_[mempool].size() > 0) {
    std::cout << "GetDptr 0" << std::endl;
    address = mempools_[mempool].back();
    mempools_[mempool].pop_back();
    used_mempools_[mempool][address] = id;
    hdl_dptr_mapping_[id] = address;
  } else {
    std::cout << "GetDptr 1" << std::endl;
    // TODO(fegin): Wait for the notification
    assert(false);
  }
  return address;
}

}   // namespace storage
}   // namespace mxnet
