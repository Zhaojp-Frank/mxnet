#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <unordered_map>
#include <mxnet/mm_dptr.h>
#include <mxnet/sa_util.h>
#include "./swapadv_mm_dptr.h"

namespace mxnet {
namespace storage {

SA_MM_Dptr::SA_MM_Dptr() {
  // TODO(fegin): Determine this dynamically.
  memory_size_ = 3L * 1024 * 1024 * 1024;
  temp_size_ = 0.5L * 1024 * 1024 * 1024;
  cudaHostAlloc((void**)&(cpu_memory_), 1L * 1024 * 1024 * 1024, 0);
  if (cpu_memory_ == nullptr) {
    LOG(FATAL) << "Can't allocate the CPU memory in the initialization of "
               << "the memory manager. The required size = ";
  }
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
  cudaStreamCreate(&stream_out_);
  cudaStreamCreate(&stream_in_);

  ReadScheduleDepsRst();
  ReadAllocationRst();
  ReadInitialHandlesRst();
  ReadDeallocationRst();

  void* address = memory_;
  for (size_t i = 0; i < mempool_counts_.size(); i++) {
    mempools_.emplace_back(std::vector<void*>());
    used_mempools_.emplace_back(std::unordered_map<void*, uint32_t>());
    for (size_t j = 0; j < mempool_counts_[i]; j++) {
      mempools_[i].push_back(address);
      address = (void*)((size_t)address + mempool_to_size_[i]);
    }
  }
  alloc_finalized_ = false;
  used_memory_ = 0;
  temp_user_ = 0;
  iteration_started = false;
  curr_iteration = -1;
  sa_log << "SA_MM_Dptr initialized" << std::endl;
}

void SA_MM_Dptr::ReadScheduleDepsRst() {
  sa_log << "ReadScheduleDepsRst" << std::endl;
  std::ifstream ifs("schedule.rst");
  std::string line;
  size_t next = 0, last = 0;
  while (std::getline(ifs, line)) {
    next = line.find(":", last);
    uint32_t nid = std::stol(line.substr(last, next - last));
    uint32_t dep_nid = std::stol(line.substr(next + 1));
    schedule_deps_[nid].push_back(dep_nid);
  }
}

void SA_MM_Dptr::ReadAllocationRst() {
  sa_log << "ReadAllocationRst" << std::endl;
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

void SA_MM_Dptr::ReadDeallocationRst() {
  sa_log << "ReadDeallocationRst" << std::endl;
  std::ifstream ifs("deallocation.rst");
  std::string line;

  while (std::getline(ifs, line)) {
    size_t next = 0, last = 0;
    next = line.find(",", last);
    uint32_t nid = std::stoi(line.substr(last, next - last));
    last = next + 1;
    while ((next = line.find(",", last)) != std::string::npos) {
      handle_id_t hid = std::stoi(line.substr(last, next - last));
      last = next + 1;
      deallocations_[nid].push_back(hid);
    }
  }
  sa_log << "Deallocations_.size() = " << deallocations_.size() << std::endl;
}

void SA_MM_Dptr::ReadInitialHandlesRst() {
  sa_log << "ReadInitialHandlesRst" << std::endl;
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

void* SA_MM_Dptr::Alloc_(handle_id_t id, bool do_swapin) {
  sa_log << "Alloc_ " << id << std::endl;
  // Check if the handle is already in memory.
  lock_.Lock();
  auto it = hdl_dptr_mapping_.find(id);
  if (it != hdl_dptr_mapping_.end()) {
    lock_.UnLock();
    return it->second;
  }
  sa_log << "Alloc_ " << id << " not in memory " << std::endl;
  CHECK(!alloc_finalized_ || new_to_old_hids_.count(id) > 0);
  // Check if we still have mempool.
  uint32_t mempool_idx = hdl_to_mempool_.at(id);
  auto& mempool = mempools_.at(mempool_idx);
  size_t counter = 0;
  while (mempool.size() == 0) {
    lock_.UnLock();
    usleep(30);
    counter += 1;
    lock_.Lock();
    if (counter % 10000 == 9999) {
      sa_log << std::dec << "Wait!!!" << id << " " << new_to_old_hids_.at(id)
             << " " << mempool_idx << " " << mempools_[mempool_idx].size()
             << std::endl;
      for (auto it : used_mempools_.at(mempool_idx)) {
        sa_log << it.second << "," << new_to_old_hids_.at(it.second)
               << std::endl;
      }
    }
  }
  sa_log << "Alloc_ " << id << " found " << std::endl;
  void* address = mempool.back();
  mempool.pop_back();
  used_mempools_[mempool_idx][address] = id;
  if (do_swapin) {
    size_t size = hdl_size_mapping_.at(id);
    lock_.UnLock();
    cudaMemcpyAsync(address, cpu_memory_, size, cudaMemcpyHostToDevice,
                    stream_in_);
    cudaStreamSynchronize(stream_in_);
    lock_.Lock();
  }
  hdl_dptr_mapping_[id] = address;
  lock_.UnLock();
  return address;
}

void SA_MM_Dptr::Free_(handle_id_t id, bool do_swapout) {
  sa_log << "FREE_ 1 " << id << std::endl;
  lock_.Lock();
  sa_log << "FREE_ 2" << std::endl;
  uint32_t mempool = hdl_to_mempool_.at(id);
  auto it = hdl_dptr_mapping_.find(id);
  if (it == hdl_dptr_mapping_.end()) {
    sa_log << "FREE_ 2.1" << std::endl;
    CHECK(false);
  }
  sa_log << "FREE_ 3" << std::endl;
  size_t size = hdl_size_mapping_.at(id);
  sa_log << "FREE_ 3" << id << std::endl;
  void* address = it->second;
  if (do_swapout) {
    lock_.UnLock();
    cudaMemcpyAsync(address, cpu_memory_, size, cudaMemcpyDeviceToHost,
                    stream_out_);
    cudaStreamSynchronize(stream_out_);
    lock_.Lock();
  }
  hdl_dptr_mapping_.erase(it);
  CHECK_EQ(used_mempools_[mempool].erase(address), 1);
  mempools_.at(mempool).push_back(address);
  lock_.UnLock();
}

void* SA_MM_Dptr::FreeAlloc_(handle_id_t id) {
  sa_log << "FreeAlloc_ " << id << std::endl;
  CHECK(!iteration_started);
  CHECK_EQ(curr_iteration, -1);
  uint32_t mempool = hdl_to_mempool_.at(id);
  auto it = used_mempools_[mempool].begin();
  void* address = it->first;
  CHECK_EQ(hdl_dptr_mapping_.erase(it->second), 1);
  it->second = id;
  hdl_dptr_mapping_[id] = address;
  return address;
}

void SA_MM_Dptr::StartIteration() {
  sa_log << "StartIteration " << std::endl;
  iteration_started = true;
  curr_iteration += 1;
  if (curr_iteration == 0) {
    // Release all the memory.
    std::unordered_map<handle_id_t, void*> hdl_dptr_mapping(hdl_dptr_mapping_);
    for (auto &it : hdl_dptr_mapping) {
      Free_(it.first, false);
    }
    for (uint32_t i = 0; i < used_mempools_.size(); i++) {
      if (used_mempools_[i].size() != 0) {
        sa_log << "used_mempools_[" << i << "] = "
               << used_mempools_[i].begin()->first << ","
               << used_mempools_[i].begin()->second << std::endl;
      }
      CHECK_EQ(used_mempools_[i].size(), 0);
    }
    // Swapin all weights should be in the memory in the beginning.
    for (auto old_hid : initial_handles_) {
      handle_id_t hid = old_to_new_hids_.at(old_hid);
      void* address = Alloc_(hid, false);
      sa_log << "Initial handles " << hid << " " << old_hid << " "
                << address << std::endl;
    }
  } else {
    size_t size_in_memory = 0;
    for (uint32_t i = 0; i < used_mempools_.size(); i++) {
      size_in_memory += used_mempools_[i].size();
    }
    CHECK_EQ(size_in_memory, hdl_dptr_mapping_.size());
    CHECK_EQ(size_in_memory, initial_handles_.size());
  }
  created_handles_ = arg_handles_;
  return;
}

void* SA_MM_Dptr::Alloc(handle_id_t id, size_t size, void* ptr=nullptr) {
  sa_log << "SA_MM_Dptr Alloc " << id << ", size " << size << std::endl;
  //std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  if (alloc_finalized_) {
    // FIXME(fegin)
    // CHECK_EQ(temp_user_, 0);
    temp_user_ = id;
    hdl_dptr_mapping_[id] = temp_memory_;
    sa_log << "SA_MM_Dptr Alloc temporary memory done " << id << std::endl;
    return temp_memory_;
  } else {
    if (size == 4) {
      swap_handles_.insert(id);
      return (void*)1;
    }
    size_t mempool = rsize_to_mempool_.at(size);
    hdl_size_mapping_[id] = size;
    hdl_to_mempool_[id]= mempool;
    if (doing_allocargs_) {
      arg_handles_.insert(id);
    }
    void* address = nullptr;
    if (mempools_.at(mempool).size() == 0) {
      address = FreeAlloc_(id);
    } else {
      address = Alloc_(id, false);
    }
    sa_log << "SA_MM_Dptr Alloc done " << address << std::endl;
    return address;
  }
}

void SA_MM_Dptr::Swapin(uint32_t old_nid, uint32_t idx) {
  uint32_t nid = old_to_new_nids_[old_nid];
  handle_id_t hid = entry_hdl_mapping_.at(EID(nid, idx)).first;
  sa_log << "About to swapin " << hid << std::endl;
  Alloc_(hid, true);
}

void SA_MM_Dptr::Swapout(uint32_t old_nid, uint32_t idx) {
  uint32_t nid = old_to_new_nids_[old_nid];
  handle_id_t hid = entry_hdl_mapping_.at(EID(nid, idx)).first;
  sa_log << "About to swapout " << hid << std::endl;
  Free_(hid, true);
}

void* SA_MM_Dptr::GetDptr_(handle_id_t id) {
  auto old_it = new_to_old_hids_.find(id);
  if (old_it != new_to_old_hids_.end()) {
    sa_log << std::dec << "GetDptr_ " << id << " " << new_to_old_hids_.at(id)
           << std::endl;
  } else {
    sa_log << std::dec << "GetDptr_ " << id << " " << -1 << std::endl;
  }
  void* address = nullptr;
  if (created_handles_.count(id) == 0) {
    sa_log << "Not created handle" << std::endl;
    address = Alloc_(id, false);
    created_handles_.insert(id);
  } else {
    sa_log << "Created handle" << std::endl;
    lock_.Lock();
    auto it = hdl_dptr_mapping_.find(id);
    while (it == hdl_dptr_mapping_.end()) {
      lock_.UnLock();
      usleep(30);
      lock_.Lock();
      it = hdl_dptr_mapping_.find(id);
    }
    lock_.UnLock();
    if (lock_.TryLock()) {
      lock_.UnLock();
      sa_log << "TryLock success" << std::endl;
    }
    address = it->second;
    sa_log << "Created handle done " << std::endl;
  }
  return address;
}

void* SA_MM_Dptr::GetDptr(handle_id_t id) {
  if (swap_handles_.count(id) == 1) {
    return (void*)1;
  }
  sa_log << "SA_MM_Dptr GetDptr " << id << std::endl;
#if SWAPADV_REPORT_PROGRESS
  if (std::this_thread::get_id() == model_tid_) {
    if (std::find(model_access_.begin(), model_access_.end(), id) ==
        model_access_.end()) {
      model_access_.push_back(id);
    }
  }
#endif
  void* address = nullptr;
  if (alloc_finalized_) {
    address = GetDptr_(id);
  } else {
    auto it = hdl_dptr_mapping_.find(id);
    address = (it != hdl_dptr_mapping_.end()) ? it->second : FreeAlloc_(id);
  }
  sa_log << "SA_MM_Dptr GetDptr done " << address << std::endl;
  return address;
}

void SA_MM_Dptr::SetDptr(handle_id_t id, void* ptr, uint32_t dev_id) {
  if (ptr == nullptr) return;
  sa_log << "SA_MM_Dptr SetDptr " << id << " " << ptr << std::endl;
  for (auto& used_mempool : used_mempools_) {
    auto it = used_mempool.find(ptr);
    if (it != used_mempool.end()) {
      size_t size = hdl_size_mapping_.at(it->second);
      uint32_t mempool = hdl_to_mempool_.at(it->second);
      hdl_size_mapping_[id] = size;
      hdl_to_mempool_[id] = mempool;
      break;
    }
  }
  CHECK_EQ(hdl_size_mapping_.count(id), 1);
  CHECK_EQ(hdl_to_mempool_.count(id), 1);
  hdl_dptr_mapping_[id] = ptr;
}

void SA_MM_Dptr::ReportProgress() {
  std::thread::id curr_id = std::this_thread::get_id();
  std::ofstream result;
  if (curr_id == model_tid_) {
    result.open("mxnet_model_progress.rst",
                std::ofstream::out | std::ofstream::app);
    std::sort(model_access_.begin(), model_access_.end());
    for (auto hid : model_access_) {
      if (new_to_old_hids_.count(hid) == 0) continue;
      handle_id_t old_hid = new_to_old_hids_.at(hid);
      if (new_to_old_nids_.at(model_nid_) == 1457) {
        sa_log << "1475=====" << model_access_.size() << std::endl;
      }
      result << new_to_old_nids_.at(model_nid_) << ", " <<  old_hid << ":";
      int mempool_idx = hdl_to_mempool_.at(hid);
      std::vector<handle_id_t> handles;
      for (auto it : used_mempools_[mempool_idx]) {
        handles.push_back(new_to_old_hids_.at(it.second));
      }
      std::sort(handles.begin(), handles.end());
      for (auto used_hid : handles) {
        result << " " << used_hid << ",";
      }
      result << std::endl;
    }
  } else if (curr_id == swapin_tid_) {
    result.open("mxnet_swapi_progress.rst",
                std::ofstream::out | std::ofstream::app);
    result << swapin_nid_ << std::endl;
  } else if (curr_id == swapout_tid_) {
    result.open("mxnet_swapo_progress.rst",
                std::ofstream::out | std::ofstream::app);
    result << swapout_nid_ << std::endl;
  } else {
    return;
  }
  result.close();
}

void SA_MM_Dptr::NotifyDone(uint32_t id) {
  sa_log << "NotifyDone " << id << std::endl;
  auto old_it = new_to_old_nids_.find(id);
  if (old_it != new_to_old_nids_.end()) {
    auto it = deallocations_.find(old_it->second);
    if (it != deallocations_.end()) {
      sa_log << "Doing deallocation for nid = " << id << std::endl;
      for (auto hid : it->second) {
        Free_(old_to_new_hids_.at(hid), false);
      }
    }
  }
#if SWAPADV_REPORT_PROGRESS
  ReportProgress();
#endif
}

SA_MM_Dptr* SA_MM_DPTR() {
  return dynamic_cast<SA_MM_Dptr*>(MM_DPTR());
}

}   // namespace storage
}   // namespace mxnet
