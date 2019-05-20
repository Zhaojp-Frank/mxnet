#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <unordered_map>
#include <mxnet/mm_dptr.h>
#include <mxnet/sa_util.h>
#include "./swapadv_mm_dptr.h"

namespace mxnet {
namespace storage {

void ProgressTracker::ReportProgress(
      const NidMapping& new_to_old_nids,
      const HidMapping& new_to_old_hids,
      const HidSizeMapping& hdl_to_mempool,
      const std::vector<std::unordered_map<void*, node_t>>& used_mempools) {
  std::thread::id curr_id = std::this_thread::get_id();
  std::ofstream result;
  if (curr_id == model_tid_) {
    result.open("mxnet_model_progress.rst",
                std::ofstream::out | std::ofstream::app);
    std::sort(model_access_.begin(), model_access_.end());
    for (auto hid : model_access_) {
      if (new_to_old_hids.count(hid) == 0) continue;
      handle_t old_hid = new_to_old_hids.at(hid);
      result << new_to_old_nids.at(model_nid_) << ", " <<  old_hid << ":";
      int mempool_idx = hdl_to_mempool.at(hid);
      std::vector<handle_t> handles;
      for (auto it : used_mempools[mempool_idx]) {
        handles.push_back(new_to_old_hids.at(it.second));
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

SA_MM_Dptr::SA_MM_Dptr() {
  // TODO(fegin): Determine this dynamically.
  memory_size_ = 14L * 1024 * 1024 * 1024;
  temp_size_ = 0.5L * 1024 * 1024 * 1024;
  cudaHostAlloc((void**)&(cpu_memory_), 8L * 1024 * 1024 * 1024,
                cudaHostAllocPortable);
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
  //cudaStreamCreate(&stream_out_);
  //cudaStreamCreate(&stream_in_);
  cudaStreamCreateWithPriority(&stream_out_, cudaStreamNonBlocking, -1);
  cudaStreamCreateWithPriority(&stream_in_, cudaStreamNonBlocking, -2);

  //ReadScheduleDepsRst();
  ReadAllocationRst();
  ReadInitialHandlesRst();
  ReadDeallocationRst();

  void* address = memory_;
  for (size_t i = 0; i < mempool_counts_.size(); i++) {
    mempools_.emplace_back(std::vector<void*>());
    used_mempools_.emplace_back(std::unordered_map<void*, node_t>());
    for (size_t j = 0; j < mempool_counts_[i]; j++) {
      mempools_[i].push_back(address);
      address = (void*)((size_t)address + mempool_to_size_[i]);
    }
  }
  alloc_finalized_ = false;
  temp_hdl_ = 0;
  iteration_started_ = false;
  curr_iteration_ = -1;
  is_finished_ = true;
  sa_log << "SA_MM_Dptr initialized" << std::endl;
}

#if 0
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
#endif

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
    node_t nid = std::stoi(line.substr(last, next - last));
    last = next + 1;
    while ((next = line.find(",", last)) != std::string::npos) {
      handle_t hid = std::stol(line.substr(last, next - last));
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
    handle_t hid = std::stol(line.substr(last, next - last));
    last = next + 1;
    initial_handles_.push_back(hid);
  }
}

void SA_MM_Dptr::Remap() {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(0, &cpuset);
  int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

  std::unordered_map<node_t, std::vector<handle_t>> deallocations;
  for (auto& pair : deallocations_) {
    node_t new_nid = old_to_new_nids_.at(pair.first);
    for (auto old_hid : pair.second) {
      deallocations[new_nid].push_back(old_to_new_hids_.at(old_hid));
    }
  }
  deallocations_ = deallocations;
}

void* SA_MM_Dptr::Alloc_(handle_t hid, bool do_swapin) {
  sa_log << "Alloc_ " << hid << std::endl;
  // Check if the handle is already in memory.
  lock_.Lock();
  auto it = hdl_dptr_mapping_.find(hid);
  if (it != hdl_dptr_mapping_.end()) {
    lock_.UnLock();
    return it->second;
  }
  sa_log << "Alloc_ " << hid << " not in memory " << std::endl;
  CHECK(!alloc_finalized_ || new_to_old_hids_.count(hid) > 0) << hid;
  // Check if we still have mempool.
  uint32_t mempool_idx = hdl_to_mempool_.at(hid);
  auto& mempool = mempools_.at(mempool_idx);
  size_t counter = 0;
  bool print = true;
  while (mempool.size() == 0) {
    if (print) {
      sa_log << "We have to wait1!!!" << std::endl;
      print = false;
    }
    lock_.UnLock();
    usleep(200);
    lock_.Lock();
    counter += 1;
#ifdef SWAPADV_DEBUG
    if (counter % 10000 == 9999) {
      sa_log << std::dec << "Wait!!!" << hid << " " << new_to_old_hids_.at(hid)
             << " " << mempool_idx << " " << mempools_[mempool_idx].size()
             << std::endl;
      for (auto it : used_mempools_.at(mempool_idx)) {
        sa_log << it.second << "," << new_to_old_hids_.at(it.second)
               << std::endl;
      }
    }
#endif
  }
  sa_log << "Alloc_ " << hid << " found " << std::endl;
  void* address = mempool.back();
  if (do_swapin) {
    cudaMemcpyAsync(address, cpu_memory_, hdl_size_mapping_.at(hid),
                    cudaMemcpyHostToDevice, stream_in_);
    mempool.pop_back();
    used_mempools_[mempool_idx][address] = hid;
    hdl_dptr_mapping_[hid] = address;
    lock_.UnLock();
    cudaStreamSynchronize(stream_in_);
  } else {
    mempool.pop_back();
    used_mempools_[mempool_idx][address] = hid;
    hdl_dptr_mapping_[hid] = address;
    lock_.UnLock();
  }
  return address;
}

void SA_MM_Dptr::Free_(handle_t hid, bool do_swapout) {
  sa_log << "FREE_ 1 " << hid << std::endl;
  lock_.Lock();
  sa_log << "FREE_ 2" << std::endl;
  auto it = hdl_dptr_mapping_.find(hid);
  if (it == hdl_dptr_mapping_.end()) {
    lock_.UnLock();
    sa_log << "FREE_ 3" << std::endl;
    CHECK(is_finished_) << "Can't find the handle in the memory to free.";
    return;
  }
  void* address = it->second;
  uint32_t mempool_idx;
  if (do_swapout) {
    cudaMemcpyAsync(address, cpu_memory_, hdl_size_mapping_.at(hid),
                    cudaMemcpyDeviceToHost, stream_out_);
    hdl_dptr_mapping_.erase(it);
    mempool_idx = hdl_to_mempool_.at(hid);
    CHECK_EQ(used_mempools_[mempool_idx].erase(address), 1);
    lock_.UnLock();
    cudaStreamSynchronize(stream_out_);
    lock_.Lock();
  } else {
    hdl_dptr_mapping_.erase(it);
    mempool_idx = hdl_to_mempool_.at(hid);
    CHECK_EQ(used_mempools_[mempool_idx].erase(address), 1);
  }
  mempools_.at(mempool_idx).push_back(address);
  lock_.UnLock();
  sa_log << "FREE_ done" << std::endl;
}

void* SA_MM_Dptr::GetDptr_(handle_t hid) {
#ifdef SWAPADV_DEBUG
  auto old_it = new_to_old_hids_.find(hid);
  if (old_it != new_to_old_hids_.end()) {
    sa_log << std::dec << "GetDptr_ " << hid << " " << new_to_old_hids_.at(hid)
           << std::endl;
  } else {
    sa_log << std::dec << "GetDptr_ " << hid << " " << -1 << std::endl;
  }
#endif
  void* address = nullptr;
  if (created_handles_.count(hid) == 0) {
    sa_log << "Not created handle" << std::endl;
    address = Alloc_(hid, false);
    created_handles_.insert(hid);
  } else {
    sa_log << "Created handle" << std::endl;
    lock_.Lock();
    auto it = hdl_dptr_mapping_.find(hid);
    bool print = true;
    while (it == hdl_dptr_mapping_.end()) {
      if (print) {
          sa_log << "We have to wait2!!!" << std::endl;
          print = false;
      }
      lock_.UnLock();
      usleep(200);
      lock_.Lock();
      it = hdl_dptr_mapping_.find(hid);
    }
    lock_.UnLock();
    address = it->second;
    sa_log << "Created handle done " << std::endl;
  }
  return address;
}

void* SA_MM_Dptr::FreeAlloc_(handle_t hid) {
  sa_log << "FreeAlloc_ " << hid << std::endl;
  CHECK(!iteration_started_);
  CHECK_EQ(curr_iteration_, -1);
  uint32_t mempool = hdl_to_mempool_.at(hid);
  auto it = used_mempools_[mempool].begin();
  void* address = it->first;
  CHECK_EQ(hdl_dptr_mapping_.erase(it->second), 1);
  it->second = hid;
  hdl_dptr_mapping_[hid] = address;
  return address;
}

void SA_MM_Dptr::StartIteration() {
  sa_log << "StartIteration " << std::endl;
  iteration_started_ = true;
  curr_iteration_ += 1;
  if (curr_iteration_ == 0) {
    // Release all the memory.
    std::unordered_map<handle_t, void*> hdl_dptr_mapping(hdl_dptr_mapping_);
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
    for (handle_t old_hid : initial_handles_) {
      handle_t hid = old_to_new_hids_.at(old_hid);
      void* address = Alloc_(hid, false);
      sa_log << "Initial handles " << hid << " " << old_hid << " "
                << address << std::endl;
    }
  } else {
    //CHECK_EQ(hdl_dptr_mapping_.erase(temp_hdl_), 1);
    temp_hdl_ = 0;
    size_t size_in_memory = 0;
    for (uint32_t i = 0; i < used_mempools_.size(); i++) {
      size_in_memory += used_mempools_[i].size();
    }
    CHECK_EQ(size_in_memory, hdl_dptr_mapping_.size());
    CHECK_EQ(size_in_memory, initial_handles_.size());
  }
  created_handles_ = arg_handles_;
  sa_log << "StartIteration end" << std::endl;
  return;
}

void* SA_MM_Dptr::Alloc(handle_t hid, size_t size, void* ptr=nullptr) {
  sa_log << "SA_MM_Dptr Alloc " << hid << ", size " << size << std::endl;
  //std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  if (alloc_finalized_) {
    // FIXME(fegin)
    // CHECK_EQ(temp_hdl_, 0);
    temp_hdl_ = hid;
    temp_handles_.insert(hid);
    sa_log << "SA_MM_Dptr Alloc temporary memory done " << hid << std::endl;
    return temp_memory_;
  } else {
    auto it = rsize_to_mempool_.find(size);
    if (it == rsize_to_mempool_.end()) {
      CHECK_EQ(size, 4);
      swap_handles_.insert(hid);
      return (void*) 1;
    }
    size_t mempool = rsize_to_mempool_.at(size);
    hdl_size_mapping_[hid] = size;
    hdl_to_mempool_[hid]= mempool;
#if 0
    if (doing_allocargs_) {
      arg_handles_.insert(id);
    }
#endif
    void* address = nullptr;
    if (mempools_.at(mempool).size() == 0) {
      address = FreeAlloc_(hid);
    } else {
      address = Alloc_(hid, false);
    }
    sa_log << "SA_MM_Dptr Alloc done " << address << std::endl;
    return address;
  }
}

void SA_MM_Dptr::Swapin(node_t old_nid, uint32_t idx) {
  node_t nid = old_to_new_nids_.at(old_nid);
  handle_t hid = entry_hdl_mapping_.at(EID(nid, idx));
  sa_log << "About to swapin " << hid << std::endl;
  Alloc_(hid, true);
}

void SA_MM_Dptr::Swapout(node_t old_nid, uint32_t idx) {
  node_t nid = old_to_new_nids_.at(old_nid);
  handle_t hid = entry_hdl_mapping_.at(EID(nid, idx));
  sa_log << "About to swapout " << hid << std::endl;
  Free_(hid, true);
}

void* SA_MM_Dptr::GetDptr(handle_t hid) {
  if (swap_handles_.count(hid) > 0) {
    sa_log << "SA_MM_Dptr GetDptr swap node " << hid << std::endl;
    return (void*)1;
  }
  if (temp_handles_.count(hid) > 0) {
    temp_hdl_ = hid;
    sa_log << "SA_MM_Dptr GetDptr temporary memory " << hid << std::endl;
    return temp_memory_;
  }
#if SWAPADV_REPORT_PROGRESS
  pgr_tracker_.HdlAccess(hid);
#endif
  void* address = nullptr;
  if (sa_likely(alloc_finalized_)) {
    address = GetDptr_(hid);
  } else {
    auto it = hdl_dptr_mapping_.find(hid);
    address = (it != hdl_dptr_mapping_.end()) ? it->second : FreeAlloc_(hid);
  }
  sa_log << "SA_MM_Dptr GetDptr done " << address << std::endl;
  return address;
}

void SA_MM_Dptr::SetDptr(handle_t hid, void* ptr, uint32_t dev_id) {
  if (ptr == nullptr) return;
  sa_log << "SA_MM_Dptr SetDptr " << hid << " " << ptr << std::endl;
  for (auto& used_mempool : used_mempools_) {
    auto it = used_mempool.find(ptr);
    if (it != used_mempool.end()) {
      size_t size = hdl_size_mapping_.at(it->second);
      uint32_t mempool = hdl_to_mempool_.at(it->second);
      hdl_size_mapping_[hid] = size;
      hdl_to_mempool_[hid] = mempool;
      break;
    }
  }
  CHECK_EQ(hdl_size_mapping_.count(hid), 1);
  CHECK_EQ(hdl_to_mempool_.count(hid), 1);
  hdl_dptr_mapping_[hid] = ptr;
}

void SA_MM_Dptr::NotifyDone(node_t nid) {
  sa_log << "NotifyDone " << nid << std::endl;
  if (new_to_old_nids_.count(nid) > 0) {
    auto it = deallocations_.find(nid);
    if (it != deallocations_.end()) {
      sa_log << "Doing deallocation for nid = " << nid << std::endl;
      for (handle_t hid : it->second) {
        Free_(hid, false);
      }
    }
  }
  sa_log << "NotifyDone2 " << nid << std::endl;
#if SWAPADV_REPORT_PROGRESS
  lock_.Lock();
  pgr_tracker_.ReportProgress(new_to_old_nids_, new_to_old_hids_,
                              hdl_to_mempool_, used_mempools_);
  lock_.UnLock();
#endif
}

SA_MM_Dptr* SA_MM_DPTR() {
  return dynamic_cast<SA_MM_Dptr*>(MM_DPTR());
}


}   // namespace storage
}   // namespace mxnet
