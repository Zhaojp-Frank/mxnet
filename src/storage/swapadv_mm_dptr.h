#ifndef MXNET_STORAGE_SWAPADVISOR_MM_DPTR_H_
#define MXNET_STORAGE_SWAPADVISOR_MM_DPTR_H_

#include <atomic>
#include <cuda_runtime.h>
#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <mxnet/sa_util.h>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <vector>
#include <fstream>
#include <unistd.h>

namespace mxnet {
namespace storage {

#define SWAPADV_REPORT_PROGRESS 1

class SpinLock {
 public:
  bool TryLock() {
    return !lock_.test_and_set(std::memory_order_acquire);
  }

  void Lock() {
    while (lock_.test_and_set(std::memory_order_acquire))
      ;  // spin
  }

  void UnLock() {
    lock_.clear(std::memory_order_release);
  }

 private:
  std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
};

class SA_MM_Dptr : virtual public MM_Dptr {
 public:
  SA_MM_Dptr();

  void* Alloc_(handle_id_t id, bool do_swapin);

  void Free_(handle_id_t id, bool do_swapout);

  // Do a random free then alloc. This should only be used during the binding
  // process.
  void* FreeAlloc_(handle_id_t id);

  void* GetDptr_(handle_id_t id);

  void ReportProgress();

  void Swapin(uint32_t nid, uint32_t idx);

  void Swapout(uint32_t nid, uint32_t idx);

  uint32_t EID(uint32_t nid, uint32_t idx) { return nid * 93333 + idx; }

  void* Alloc (handle_id_t id, size_t size, void* ptr) override;

  void* Free(handle_id_t id) override {
    sa_log << "SA_MM_Dptr Free" << std::endl;
    if (id == temp_hdl_) {
      temp_hdl_ = 0;
      CHECK_EQ(hdl_dptr_mapping_.erase(id), 1);
    } else {
      Free_(id, false);
    }
    return nullptr;
  }

  void Release(handle_id_t id, void* ptr) override { CHECK(0); }

  void StartAllocArgs() override { doing_allocargs_ = true; }

  void StopAllocArgs() override { doing_allocargs_ = false; }

  void StartBinding() override { sa_log << "StartBinding " << std::endl; }

  void StopBinding() override {
    sa_log << "StopBinding " << std::endl;
    alloc_finalized_ = true;
    remove("mxnet_model_progress.rst");
    remove("mxnet_swapo_progress.rst");
    remove("mxnet_swapi_progress.rst");
  }

  void StartIteration() override;

  void StopIteration() override {
    sa_log << "StopIteration " << std::endl;
    iteration_started = false;
    return;
  }

  void Statistics () override { }

  void* GetDptr (handle_id_t id) override;

  void SetDptr (handle_id_t id, void* ptr, uint32_t dev_id) override ;

  void RegisterEntry(uint32_t nid, uint32_t idx, handle_id_t hid,
                     uint32_t old_nid, uint32_t old_idx,
                     handle_id_t old_hid, size_t hdl_size,
                     bool is_var) override {
    sa_log << "RegisterEntry, nid = " << nid << " " << old_nid << ", hid = "
           << hid << " " << old_hid << std::endl;
    entry_hdl_mapping_[EID(nid, idx)] = std::make_pair(hid, is_var);
    new_to_old_nids_[nid] = old_nid;
    old_to_new_nids_[old_nid] = nid;
    new_to_old_hids_[hid] = old_hid;
    old_to_new_hids_[old_hid] = hid;
  }

  void FinalizeRegular() override {
    sa_log << "SA_MM_Dptr FinalizeRegular" << std::endl;
    alloc_finalized_ = true;
  }

  void NotifyBegin(uint32_t nid, const std::string& name) override {
#if SWAPADV_REPORT_PROGRESS
    if (new_to_old_nids_.count(nid) > 0) {
      model_tid_ = std::this_thread::get_id();
      model_nid_ = nid;
      model_access_.clear();
    } else if (name.find("swapin") != name.npos) {
      swapin_tid_ = std::this_thread::get_id();
      swapin_nid_ = nid;
    } else if (name.find("swapout") != name.npos) {
      swapout_tid_ = std::this_thread::get_id();
      swapout_nid_ = nid;
    } else {
      CHECK(name.find("swap") != name.npos) << name;
    }
#endif
  }

  void NotifyDone(uint32_t id) override;

  std::vector<uint32_t> GetScheduleDeps(uint32_t nid) override {
    std::vector<uint32_t> ret;
    auto sch_it = schedule_deps_.find(nid);
    if (sch_it == schedule_deps_.end()) return ret;
    for (uint32_t dep_nid : sch_it->second) {
      ret.push_back(dep_nid);
    }
    return ret;
  }

 private:
  // Spinlock
  SpinLock lock_;

  //
  std::unordered_set<handle_id_t> swap_handles_;
  std::unordered_map<handle_id_t, handle_id_t> new_to_old_hids_;
  std::unordered_map<handle_id_t, handle_id_t> old_to_new_hids_;
  std::unordered_map<uint32_t, uint32_t> old_to_new_nids_;
  std::unordered_map<uint32_t, uint32_t> new_to_old_nids_;
  // Handle to dptr mapping. If the result it nulldptr, the handle is swapped
  // out.
  std::unordered_map<handle_id_t, void*> hdl_dptr_mapping_;
  // Handle to size mapping.
  std::unordered_map<handle_id_t, size_t> hdl_size_mapping_;
  // Let we know which device does the handle belong to.
  // Not useful now since we have not supported multiple devices.
  std::unordered_map<handle_id_t, size_t> hdl_dev_mapping_;
  // An entry (NDArray) to handle mapping.
  std::unordered_map<uint32_t, std::pair<handle_id_t, bool>> entry_hdl_mapping_;
  // Initial handles
  std::vector<handle_id_t> initial_handles_;
  // Schedule dependencies
  std::unordered_map<uint32_t, std::vector<uint32_t>> schedule_deps_;
  // Iterations info
  bool iteration_started;
  int curr_iteration;
  // Record what handles are created.
  std::unordered_set<handle_id_t> created_handles_;
  // Record what handles are used by weights;
  std::unordered_set<handle_id_t> arg_handles_;
  //
  bool doing_allocargs_;
  //
  std::unordered_map<uint32_t, std::vector<uint32_t>> deallocations_;
  //
  std::thread::id model_tid_;
  std::thread::id swapin_tid_;
  std::thread::id swapout_tid_;
  uint32_t model_nid_;
  uint32_t swapin_nid_;
  uint32_t swapout_nid_;
  std::vector<handle_id_t> model_access_;

  cudaStream_t stream_out_;
  cudaStream_t stream_in_;
  // Pointer to the CPU memory allocation.
  void *cpu_memory_;
  // Pointer to the main memory allocation.
  void *memory_;
  // The size which the memory manager is allowed to use for the main memory.
  size_t memory_size_;
  // The temporary memory size. This should be dynamically determinited but is
  // predefined for now.
  size_t temp_size_;
  // Pointer to the temoprary memory;
  void *temp_memory_;
  // The size which the memory manager is allowed to use for the temp memory.
  size_t temp_memory_size_;
  // Who is using temp_memory
  handle_id_t temp_hdl_;
  // Memory pool index to memory pool size.
  std::vector<size_t> mempool_to_size_;
  // Memory pool objects count
  std::vector<size_t> mempool_counts_;
  // The mapping from real sizes to memory pool index.
  std::unordered_map<size_t, size_t> rsize_to_mempool_;
  // The mapping from hdl to memory pool index.
  std::unordered_map<handle_id_t, size_t> hdl_to_mempool_;
  // Memory pools
  std::vector<std::vector<void*>> mempools_;
  // Used memory pools
  std::vector<std::unordered_map<void*, uint32_t>> used_mempools_;
  // Used memory
  size_t used_memory_;
  // Are all regular memory allocations finalzed.
  bool alloc_finalized_;

  // Read the scedule result.
  void ReadScheduleDepsRst();
  // Read the memory allocation result from the SwapAdvisor.
  void ReadAllocationRst();
  // Read the initial handle allocation result from the SwapAdvisor.
  void ReadInitialHandlesRst();
  // Read the deallocation result from the SwapAdvisor.
  void ReadDeallocationRst();
};

SA_MM_Dptr* SA_MM_DPTR();

}  // namespace storage
}  // namespace mxnet
#endif
