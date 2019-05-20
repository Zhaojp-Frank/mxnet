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

using NidMapping = std::unordered_map<node_t, node_t>;
using HidSizeMapping = std::unordered_map<node_t, size_t>;
using HidMapping = std::unordered_map<handle_t, handle_t>;

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

class ProgressTracker {
 public:
  void Begin(const NidMapping& new_to_old_nids, const node_t nid,
             const std::string& name) {
    if (new_to_old_nids.count(nid) > 0) {
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
  }

  void HdlAccess(const handle_t hid) {
    if (std::this_thread::get_id() != model_tid_) return;
    if (std::find(model_access_.begin(), model_access_.end(), hid) !=
        model_access_.end()) return;
    model_access_.push_back(hid);
  }

  void ReportProgress(
      const NidMapping& new_to_old_nids,
      const HidMapping& new_to_old_hids,
      const HidSizeMapping& hdl_to_mempool_,
      const std::vector<std::unordered_map<void*, node_t>>& used_mempools_);

 private:
  std::thread::id model_tid_;
  std::thread::id swapin_tid_;
  std::thread::id swapout_tid_;
  node_t model_nid_;
  node_t swapin_nid_;
  node_t swapout_nid_;
  std::vector<handle_t> model_access_;
};

class SA_MM_Dptr : virtual public MM_Dptr {
 public:
  SA_MM_Dptr();

  void* Alloc_(handle_t hid, bool do_swapin);

  void Free_(handle_t hid, bool do_swapout);

  // Do a random free then alloc. This should only be used during the binding
  // process.
  void* FreeAlloc_(handle_t hid);

  void* GetDptr_(handle_t hid);

  void ReportProgress();

  void Swapin(node_t nid, uint32_t idx);

  void Swapout(node_t nid, uint32_t idx);

  void Remap();

  static long EID(node_t nid, uint32_t idx) { return nid * 93333L + idx; }

  void* Alloc(handle_t hid, size_t size, void* ptr) override;

  void* Free(handle_t id) override {
    sa_log << "SA_MM_Dptr Free" << std::endl;
    if (id == temp_hdl_) {
      temp_hdl_ = 0;
      //CHECK_EQ(hdl_dptr_mapping_.erase(id), 1);
    } else {
      Free_(id, false);
    }
    return nullptr;
  }

  void Release(handle_t id, void* ptr) override { CHECK(0); }

  void StartAllocArgs() override { doing_allocargs_ = true; }

  void StopAllocArgs() override { doing_allocargs_ = false; }

  void StartBinding() override { sa_log << "StartBinding " << std::endl; }

  void StopBinding() override {
    sa_log << "StopBinding " << std::endl;
    Remap();
    alloc_finalized_ = true;
#if SWAPADV_REPORT_PROGRESS
    remove("mxnet_model_progress.rst");
    remove("mxnet_swapo_progress.rst");
    remove("mxnet_swapi_progress.rst");
#endif
  }

  void StartIteration() override;

  void StopIteration() override {
    sa_log << "StopIteration " << std::endl;
    iteration_started_ = false;
    return;
  }

  void Statistics() override { }

  void* GetDptr(handle_t id) override;

  void SetDptr(handle_t id, void* ptr, uint32_t dev_id) override ;

  void RegisterEntry(node_t nid, uint32_t idx, handle_t hid, node_t old_nid,
                     uint32_t old_idx, handle_t old_hid, size_t hdl_size,
                     bool is_var, bool is_swap) override {
    sa_log << "RegisterEntry, nid = " << nid << " " << old_nid << ", hid = "
           << hid << " " << old_hid << ", is_var = " << is_var << std::endl;
    if (is_swap) {
      swap_handles_.insert(hid);
      hdl_size_mapping_.erase(hid);
      hdl_to_mempool_.erase(hid);
    } else {
      if (is_var) {
        arg_handles_.insert(hid);
      }
      entry_hdl_mapping_[EID(nid, idx)] = hid;
      new_to_old_nids_[nid] = old_nid;
      old_to_new_nids_[old_nid] = nid;
      new_to_old_hids_[hid] = old_hid;
      old_to_new_hids_[old_hid] = hid;
    }
  }

  void NotifyBegin(const node_t nid, const std::string& name) override {
#if SWAPADV_REPORT_PROGRESS
    pgr_tracker_.Begin(new_to_old_nids_, nid, name);
#endif
  }

  void NotifyDone(node_t id) override;

  void Finish() override { is_finished_ = true; }

#if 0
  std::vector<uint32_t> GetScheduleDeps(uint32_t nid) override {
    std::vector<uint32_t> ret;
    auto sch_it = schedule_deps_.find(nid);
    if (sch_it == schedule_deps_.end()) return ret;
    for (uint32_t dep_nid : sch_it->second) {
      ret.push_back(dep_nid);
    }
    return ret;
  }
#endif

 private:
  SpinLock lock_;

  // Handles which are used to store temporary memory.
  std::unordered_set<handle_t> temp_handles_;
  // Ouput handles which are generated by swap nodes.
  std::unordered_set<handle_t> swap_handles_;
  // Record what handles are created.
  std::unordered_set<handle_t> created_handles_;
  // Record what handles are used by weights;
  std::unordered_set<handle_t> arg_handles_;

  HidMapping new_to_old_hids_;
  HidMapping old_to_new_hids_;
  NidMapping old_to_new_nids_;
  NidMapping new_to_old_nids_;
  // Handle to dptr mapping. If the result it nulldptr, the handle is swapped
  // out.
  std::unordered_map<handle_t, void*> hdl_dptr_mapping_;
  // Handle to size mapping.
  HidSizeMapping hdl_size_mapping_;
  // An entry (NDArray) to handle mapping.
  std::unordered_map<long, handle_t> entry_hdl_mapping_;
  // Initial handles
  std::vector<handle_t> initial_handles_;
  // Schedule dependencies
  //std::unordered_map<node_t, std::vector<node_t>> schedule_deps_;
  // Iterations info
  bool iteration_started_;
  int curr_iteration_;
  bool doing_allocargs_;
  std::unordered_map<node_t, std::vector<handle_t>> deallocations_;
#if SWAPADV_REPORT_PROGRESS
  ProgressTracker pgr_tracker_;
#endif

  // Swapout and swapin CUDA streams.
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
  handle_t temp_hdl_;
  // Memory pool index to memory pool size.
  std::vector<size_t> mempool_to_size_;
  // Memory pool objects count
  std::vector<size_t> mempool_counts_;
  // The mapping from real sizes to memory pool index.
  std::unordered_map<size_t, size_t> rsize_to_mempool_;
  // The mapping from hdl to memory pool index.
  HidSizeMapping hdl_to_mempool_;
  // Memory pools
  std::vector<std::vector<void*>> mempools_;
  // Used memory pools
  std::vector<std::unordered_map<void*, node_t>> used_mempools_;
  // Are all regular memory allocations finalzed.
  bool alloc_finalized_;
  // Inidicate that the whole execution is finished.
  bool is_finished_;

  // Read the scedule result.
  //void ReadScheduleDepsRst();
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
