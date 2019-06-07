#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include "../common/cuda_utils.h"
#include "./gpu_odswap.h"

//#define FEGIN_DEBUG

namespace mxnet{

SwapInfoGroups::SwapInfoGroups() { }

SwapInfoGroups::~SwapInfoGroups() { }

std::shared_ptr<SwapInfoGroups> SwapInfoGroups::_GetSharedRef() {
  static std::shared_ptr<SwapInfoGroups> inst(new SwapInfoGroups());
  return inst;
}

SwapInfoGroups* SwapInfoGroups::Get() {
  static SwapInfoGroups *sig = _GetSharedRef().get();
  return sig;
}

void SwapInfoGroups::NewInfo(void* addr, SwapInfo* info) {
  // We don't check the duplication here. It's caller's resposibility to make
  // sure this API is called when a new SwapInfo is created.
  auto it = in_memory_.find(addr);
  if (it == in_memory_.end()) {
    std::shared_ptr<SIGroup> group(new SIGroup());
    in_memory_[addr] = group;
    all_[info] = group;
  } else {
    std::cout << "Existed " << std::endl;
    it->second->push_back(info);
    all_[info] = it->second;
  }
}

std::shared_ptr<SIGroup> SwapInfoGroups::SwapOut(void* addr) {
  auto ptr = in_memory_.at(addr);
  in_memory_.erase(addr);
  for (auto& info : *ptr) {
    info->swapped_in = false;
    info->dptr = nullptr;
  }
  return ptr;
}

std::shared_ptr<SIGroup> SwapInfoGroups::SwapIn(void* addr, SwapInfo* info) {
  auto ptr = all_.at(info);
  for (auto& info : *ptr) {
    info->swapped_in = true;
    info->dptr = addr;
  }
  in_memory_[addr] = ptr;
  return ptr;
}

ThreadAccessInfo::ThreadAccessInfo() {
  running_threshold_ = dmlc::GetEnv("MXNET_SWAP_ACCESS_THRESHOLD",
                                    kAccessThreshold);
}

ThreadAccessInfo::~ThreadAccessInfo() {}

std::shared_ptr<ThreadAccessInfo> ThreadAccessInfo::_GetSharedRef() {
  static std::shared_ptr<ThreadAccessInfo> inst(new ThreadAccessInfo());
  return inst;
}

ThreadAccessInfo* ThreadAccessInfo::Get() {
  static ThreadAccessInfo *tai = _GetSharedRef().get();
  return tai;
}

std::set<handle_t>& ThreadAccessInfo::CheckAndCreate(handle_t hid, bool access,
                                                     bool& running) {
  std::thread::id tid = std::this_thread::get_id();
  auto pair = all_threads_.emplace(tid, std::set<handle_t>{});
  if (pair.second) {
    is_running[tid] = false;
  }
  std::set<handle_t>& handles = pair.first->second;
  if (access) {
    running = is_running[tid];
    handles.insert(hid);
    auto hpair = hid_to_threads_.emplace(hid,
                                         std::unordered_set<std::thread::id>{});
    hpair.first->second.insert(tid);
  } else {
    is_running[tid] = running;
    for (auto it = handles.begin(); it != handles.end(); ) {
      std::unordered_set<std::thread::id>& hid_to_threads = hid_to_threads_[*it];
      hid_to_threads.erase(tid);
#ifdef FEGIN_DEBUG
      sa_log << "Hid=" << *it << ",size=" << hid_to_threads.size() << ",";
      if (hid_to_threads.begin() != hid_to_threads.end()) {
        std::cout << *(hid_to_threads.begin()) << std::endl;
      } else {
        std::cout << std::endl;
      }
#endif
      if (hid_to_threads.size() > 0) {
        it = handles.erase(it);
      } else {
        it++;
      }
    }
  }
#ifdef FEGIN_DEBUG
  sa_log << "TID:" << tid << ",size=" << handles.size();
  for (auto id : handles) {
    std::cout << "," << id;
  }
  std::cout << std::endl;
#endif
  return handles;
}

handle_t ThreadAccessInfo::Access(handle_t hid) {
  bool running = true;
  std::set<handle_t>& handles = CheckAndCreate(hid, true, running);
  if (!running && handles.size() > kAccessThreshold) {
    handle_t ready_id = *(handles.begin());
    handles.erase(handles.begin());
    hid_to_threads_[ready_id].erase(std::this_thread::get_id());
    if (hid_to_threads_[ready_id].size() == 0) {
      return ready_id;
    }
  }
  return kInvalidID;
}

void ThreadAccessInfo::Remove(handle_t hid) {
  std::unordered_set<std::thread::id>& hid_to_threads = hid_to_threads_[hid];
  for (auto tid : hid_to_threads) {
    all_threads_[tid].erase(hid);
  }
  hid_to_threads.clear();
}

std::shared_ptr<ODSwap> ODSwap::_GetSharedRef() {
  static std::shared_ptr<ODSwap> inst(new ODSwap());
  return inst;
}

ODSwap* ODSwap::Get() {
  static ODSwap *s = _GetSharedRef().get();
  return s;
}

ODSwap::ODSwap() {
  std::cout << "Initialize ODSwap" << std::endl;
  swap_lock_ = PTHREAD_RWLOCK_INITIALIZER;
  swap_async_ = dmlc::GetEnv("MXNET_SWAP_ASYNC", true);
  std::cout << "SWAP_ASYNC=" << swap_async_ << std::endl;
  infinite_memory_ = dmlc::GetEnv("MXNET_INFINITE_MEMORY", false);
  infinite_cpu_memory_ = dmlc::GetEnv("MXNET_INFINITE_CPU_MEMORY", false);
  if (infinite_cpu_memory_) {
    const size_t fake_cpu_size = 20L*1024*1024*1024;
    cudaHostAlloc((void**)&(fake_cpu_address_), fake_cpu_size, 0);
    CHECK(fake_cpu_address_ != nullptr)
      << "Fake cpu memory allocation failed" << std::endl;
    std::cout << "Initialize fake cpu memory of size: "
              << fake_cpu_size << "B" << std::endl;
  }
  for (int i = 0; i < NUMBER_OF_GPU; ++i) {
    locks_[i] = PTHREAD_RWLOCK_INITIALIZER;
    // TODO(fegin): This and other cuda related code should be protected
    //              by MXNET_USE_CUDA.
    cudaStreamCreate(&streams_out_[i]);
    cudaStreamCreate(&streams_in_[i]);
  }
  swapinfo_groups_ = SwapInfoGroups::_GetSharedRef();
  thread_info_ = ThreadAccessInfo::_GetSharedRef();
  memory_history_ = MemoryHistory::_GetSharedRef();
  memory_manager_ = GetMemoryManagerRef();
  std::cout << "Initialize ODSwap done" << std::endl;
}

ODSwap::~ODSwap() {
  std::cout << "Destroy ODSwap" << std::endl;
  cudaFreeHost(fake_cpu_address_);
}

void ODSwap::SwapOutLocked(unsigned required_memory, int device_id, bool async) {
  pthread_rwlock_wrlock(&swap_lock_);
  SwapOut(required_memory, device_id, async);
  pthread_rwlock_unlock(&swap_lock_);
}

std::set<handle_t>& ThreadAccessInfo::GetProtectedHandles() {
  std::thread::id tid = std::this_thread::get_id();
  auto pair = all_threads_.emplace(tid, std::set<handle_t>{});
  std::set<handle_t>& handles = pair.first->second;
  return handles;
}

// Caller holds swap_lock_
void ODSwap::SwapOut(unsigned required_memory, int device_id, bool async) {
  while (!memory_manager_->TryAllocate(device_id, required_memory)) {
    SwapParams param = {0, required_memory, &divided_handles_[device_id]};
#ifdef FEGIN_DEBUG
    sa_log << "Swapout calling Decide victim. Swappable size = "
           << swappable_handles_[device_id].size() << std::endl;
#endif
    if(swappable_handles_[device_id].size() <= 0) {
      std::cout << "Handle exhausted!" << std::endl;

      std::cout << "Handles kept by PrePostAccess:" << std::endl;
      size_t kept_total = 0;
      auto handles = thread_info_->GetProtectedHandles();
      for (auto& handle: handles) {
        CHECK(swap_info_.find(handle) != swap_info_.end());
        auto& info = swap_info_.at(handle);
        kept_total += info->size;
        std::cout << handle << " size: " << info->size << " is_in: "
               << info->swapped_in << std::endl;
      }
      std::cout << "Total Size of kept handles: " << kept_total << std::endl;

      std::cout << "Handles that are swapped in:" << std::endl;
      size_t swap_total = 0;
      for (auto& tmp: swap_info_) {
        auto& info = tmp.second;
        if(info->swapped_in) {
          swap_total += info->size;
          std::cout << tmp.first << " size: " << info->size 
                 << " Address: " << info->dptr
                 << " Swap counts: " << info->swap_count << std::endl;
        }
      }
      std::cout << "Total Size of swapped in handles: " << swap_total << std::endl;

      std::cout << "Handles that are swapped out:" << std::endl;
      swap_total = 0;
      for (auto& tmp: swap_info_) {
        auto& info = tmp.second;
        if(!info->swapped_in) {
          swap_total += info->size;
          std::cout << tmp.first << " size: " << info->size 
                 << " Address: " << info->cpu_address
                 << " Swap counts: " << info->swap_count << std::endl;
        }
      }
      std::cout << "Total Size of swapped out handles: " << swap_total << std::endl;
      std::cout << "Sum of kept and swapped out handles: "
             << swap_total + kept_total << std::endl;

      std::cout << "Memory Info:" << std::endl;
      size_t total, avail;
      memory_manager_->MemGetInfo(0, &total, &avail);
      std::cout << "Total: " << total << " Avail: " << avail << std::endl;
    }
    CHECK(swappable_handles_[device_id].size() > 0)
      << "Set of swappable handles is exhausted" << std::endl;
    handle_t victim =
      memory_history_->DecideVictim(swappable_handles_[device_id], device_id,
                                    &param);
    CHECK(swap_info_.find(victim) != swap_info_.end())
      << "Victim(" << victim << ") does not exist (deleted?) " << std::endl;
    SwapInfo *target = swap_info_[victim];
#ifdef FEGIN_DEBUG
    sa_log << "SwapOut " << victim << " " << target->size << " "
           << target->swap_count << std::endl;
#endif
    target->swap_count++;
    memory_history_->DevHistory(device_id).num_swap_out++;
    memory_history_->DevHistory(device_id).swap_out_total += target->size;
    if (!infinite_memory_) {
      if (target->cpu_address == nullptr) {
        if (infinite_cpu_memory_) {
          target->cpu_address = fake_cpu_address_; 
        }
        else {
          cudaHostAlloc((void**)&(target->cpu_address), target->size, 0);
        }
      }
      CHECK(target->cpu_address != nullptr);
    }
    CHECK(target->swapped_in);
    CHECK(!target->is_swapping.test_and_set(std::memory_order_acquire));
    CHECK(target->dptr != nullptr);
    target->swapped_in = false;
    //swapinfo_groups_->SwapOut(target->dptr);
    swappable_handles_[device_id].erase(victim);
    divided_handles_[device_id][target->size].erase(victim);
#ifdef FEGIN_DEBUG
    sa_log << "SwapOut: Swapping out, remove(1) swappable handle_id = "
           << victim << std::endl;
#endif
    thread_info_->Remove(victim);
    pthread_rwlock_unlock(&swap_lock_);
    if (!infinite_memory_) {
#if 1
      if (async) {
        memory_manager_->MemcpyAsync(device_id, target->cpu_address,
            target->dptr, target->size, cudaMemcpyDeviceToHost,
            streams_out_[device_id]);
        memory_manager_->StreamSynchronize(device_id, streams_out_[device_id]);
      } else {
        memory_manager_->Memcpy(device_id, target->cpu_address, target->dptr,
            target->size, cudaMemcpyDeviceToHost);
      }
#endif
    }
    memory_manager_->Free(target->dptr, device_id);
    pthread_rwlock_wrlock(&swap_lock_);
    target->is_swapping.clear(std::memory_order_release);
#ifdef FEGIN_DEBUG
    sa_log << "SwapOut: Finish swapping out handle: " << victim << std::endl; 
#endif
  }
}

// Caller holds swap_lock_
void ODSwap::SwapIn(SwapInfo *info, bool async) {
  while (info->is_swapping.test_and_set(std::memory_order_acquire)) {
    // TODO(fegin): usleep may not be efficient and may cause unstable
    //              execution. This is not important for now but can be
    //              something to imporve in the future. However, this
    //              must be designed carefully. One mutex per handle is not
    //              acceptable unlike current atmoic variable design.
    pthread_rwlock_unlock(&swap_lock_);
    usleep(10);
    pthread_rwlock_wrlock(&swap_lock_);
  }
  if (!info->swapped_in) {
    CHECK(!info->swapped_in);
    CHECK(info->cpu_address != nullptr || infinite_memory_);
#ifdef FEGIN_DEBUG
    sa_log << "SwapIn "<< info->handle_id << " " << info->size << " "
           << info->swap_count << std::endl;
#endif
    SwapOut(info->size, info->device_id, async);
    CHECK(memory_manager_->Malloc(info->dptr, info->size, info->device_id) ==
          cudaSuccess);
    pthread_rwlock_unlock(&swap_lock_);
    memory_history_->DevHistory(info->device_id).num_swap_in++;
    memory_history_->DevHistory(info->device_id).swap_in_total += info->size;
    if (!infinite_memory_) {
#if 1
      if (async) {
        memory_manager_->MemcpyAsync(info->device_id, info->dptr,
            info->cpu_address, info->size, cudaMemcpyHostToDevice,
            streams_in_[info->device_id]);
        memory_manager_->StreamSynchronize(info->device_id,
              streams_in_[info->device_id]);
      } else {
        memory_manager_->Memcpy(info->device_id, info->dptr, info->cpu_address,
                                info->size, cudaMemcpyHostToDevice);
      }
#endif
      // delete info->cpu_address;
    }
    // info->cpu_address = nullptr;
    pthread_rwlock_wrlock(&swap_lock_);
    info->swapped_in = true;
    //swapinfo_groups_->SwapIn(info->dptr, info);
    swappable_handles_[info->device_id].insert(info->handle_id);
    divided_handles_[info->device_id][info->size].insert(info->handle_id);
#ifdef FEGIN_DEBUG
    sa_log << "Insert(1) swappable handle_id = "
           << info->handle_id << std::endl;
#endif
  }
  info->is_swapping.clear(std::memory_order_release);
}

void ODSwap::SetAddr(handle_t handle_id, void* dptr, size_t size,
                     int device_id, bool is_pre) {
  if (device_id != -1 && is_pre) {
    memory_history_->PutRecord(handle_id, device_id, MemoryHistory::SET_ADDR,
                               size);
  }
#ifdef FEGIN_DEBUG
  sa_log << "SetAddr=" << handle_id << ", size=" << size <<  std::endl;
#endif
  if (dptr == nullptr) {
    return;
  }
  pthread_rwlock_wrlock(&swap_lock_);
  auto iter = swap_info_.find(handle_id);
  if (is_pre) {
    CHECK(iter == swap_info_.end());
    SwapInfo* info = new SwapInfo{handle_id, true, device_id,
      dptr, nullptr, size, 0, ATOMIC_FLAG_INIT};
    swap_info_[handle_id] = info;
    //swapinfo_groups_->NewInfo(dptr, info);
  } else {
    CHECK(iter != swap_info_.end());
    iter->second->dptr = dptr;
    handle_t ready_id = thread_info_->Access(handle_id);
    if (ready_id != thread_info_->kInvalidID) {
      auto ready_iter = swap_info_.find(ready_id);
      int device_id = ready_iter->second->device_id;
      swappable_handles_[device_id].insert(ready_id);
      divided_handles_[device_id][ready_iter->second->size].insert(ready_id);
#ifdef FEGIN_DEBUG
      sa_log << "Insert(2) swappable handle_id = " << ready_id
             << std::endl;
#endif
    }
  }
  pthread_rwlock_unlock(&swap_lock_);
  #ifdef FEGIN_DEBUG
      sa_log << "SetAddr " << handle_id << " Returning"<< std::endl;
  #endif
}

void ODSwap::FreeAddr(handle_t handle_id) {
  pthread_rwlock_wrlock(&swap_lock_);
  //std::cout << "FreeAddr " << handle_id << std::endl;
  auto info = swap_info_.at(handle_id);
  if (info->device_id != -1) {
    memory_history_->PutRecord(handle_id, info->device_id,
                               MemoryHistory::DEL_ADDR, info->size);
    if (swappable_handles_[info->device_id].find(handle_id)
        != swappable_handles_[info->device_id].end()) {
      swappable_handles_[info->device_id].erase(handle_id);
      thread_info_->Remove(handle_id);
    }
    if (divided_handles_[info->device_id][info->size].find(handle_id)
        != divided_handles_[info->device_id][info->size].end()) {
      divided_handles_[info->device_id][info->size].erase(handle_id);
    }
  }
  size_t free, total;
  memory_manager_->MemGetInfo(info->device_id, &total, &free);
  if (info->swapped_in) {
    memory_manager_->Free(info->dptr, info->device_id);
  }
  if (info->cpu_address != nullptr) {
    //delete info->cpu_address;
    cudaFreeHost(info->cpu_address);
  }
  delete info;
  swap_info_.erase(handle_id);
  pthread_rwlock_unlock(&swap_lock_);
}

void ODSwap::DelAddr(handle_t handle_id) {
  pthread_rwlock_wrlock(&swap_lock_);
  //std::cout << "DelAddr " << handle_id << std::endl;
  auto info = swap_info_.at(handle_id);
  if (info->device_id != -1) {
    memory_history_->PutRecord(handle_id, info->device_id,
                               MemoryHistory::DEL_ADDR, info->size);
    if (swappable_handles_[info->device_id].find(handle_id)
        != swappable_handles_[info->device_id].end()) {
      swappable_handles_[info->device_id].erase(handle_id);
      thread_info_->Remove(handle_id);
#ifdef FEGIN_DEBUG
      sa_log << "Remove(2) swappable handle_id = " << handle_id
             << std::endl;
#endif
    }
    if (divided_handles_[info->device_id][info->size].find(handle_id)
        != divided_handles_[info->device_id][info->size].end()) {
      divided_handles_[info->device_id][info->size].erase(handle_id);
    }
  }
  if (info->cpu_address != nullptr) {
    //delete info->cpu_address;
    if (!infinite_cpu_memory_) {
      cudaFreeHost(info->cpu_address);
    }
  }
  delete info;
  swap_info_.erase(handle_id);
  pthread_rwlock_unlock(&swap_lock_);
}

void* ODSwap::GetAddr(handle_t handle_id, bool prefetch) {
#ifdef FEGIN_DEBUG
  sa_log << "GetAddr: "<< handle_id << std::endl;
  sa_log << "Memory Info:" << std::endl;
  size_t total, avail;
  memory_manager_->MemGetInfo(0, &total, &avail);
  sa_log << "Total: " << total << " Avail: " << avail << std::endl;
#endif
  pthread_rwlock_wrlock(&swap_lock_);
  auto info = swap_info_.at(handle_id);
  if (info->device_id != -1 && !prefetch) {
    memory_history_->DevHistory(info->device_id).num_get_addr++;
    memory_history_->PutRecord(handle_id, info->device_id,
                               MemoryHistory::GET_ADDR, info->size);
  }
#ifdef FEGIN_DEBUG
  sa_log << "GetAddr info size = " << info->size << std::endl;
#endif
  if (!info->swapped_in) {
    if (!prefetch) {
      ++(memory_history_->DevHistory(info->device_id).cache_miss);
    }
    SwapIn(info, swap_async_);
  }
  CHECK(info->swapped_in) << "Info is not swapped in after SwapIn" << std::endl;
  swappable_handles_[info->device_id].erase(handle_id);
  divided_handles_[info->device_id][info->size].erase(handle_id);
#ifdef FEGIN_DEBUG
  sa_log << "Remove(3) swappable handle_id = " << handle_id << std::endl;
#endif

  if (!prefetch) {
    // Don't let the pretch thread interfere the access information.
    handle_t ready_id = thread_info_->Access(handle_id);
    if (ready_id != thread_info_->kInvalidID) {
      auto ready_iter = swap_info_.find(ready_id);
      int device_id = ready_iter->second->device_id;
      swappable_handles_[device_id].insert(ready_id);
      divided_handles_[device_id][ready_iter->second->size].insert(ready_id);
#ifdef FEGIN_DEBUG
      sa_log << "Insert(3) swappable handle_id = " << ready_id
             << std::endl;
#endif
    }
  }
  pthread_rwlock_unlock(&swap_lock_);
  return info->dptr;
}

void ODSwap::PrePostAccess(bool is_pre) {
  // FIXME(fegin): If multiple thread access the same record,
  // how are we going to handle this?
#ifdef FEGIN_DEBUG
  if (is_pre) {
    sa_log << "PreAccess " << std::endl;
  } else {
    sa_log << "PostAccess " << std::endl;
  }
#endif
  pthread_rwlock_wrlock(&swap_lock_);
  std::set<handle_t>& handles =
    thread_info_->CheckAndCreate(0, false, is_pre);
  for (auto id : handles) {
    auto it = swap_info_.find(id);
    swappable_handles_[0].insert(id);
    divided_handles_[it->second->device_id][it->second->size].insert(id);
#ifdef FEGIN_DEBUG
    if (is_pre) {
      sa_log << "Insert(Pre) swappable handle_id = " << id << std::endl;
    } else {
      sa_log << "Insert(Post) swappable handle_id = " << id << std::endl;
    }
#endif
  }
  handles.clear();
  pthread_rwlock_unlock(&swap_lock_);
}

void ODSwap::PrintHandles() {
  std::cout << "Print Handles" << std::endl;
  //std::map<size_t, std::unordered_set<handle_t> > _divided_handles_;
  for (auto it : swap_info_) {
    //_divided_handles_[it.second->size].insert(it.first);
    std::cout << it.first << ": " << it.second->size << " "
              << it.second->swap_count << " " << it.second->device_id
              << std::endl;
  }
}

} // namespace mxnet
