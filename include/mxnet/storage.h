/*!
 * Copyright (c) 2015 by Contributors
 * \file storage.h
 * \brief Storage manager across multiple devices.
 */
#ifndef MXNET_STORAGE_H_
#define MXNET_STORAGE_H_

#include <atomic>
#include <chrono>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <unistd.h>
#include <memory>
#if MXNET_USE_CUDA
#include <cuda_runtime.h>
#endif  // MXNET_USE_CUDA
#include "./base.h"

using namespace std::chrono;

namespace mxnet {
using handle_id_t = unsigned long long;
using timestamp_t = unsigned long long;

struct SwapInfo {
    handle_id_t handle_id;
    bool swap_in;
    std::atomic_flag is_swapping;
    int swap_count;
    int device;
    void* dptr;
    char* cpu_address;
    size_t size;
    std::list<SwapInfo*>::iterator it;
};

class MemHistory {
public:
    enum record_t {SET_ADDR, GET_ADDR, DEL_ADDR};
    struct MemRecord {
        handle_id_t handle_id;
        record_t type;
        timestamp_t time;
        size_t size;
    };
    struct MemRecordSet {
        std::vector<MemRecord> unique_records;
        std::vector<MemRecord> all_records;
    };

    ~MemHistory();
    static MemHistory* Get();
    static std::shared_ptr<MemHistory> _GetSharedRef();
    void PutRecord(handle_id_t handle_id, int device, record_t type,
                   size_t size);
    void StartIteration();
    void StopIteration();
    void MakeRecordSet(std::unordered_map<handle_id_t, MemRecord>& members,
                       MemRecordSet *record_set, int device);
    void Analyze();
    bool IterationStarted() { return iteration_started_; };
    bool HistoryRecorded() { return history[0].size() != 0 && !do_record_; };
    bool IsRecording() { return do_record_; };
    int GetNumDevice() { return num_device_; };

    std::vector<MemRecord> history[8];
    std::vector<MemRecordSet*> set_history[8];
    std::vector<int> set_accu_nrecords[8];
    std::unordered_map<handle_id_t, int> access_stats;
    std::atomic<int>curr_idx[8];

private:
    MemHistory();
    bool iteration_started_;
    bool do_record_;
    size_t iteration_idx_;
    high_resolution_clock::time_point begin_time_;
    std::mutex mutex_[8];
    int num_device_;
};

class Cache {
public:
    struct CacheItem {
        void *dptr;
        size_t size;
        bool cached;
        std::atomic_flag loading;
        std::unordered_set<handle_id_t> all_handles;
    };
    ~Cache();
    static std::shared_ptr<Cache> _GetSharedRef();
    static Cache* Get();
    void StartIteration();
    void StopIteration();
    void Register(int tensor_id, TShape offset, void* output_dptr);
    bool Cached(void* dptr);
    void Release(void* dptr);
    bool CacheIt(SwapInfo *info);
    void SetAddr(SwapInfo *info, bool record);
    void GetAddr(SwapInfo *info, bool record);
    void DelAddr(SwapInfo *info, bool record);

private:
    Cache();
    void Loader(int device);
    bool enabled_;
    bool ready_;
    bool should_stop_;
    int processed_cache_idx_;
    int num_device_;
    size_t cache_threshold_;
    std::shared_ptr<MemHistory> mhistory_;
    std::list<CacheItem*> swap_queues_[8];
    std::thread swapper_[8];
    std::unordered_map<unsigned long long, CacheItem*> key_to_cache_[8];
    std::unordered_map<handle_id_t, CacheItem*> cache_[8];
    std::unordered_map<handle_id_t, SwapInfo*> temporary_items_[8];
    std::unordered_map<void*, handle_id_t> dptr_to_handle_[8];
    pthread_rwlock_t locks_[8];
    pthread_rwlock_t swap_locks_[8];
    cudaStream_t streams_[8];
    bool streams_init_;
};

class Swap {
public:
    ~Swap();
    static Swap* Get();
    static std::shared_ptr<Swap> _GetSharedRef();
    void DoSwap(SwapInfo* info, bool swap_out, bool async);
    void SwapOut(unsigned required_memory, int device, bool acquire_lock,
                 bool async);
    void SwapIn(SwapInfo *info, bool async);
    int UpdateFree(int device);
    void SetAddr(handle_id_t handle_id, void* dptr, size_t size, int dev_id,
                 bool record=true);
    void SetAddr_Cache(SwapInfo *info, bool record);
    void SetAddr_Swap(SwapInfo *info, bool record);
    void DelAddr(handle_id_t handle_id, size_t size, bool preserve,
                 bool record=true);
    void DelAddr_Swap(SwapInfo *swap_info, bool preserve, bool record);
    void DelAddr_Cache(SwapInfo *swap_info, bool preserve, bool record);
    void* GetAddr(handle_id_t handle_id, size_t size,
                  bool record=true);
    void GetAddr_Cache(SwapInfo *info, bool record);
    void GetAddr_Swap(SwapInfo *info, bool record);
    void AllocateReserved(size_t required, int device);
    bool FreeReserved(void *ptr, size_t size);
    bool CheckReservedAndFree(void *ptr, size_t size);
    void StartIteration();
    void StopIteration();

private:
    Swap();
    void Swapper(int device);
    void SwapperLookahead(int device, int& curr_pos);
    void SwapperSetLookahead(int device, int& curr_pos);
    std::shared_ptr<MemHistory> mhistory_;
    std::shared_ptr<Cache> cache_;
    std::unordered_map<handle_id_t, SwapInfo*> swap_info_;
    std::vector<std::unordered_map<void*, size_t>> reserved_mem_;
    std::vector<std::list<SwapInfo*>> lru_;
    std::vector<size_t> free_memory_;
    pthread_rwlock_t swap_lock_;
    pthread_rwlock_t locks_[8];
    cudaStream_t streams_[8];
    bool streams_init_[8];
    bool do_swap_;
    bool do_cache_;
    bool free_cpu_;
    size_t swap_threshold_;
    bool should_stop_;
    bool swapper_began_;
    std::thread swapper_[8];
    int look_ahead_;
    int swapper_select_;
    size_t cache_miss_;
    size_t waiting_swapping_;
    int num_device_;
    std::unordered_map<handle_id_t, int> access_stats_;
};


/*!
 * \brief Storage manager across multiple devices.
 */
class Storage {
 public:
  /*!
   * \brief Storage handle.
   */

  struct Handle {
    Handle() {
        id_ = (base_id_.fetch_add(1, std::memory_order_relaxed)) + 1;
    }

    void Free(bool preserve) {
        Swap::Get()->DelAddr(id_, size, preserve);
    }

    /*!
     * \brief Pointer to the data.
     */
    void SetDptr(void* ptr, int dev_id) {
        Swap::Get()->SetAddr(id_, ptr, size, dev_id);
        dptr_ = ptr;
    }
    void* GetDptr() {
        return Swap::Get()->GetAddr(id_, size);
    }
    /*!
     * \brief Size of the storage.
     */
    size_t size;
    /*!
     * \brief Context information about device and ID.
     */
    Context ctx;

   private:
    static std::atomic<handle_id_t> base_id_;
    void* dptr_;
    handle_id_t id_;
  };
  /*!
   * \brief Allocate a new contiguous memory for a given size.
   * \param size Total size of memory in bytes.
   * \param ctx Context information about the device and ID.
   * \return Handle struct.
   */
  virtual Handle Alloc(size_t size, Context ctx, bool direct = false) = 0;
  /*!
   * \brief Free storage.
   * \param handle Handle struect.
   */
  virtual void Free(Handle handle) = 0;
  /*!
   * \brief Free storage directly, without putting it into memory pool.
   *  This can synchronization of all previous runned device functions.
   *
   *  This function is suitable for conatiner structure with requirement on upsizing
   *  in the beginning phase of the iteration.
   *
   * \param handle Handle struct.
   */
  virtual void DirectFree(Handle handle) = 0;
  /*!
   * \brief Destructor.
   */
  virtual ~Storage() {}
  /*!
   * \return Storage singleton.
   */
  static Storage* Get();
  /*!
   * \brief Get shared pointer reference to engine singleton.
   *  Most user should not call this function.
   *  This function is called by another singleton X who requires
   *  Storage to be destructed after X.
   *
   * \return A shared pointer to Storage singleton.
   */
  static std::shared_ptr<Storage> _GetSharedRef();
};  // class Storage
}  // namespace mxnet
#endif  // MXNET_STORAGE_H_
