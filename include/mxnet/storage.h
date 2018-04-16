/*!
 * Copyright (c) 2015 by Contributors
 * \file storage.h
 * \brief Storage manager across multiple devices.
 */
#ifndef MXNET_STORAGE_H_
#define MXNET_STORAGE_H_

#include <mutex>
#include <pthread.h>
#include <thread>
#include <memory>
#include <chrono>
#if MXNET_USE_CUDA
#include <cuda_runtime.h>
#endif  // MXNET_USE_CUDA
#include "./base.h"

using namespace std::chrono;

namespace mxnet {
using handle_id_t = unsigned long long;
using timestamp_t = unsigned long long;

class SwapHistory {
public:
    enum record_t {SET_ADDR, GET_ADDR, DEL_ADDR};
    struct SwapRecord {
        handle_id_t id;
        record_t type;
        timestamp_t time;
        size_t size;
    };

    ~SwapHistory();
    static SwapHistory* Get();
    static std::shared_ptr<SwapHistory> _GetSharedRef();
    void PutRecord(handle_id_t id, record_t type, size_t size);
    void StartIteration();
    void StopIteration();

private:
    SwapHistory();

    std::vector<SwapRecord> history_;
    bool iteration_started_;
    bool do_record_;
    size_t record_idx_;
    size_t iteration_idx_;
    high_resolution_clock::time_point begin_time_;
    std::mutex mutex_;
};

class Swap {
public:
    struct SwapInfo {
        handle_id_t handle_id;
        bool swap_in;
        int swap_count;
        int device;
        void* dptr;
        char* cpu_address;
        size_t size;
        std::list<SwapInfo*>::iterator it;
    };
    ~Swap();
    static Swap* Get();
    static std::shared_ptr<Swap> _GetSharedRef();
    void SwapOut(unsigned required_memory, int device);
    void SwapIn(SwapInfo *info);
    int UpdateFree();
    void SetAddr(handle_id_t handle_id, void* dptr, size_t size);
    void DelAddr(handle_id_t handle_id, size_t size, bool preserve);
    bool FreeReserved(void *ptr, size_t size);
    bool CheckReservedAndFree(void *ptr, size_t size);
    void* GetAddr(handle_id_t handle_id, size_t size);

private:
    Swap();
    std::unordered_map<handle_id_t, SwapInfo*> swap_info_;
    std::unordered_map<void*, size_t> reserved_mem_;
    std::vector<std::list<SwapInfo*>> lru_;
    std::vector<size_t> free_memory_;
    std::mutex mutex_;
    bool do_swap_;
    size_t swap_threshold_;
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
        if (!handle_random) {
            handle_gen = std::mt19937_64(handle_rd());
            handle_random = true;
            //std::cout << "Init random number seed." << std::endl;
        }
        //id_ = handle_dis(handle_gen);
        id_ = ++base_id_;
        std::cout << "Handle id " << id_ << std::endl;
    }

    void Free(bool preserve) {
        Swap::Get()->DelAddr(id_, size, preserve);
    }

    /*!
     * \brief Pointer to the data.
     */
    void SetDptr(void* ptr) {
        Swap::Get()->SetAddr(id_, ptr, size);
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
    static std::random_device handle_rd;
    static std::mt19937_64 handle_gen;
    static std::uniform_int_distribution<unsigned long long> handle_dis;
    static handle_id_t base_id_;
    static bool handle_random;
    void* dptr_;
    handle_id_t id_;
  };
  /*!
   * \brief Allocate a new contiguous memory for a given size.
   * \param size Total size of memory in bytes.
   * \param ctx Context information about the device and ID.
   * \return Handle struct.
   */
  virtual Handle Alloc(size_t size, Context ctx) = 0;
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
