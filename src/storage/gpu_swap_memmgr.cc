#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dmlc/parameter.h>
#include <dmlc/logging.h>
#include <math.h>

#include "../common/cuda_utils.h"
#include "./gpu_swap_memmgr.h"
#include "./gpu_swap_util.h"

namespace mxnet {

#define CUDA_CALL(func) {                                   \
  cudaError_t e = (func);                                   \
  CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)  \
        << __FUNCTION__ << ":" << __LINE__                  \
        << "has a CUDA error: " << cudaGetErrorString(e);   \
}

cudaError_t MemoryManager::Memcpy(int device_id, void* dst, const void* src,
                                  size_t count, enum cudaMemcpyKind kind) {
  CUDA_CALL(cudaSetDevice(device_id));
  CUDA_CALL(cudaMemcpy(dst, src, count, kind));
  return cudaSuccess;
}

cudaError_t MemoryManager::MemcpyAsync(
        int device_id, void* dst, const void* src, size_t count,
        cudaMemcpyKind kind, cudaStream_t stream) {
  CUDA_CALL(cudaSetDevice(device_id));
  CUDA_CALL(cudaMemcpyAsync(dst, src, count, kind, stream));
  return cudaSuccess;
}

cudaError_t MemoryManager::StreamSynchronize(int device_id,
                                             cudaStream_t stream) {
  CUDA_CALL(cudaSetDevice(device_id));
  CUDA_CALL(cudaStreamSynchronize(stream));
  return cudaSuccess;
}

cudaError_t MemoryManager::Malloc(void*& devptr, size_t size, int device_id) {
  malloc_count_[device_id]++;
  malloc_size_[device_id] += size;
  malloc_type_[device_id][size] += 1;
  return this->MallocInternal(devptr, size, device_id);
}

cudaError_t MemoryManager::Free(void* devptr, int device_id) {
  free_count_[device_id]++;
  return this->FreeInternal(devptr, device_id);
}

void MemoryManager::Statistics() {
  for (int i = 0; i < NUMBER_OF_GPU; i++) {
    std::cout << "MemoryManager " << i << " Statistics:" << std::endl
              << "=> Malloc count: " << malloc_count_[i] << std::endl
              << "=> Malloc size: " << GBString(malloc_size_[i]) << std::endl
              << "=> Malloc type: " << malloc_type_[i].size() << std::endl
              << "=> Free count: " << free_count_[i] << std::endl;
    malloc_count_[i] = malloc_size_[i] = free_count_[i] = 0;
  }
  this->StatisticsInternal();
}

CudaMemoryManager::CudaMemoryManager() {
  std::cout << "Initialize CUDA Memory Allocator" << std::endl;
}

CudaMemoryManager::~CudaMemoryManager() {
  std::cout << "Destroy Cuda Memory Allocator" << std::endl;
}

cudaError_t CudaMemoryManager::MemGetInfo(int device_id, size_t* total,
                                          size_t* free) {
  CUDA_CALL(cudaSetDevice(device_id));
  CUDA_CALL(cudaMemGetInfo(free, total));
  return cudaSuccess;
}

bool CudaMemoryManager::TryAllocate(int device_id, size_t size) {
  CUDA_CALL(cudaSetDevice(device_id));
  size_t free, total;
  CUDA_CALL(cudaMemGetInfo(&free, &total));
  // TODO(fegin): This fixed threshold is not acceptable.
  // FIXME(fegin): The maximum threshould I used in the old MXNet is 512 MB.
  //               We should figure out why such a large threshold is needed
  //               for current implementation.
  return free > size + 1500000000;
}

cudaError_t CudaMemoryManager::MallocInternal(void*& devptr, size_t size,
                                              int device_id) {
  CUDA_CALL(cudaSetDevice(device_id));
  CUDA_CALL(cudaMalloc(&devptr, size));
  return cudaSuccess;
}

cudaError_t CudaMemoryManager::FreeInternal(void* devptr, int device_id) {
  CUDA_CALL(cudaSetDevice(device_id));
  CUDA_CALL(cudaFree(devptr));
  return cudaSuccess;
}

void CudaMemoryManager::StatisticsInternal() {
}

BuddyMemoryManager::BuddyMemoryManager() {
  std::cout << "Initializing Memory Manager" << std::endl;
  buddy_.resize(NUMBER_OF_GPU);
  for (size_t device = 0; device < NUMBER_OF_GPU; device++) {
    size_t avail, total;
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMemGetInfo(&avail, &total));
    bool infinite_memory = dmlc::GetEnv("MXNET_INFINITE_MEMORY", false);
    if (infinite_memory) {
        avail = static_cast<size_t>(avail * 0.8);
    } else {
        float ratio = dmlc::GetEnv("MXNET_GPU_UTIL_RATIO", kGPUUtilRatio);
        avail = static_cast<size_t>(avail * ratio);
    }
    void* memory = nullptr;
    while (cudaMalloc((void**)&memory, avail) == cudaErrorMemoryAllocation) {
      avail -= kMB;
      if (avail == 0) {
        break;
      }
    }
    CHECK(avail > 0);
    buddy_[device] = new BuddySystem(memory, avail, device);
    std::cout << "Buddy System No." << device << " initialized with size = "
              << GBString(avail) << std::endl;
  }
  std::cout << "Memory Manager initialization completed" << std::endl;
}

BuddyMemoryManager::~BuddyMemoryManager() {
  for (size_t device = 0; device < NUMBER_OF_GPU; device++) {
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaFree((void*)(buddy_[device]->Memory())));
    delete buddy_[device];
    buddy_[device] = nullptr;
    std::cout << "Buddy System No." << device << " destructed" << std::endl;
  }
}

cudaError_t BuddyMemoryManager::MemGetInfo(int device_id, size_t* total,
                                           size_t* free) {
  std::lock_guard<std::mutex> lock(mutex_[device_id]);
  buddy_[device_id]->MemoryUsage(total, free);
  return cudaSuccess;
}

bool BuddyMemoryManager::TryAllocate(int device_id, size_t size) {
  std::lock_guard<std::mutex> lock(mutex_[device_id]);
  return buddy_[device_id]->TryAllocate(size);
}

cudaError_t BuddyMemoryManager::MallocInternal(void*& devptr, size_t size,
                                               int device_id) {
  std::lock_guard<std::mutex> lock(mutex_[device_id]);
  devptr = buddy_[device_id]->Malloc(size);
  return (devptr) ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t BuddyMemoryManager::FreeInternal(void* devptr, int device_id) {
  std::lock_guard<std::mutex> lock(mutex_[device_id]);
  buddy_[device_id]->Free(devptr);
  return cudaSuccess;
}

void BuddyMemoryManager::StatisticsInternal() {
  for (int i = 0; i < NUMBER_OF_GPU; i++) {
    buddy_[i]->Statistics();
  }
}

FakeMemoryManager::FakeMemoryManager() {
  std::cout << "Initializing Fake Memory Manager" << std::endl;
  ptrs_.resize(NUMBER_OF_GPU);
  for (size_t device = 0; device < NUMBER_OF_GPU; device++) {
    size_t avail, total;
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMemGetInfo(&avail, &total));
    float ratio = dmlc::GetEnv("MXNET_GPU_UTIL_RATIO", kGPUUtilRatio);
    avail = static_cast<size_t>(avail * ratio);
    void* memory = nullptr;
    while (cudaMalloc((void**)&memory, avail) == cudaErrorMemoryAllocation) {
      avail -= kMB;
      if (avail == 0) {
        break;
      }
    }
    CHECK(avail > 0);
    ptrs_[device] = memory;
    std::cout << "Fake system No." << device << " initialized with size = "
              << GBString(avail) << std::endl;
  }
  std::cout << "Fake Memory Manager initialization completed" << std::endl;
}

FakeMemoryManager::~FakeMemoryManager() {
  for (size_t device = 0; device < NUMBER_OF_GPU; device++) {
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaFree((void*)ptrs_[device]));
    ptrs_[device] = nullptr;
    std::cout << "Fake system No." << device << " destructed" << std::endl;
  }
  std::cout << "Destroy Fake Memory Allocator" << std::endl;
}

cudaError_t FakeMemoryManager::MemGetInfo(int device_id, size_t* total,
                                          size_t* free) {
  *total = ~1;
  *free = ~1;
  return cudaSuccess;
}

bool FakeMemoryManager::TryAllocate(int device_id, size_t size) {
  return true;
}

cudaError_t FakeMemoryManager::MallocInternal(void*& devptr, size_t size,
                                              int device_id) {
  devptr = ptrs_[device_id];
  return cudaSuccess;
}

cudaError_t FakeMemoryManager::FreeInternal(void* devptr, int device_id) {
  return cudaSuccess;
}

void FakeMemoryManager::StatisticsInternal() {
}

/*
SlabMemoryManager::SlabMemoryManager() {
  std::cout << "Initializing Memory Manager" << std::endl;
  buddy_.resize(NUMBER_OF_GPU);
  for (size_t device = 0; device < NUMBER_OF_GPU; device++) {
    size_t avail, total;
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMemGetInfo(&avail, &total));
    float ratio = dmlc::GetEnv("MXNET_GPU_UTIL_RATIO", kGPUUtilRatio);
    avail = static_cast<size_t>(avail * ratio);
    void* memory = nullptr;
    while (cudaMalloc((void**)&memory, avail) == cudaErrorMemoryAllocation) {
      avail -= kMB;
      if (avail == 0) {
        break;
      }
    }
    CHECK(avail > 0);
    buddy_[device] = new SlabSystem(memory, avail, device);
    std::cout << "Slab System No." << device << " initialized with size = "
              << avail << " bytes" << std::endl;
  }
  std::cout << "Memory Manager initialization completed" << std::endl;
}

SlabMemoryManager::~SlabMemoryManager() {
  for (size_t device = 0; device < NUMBER_OF_GPU; device++) {
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaFree((void*)(buddy_[device]->Memory())));
    delete buddy_[device];
    buddy_[device] = nullptr;
    std::cout << "Slab System No." << device << " destructed" << std::endl;
  }
}

cudaError_t SlabMemoryManager::Malloc(void*& devptr, size_t size,
                                       int device_id) {
  std::lock_guard<std::mutex> lock(mutex_[device_id]);
  devptr = buddy_[device_id]->Malloc(size);
  return (devptr) ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t SlabMemoryManager::Free(void* devptr, int device_id) {
  std::lock_guard<std::mutex> lock(mutex_[device_id]);
  buddy_[device_id]->Free(devptr);
  return cudaSuccess;
}

//returns total memory and total free memory(not necessarily consequtive) in mmu
cudaError_t SlabMemoryManager::MemGetInfo(int device_id, size_t* total,
                                           size_t* free) {
  std::lock_guard<std::mutex> lock(mutex_[device_id]);
  buddy_[device_id]->MemoryUsage(total, free);
  return cudaSuccess;
}

bool SlabMemoryManager::TryAllocate(int device_id, size_t size) {
  std::lock_guard<std::mutex> lock(mutex_[device_id]);
  return buddy_[device_id]->TryAllocate(size);
}
*/
// Factory functions.
std::shared_ptr<MemoryManager> GetMemoryManagerRef() {
  static std::shared_ptr<MemoryManager> inst;
  static bool set = false;
  if (!set) {
    std::string mem_mgr_type = dmlc::GetEnv("MXNET_MEM_MGR_TYPE",
                                            std::string("CUDA"));
    bool infinite_memory = dmlc::GetEnv("MXNET_INFINITE_MEMORY", false);
    if (infinite_memory) {
        mem_mgr_type = "Buddy";
    }
    std::cout << "MXNET_MEM_MGR_TYPE: " << mem_mgr_type << std::endl;
    if (mem_mgr_type == "CUDA") {
      inst.reset(dynamic_cast<MemoryManager*>(new CudaMemoryManager()));
    } else if (mem_mgr_type == "Buddy") {
      inst.reset(dynamic_cast<MemoryManager*>(new BuddyMemoryManager()));
    } else if (mem_mgr_type == "FAKE") {
      inst.reset(dynamic_cast<MemoryManager*>(new FakeMemoryManager()));
    }
    set = true;
  }
  return inst;
}

MemoryManager* GetMemoryManager() {
  static MemoryManager* mm = GetMemoryManagerRef().get();
  return mm;
}

} // namespace mxnet
