#ifndef MXNET_MEM_MGR_H_
#define MXNET_MEM_MGR_H_

#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <stdio.h>

namespace mxnet {

const int MINALLOCSIZE_ = 128;
const int FREELISTSIZE_ = 21;

typedef enum {
  memStatus_Sucess,
  memStatus_InvalidValue,
  memStatus_OutOfMemory,
  memStatus_CUDAError,
} memStatus_t;

inline std::string memGetStatusString(memStatus_t status) {
  switch (status) {
    case memStatus_Sucess: return "Sucess";
    case memStatus_InvalidValue: return "Invalid value";
    case memStatus_OutOfMemory: return "Out of memory";
    case memStatus_CUDAError: return "CUDA error";
  }
  return "Unknown error";
}

//min allocation size = 2^7 = 128 bytes
inline int getFreeListIdx(size_t size) {
  int idx = 0;
  while (size > 0) {
    size >>= 1;
    idx++;
  }  
  idx -= 7;
  //prevent seg fault if index is out of range
  idx = (idx < 0) ? 0 : idx;
  idx = (idx > 20) ? 20 : idx;
  return idx;
}

class Block {
  private: 
    char* data_;
    std::size_t size_;
    Block* nextBlock_;
    bool isFree_;
    bool isHead_;

  public:
    Block(char* data, size_t size)
      : data_(data),
        size_(size),
        nextBlock_(NULL),
        isFree_(true),
        isHead_(false) {
    }

    char* getData() { return data_; }
    size_t getSize() { return size_; }
    Block* getNext() {return nextBlock_; }
    bool isHead() { return isHead_; }

    void setHead() { isHead_ = true; }
    void setNext(Block* b) { nextBlock_ = b; }
    void setAllocated() { isFree_ = false; }
    void setFree() { isFree_ = true; }
}; // Class Block

class MemoryManager {
  Block* allocatedList_;
  Block* freeList_[FREELISTSIZE_];
  Block* usedList_;
  std::mutex mutex_;
  public:
    static MemoryManager* Get();
    static std::shared_ptr<MemoryManager> _GetSharedRef();
    ~MemoryManager();
    cudaError_t Malloc(void** devptr, size_t size);
    cudaError_t Free(void* devptr);

  private:
    MemoryManager();
    Block* FindFirstFit(int idx, Block* prev, size_t size);
    Block* AllocateBlock(size_t size);
    void SplitAndPlace(Block* b, Block* prev, int idx, size_t size);
};  // Class MemoryManager

} //namespace mxnet

#endif // MXNET_MEM_MGR_H_
