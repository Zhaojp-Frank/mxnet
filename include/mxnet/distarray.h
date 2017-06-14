#ifndef MXNET_DISTARRAY_H_
#define MXNET_DISTARRAY_H_

#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <dmlc/type_traits.h>
#include "./ndarray.h"

namespace mxnet {
class DistArray {
 public:
  DistArray() {
  }

  inline const TShape &shape() const {
    /* TBD */
  }

  inline int dtype() const {
    /* TBD */
  }

  /*
   * TODO: Does DistArray need a context?
   * inline Context ctx() const {
   * }
   */

  /* 
   * TODO: Does DistArray must be associated with a variable?
   * inline Engine::VarHandle var() const {
   * }
   */

  inline bool is_none() const {
    return false;
  }

  inline void WaitToRead() const {
    /* TBD */
  }

  inline void WaitToWrite() const {
    /* TBD */
  }

  /* Unlike NDArray which has Save/Load, A DistArray must be converted from/into
   * a NDArray. Thus a DistArray can only be saved/loaded through converting
   * from/into a NDArray first.
   *
   * void Save(dmlc::Stream *strm) const;
   * bool Load(dmlc::Stream *strm);
   * void SyncCopyFromCpu(const void *data, size_t size) const;
   * void SyncCopyToCpu(const void *data, size_t size) const;
   */
  NDArray ToNDArray() const {
  }

  void FromNDArray() {
  }

  DistArray &operator=(real_t scalar);

  DistArray &operator+=(const DistArray &src);

  DistArray &operator+=(const NDArray &src);

  DistArray &operator+=(const real_t &src);

  DistArray &operator-=(const DistArray &src);

  DistArray &operator-=(const NDArray &src);

  DistArray &operator-=(const real_t &src);

  DistArray &operator*=(const DistArray &src);

  DistArray &operator*=(const NDArray &src);

  DistArray &operator*=(const real_t &src);

  DistArray &operator/=(const DistArray &src);

  DistArray &operator/=(const NDArray &src);

  DistArray &operator/=(const real_t &src);

  DistArray T() const;

  /* TODO: This is necessary but do we need a Context to do so? */
  DistArray Copy(Context ctx) const;

  /* 
   * TODO: How to support these easily? 
   * inline DistArray Slice(index_t begin, index_t end) const {
   * }
   * inline DistArray At(index_t idx) const {
   * }
   * inline NDArray AsArray(const TShape &shape, int dtype) const {
   * }
   * inline NDArray Reshape(const TShazpe &shape) const {
   * }
   */

private:
  /* std::vectors<DistArrayWorker> sub_arrays;*/
  TShape shape_;
  int dtype_ = -1;


};

}  // namespace mxnet

#endif
