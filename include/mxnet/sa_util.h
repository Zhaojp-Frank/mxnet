#ifndef MXNET_SA_UTIL_H_
#define MXNET_SA_UTIL_H_

#include <sstream>
#include <thread>
#include <iostream>

//#define SWAPADV_DEBUG


#define sa_likely(x)      __builtin_expect(!!(x), 1)
#define sa_unlikely(x)    __builtin_expect(!!(x), 0)


#ifdef SWAPADV_DEBUG
#define SWAPADV_REPORT_PROGRESS 1
#define sa_log \
  SA_Log().GetStream()
#else
#define SWAPADV_REPORT_PROGRESS 0
#define sa_log \
  if (false) SA_Log().GetStream()
#endif

class SA_Log {
 public:
  explicit SA_Log() {
    id_ = std::this_thread::get_id();
    sstream_ << "[" << id_ << "] ";
  }
  std::ostringstream& GetStream() { return sstream_; }
  ~SA_Log() { std::cout << sstream_.str(); }

 private:
  std::ostringstream sstream_;
  std::thread::id id_;
};
#endif
