#ifndef LEARN_CUDA_UTILS_CUDA_EVENT_H_
#define LEARN_CUDA_UTILS_CUDA_EVENT_H_

#include "utils/macros.h"

class CudaEvent {
 public:
  DISALLOW_COPY_AND_ASSIGN(CudaEvent)
  explicit CudaEvent(unsigned int flags = cudaEventDefault) {
    CUDA_CHECK(cudaEventCreateWithFlags(&cuda_event_, flags));
  }
  ~CudaEvent() { CUDA_CHECK(cudaEventDestroy(cuda_event_)); }

  cudaEvent_t get() { return cuda_event_; }

 private:
  cudaEvent_t cuda_event_;
};

#endif  // LEARN_CUDA_UTILS_CUDA_EVENT_H_
