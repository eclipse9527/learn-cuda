#ifndef LEARN_CUDA_UTILS_MACROS_H_
#define LEARN_CUDA_UTILS_MACROS_H_

#include <iostream>

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  TypeName& operator=(const TypeName&) = delete;

#define CUDA_CHECK(callstr)                                                   \
  {                                                                           \
    auto error_code = callstr;                                                \
    /* Code block avoids redefinition of cudaError_t error */                 \
    if (error_code != cudaSuccess) {                                          \
      std::cerr << "CUDA error: " << cudaGetErrorString(error_code) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;                  \
      exit(1);                                                                \
    }                                                                         \
  }

#endif  // LEARN_CUDA_UTILS_MACROS_H_
