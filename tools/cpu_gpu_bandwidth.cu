#include <stdio.h>

#include "utils/cuda_event.h"
#include "utils/macros.h"

void ProfileCopies(float *h_a, float *h_b, float *d, uint32_t n,
                   const char *desc) {
  printf("\n%s transfers\n", desc);

  uint32_t bytes = n * sizeof(float);

  // events for timing
  CudaEvent start, stop;

  CUDA_CHECK(cudaEventRecord(start.get(), 0));
  CUDA_CHECK(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(stop.get(), 0));
  CUDA_CHECK(cudaEventSynchronize(stop.get()));

  float time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&time_ms, start.get(), stop.get()));
  printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time_ms);

  CUDA_CHECK(cudaEventRecord(start.get(), 0));
  CUDA_CHECK(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(stop.get(), 0));
  CUDA_CHECK(cudaEventSynchronize(stop.get()));

  CUDA_CHECK(cudaEventElapsedTime(&time_ms, start.get(), stop.get()));
  printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time_ms);

  for (uint32_t i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      printf("*** %s transfers failed ***", desc);
      break;
    }
  }
}

int main() {
  const uint32_t N = 4 * 1024 * 1024;
  const uint32_t bytes = N * sizeof(float);

  // host arrays
  float *h_aPageable, *h_bPageable;
  float *h_aPinned, *h_bPinned;

  // device array
  float *d_a;

  // allocate and initialize
  h_aPageable = (float *)malloc(bytes);                    // host pageable
  h_bPageable = (float *)malloc(bytes);                    // host pageable
  CUDA_CHECK(cudaMallocHost((void **)&h_aPinned, bytes));  // host pinned
  CUDA_CHECK(cudaMallocHost((void **)&h_bPinned, bytes));  // host pinned
  CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));            // device

  for (uint32_t i = 0; i < N; ++i) h_aPageable[i] = i;
  memcpy(h_aPinned, h_aPageable, bytes);
  memset(h_bPageable, 0, bytes);
  memset(h_bPinned, 0, bytes);

  // output device info and transfer size
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  printf("\nDevice: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

  // perform copies and report bandwidth
  ProfileCopies(h_aPageable, h_bPageable, d_a, N, "Pageable");
  ProfileCopies(h_aPinned, h_bPinned, d_a, N, "Pinned");

  // cleanup
  cudaFree(d_a);
  cudaFreeHost(h_aPinned);
  cudaFreeHost(h_bPinned);
  free(h_aPageable);
  free(h_bPageable);

  return 0;
}
