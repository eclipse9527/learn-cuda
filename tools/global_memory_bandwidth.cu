#include <stdio.h>

#include "utils/cuda_event.h"
#include "utils/macros.h"

__global__ void Saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

int main() {
  int device_num;

  cudaGetDeviceCount(&device_num);
  for (int i = 0; i < device_num; i++) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
  }

  const int N = 20 * (1 << 20);
  float *x, *y, *d_x, *d_y;

  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));

  CudaEvent start, stop;

  CUDA_CHECK(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaEventRecord(start.get()));
  Saxpy<<<(N + 511) / 512, 512>>>(N, 2.0f, d_x, d_y);
  CUDA_CHECK(cudaEventRecord(stop.get()));

  cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop.get());

  float elapsed_time_ms = 0;
  cudaEventElapsedTime(&elapsed_time_ms, start.get(), stop.get());

  printf("Effective Bandwidth (GB/s): %f\n", N * 4 * 3 / elapsed_time_ms / 1e6);

  free(x);
  free(y);
  cudaFree(d_x);
  cudaFree(d_y);

  return 0;
}
