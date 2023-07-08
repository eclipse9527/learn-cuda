#include <stdio.h>

#include "utils/macros.h"

__global__ void HelloWorld() { printf("Hello Cuda!\n"); }

int main(int argc, char **argv) {
  HelloWorld<<<1, 1>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
