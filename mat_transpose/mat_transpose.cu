#include <stdio.h>

#include "utils/cuda_event.h"
#include "utils/macros.h"

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;
constexpr int kNumRows = 4096;
constexpr int kNumCols = 2048;
constexpr int kDeviceId = 0;
constexpr int kNumRuns = 100;

void DataCheck(const float* ref, const float* res, int n, float ms) {
  bool passed = true;
  for (int i = 0; i < n; i++)
    if (res[i] != ref[i]) {
      printf("%d %f %f\n", i, res[i], ref[i]);
      printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
  if (passed) {
    printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * kNumRuns / ms);
  }
}

__global__ void copy(float* odata, float* idata) {
  int x = blockIdx.x * kTileDim + threadIdx.x;
  int y = blockIdx.y * kTileDim + threadIdx.y;
  int width = gridDim.x * kTileDim;

  for (int j = 0; j < kTileDim; j += kBlockRows) {
    odata[(y + j) * width + x] = idata[(y + j) * width + x];
  }
}

__global__ void copySharedMem(float* odata, const float* idata) {
  __shared__ float tile[kTileDim * kTileDim];

  int x = blockIdx.x * kTileDim + threadIdx.x;
  int y = blockIdx.y * kTileDim + threadIdx.y;
  int width = gridDim.x * kTileDim;

  for (int j = 0; j < kTileDim; j += kBlockRows)
    tile[(threadIdx.y + j) * kTileDim + threadIdx.x] =
        idata[(y + j) * width + x];

  __syncthreads();

  for (int j = 0; j < kTileDim; j += kBlockRows)
    odata[(y + j) * width + x] =
        tile[(threadIdx.y + j) * kTileDim + threadIdx.x];
}

__global__ void transposeNaive(float* odata, const float* idata) {
  int x = blockIdx.x * kTileDim + threadIdx.x;
  int y = blockIdx.y * kTileDim + threadIdx.y;
  int i_width = gridDim.x * kTileDim;
  int o_width = gridDim.y * kTileDim;
  for (int j = 0; j < kTileDim; j += kBlockRows) {
    odata[x * o_width + (y + j)] = idata[(y + j) * i_width + x];
  }
}

__global__ void transposeSharedMem(float* odata, const float* idata) {
  __shared__ float tile[kTileDim][kTileDim];

  int x = blockIdx.x * kTileDim + threadIdx.x;
  int y = blockIdx.y * kTileDim + threadIdx.y;
  int i_width = gridDim.x * kTileDim;
  int o_width = gridDim.y * kTileDim;

  for (int j = 0; j < kTileDim; j += kBlockRows) {
    tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * i_width + x];
  }
  __syncthreads();

  int ox = blockIdx.y * kTileDim + threadIdx.x;
  int oy = blockIdx.x * kTileDim + threadIdx.y;

  for (int j = 0; j < kTileDim; j += kBlockRows) {
    odata[(oy + j) * o_width + ox] = tile[threadIdx.x][threadIdx.y + j];
  }
}

__global__ void transposeNoBankConflicts(float* odata, const float* idata) {
  __shared__ float tile[kTileDim][kTileDim + 1];

  int x = blockIdx.x * kTileDim + threadIdx.x;
  int y = blockIdx.y * kTileDim + threadIdx.y;
  int i_width = gridDim.x * kTileDim;
  int o_width = gridDim.y * kTileDim;

  for (int j = 0; j < kTileDim; j += kBlockRows) {
    tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * i_width + x];
  }
  __syncthreads();

  int ox = blockIdx.y * kTileDim + threadIdx.x;
  int oy = blockIdx.x * kTileDim + threadIdx.y;

  for (int j = 0; j < kTileDim; j += kBlockRows) {
    odata[(oy + j) * o_width + ox] = tile[threadIdx.x][threadIdx.y + j];
  }
}

int main() {
  if (kNumRows % kTileDim || kNumCols % kTileDim) {
    printf("The number of rows and cols must be a multiple of %d\n",
           kBlockRows);
    return 1;
  }
  if (kTileDim % kBlockRows) {
    printf("The tile dimension must be a multiple of %d\n", kBlockRows);
    return 1;
  }

  const int mem_size = kNumRows * kNumCols * sizeof(float);

  dim3 dim_block(kTileDim, kBlockRows, 1);
  dim3 dim_grid(kNumCols / kTileDim, kNumRows / kTileDim, 1);

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, kDeviceId));
  printf("\nDevice : %s\n", prop.name);
  printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", kNumCols,
         kNumRows, kTileDim, kBlockRows, kTileDim, kTileDim);
  printf("dim_grid: %d %d %d. dim_block: %d %d %d\n", dim_grid.x, dim_grid.y,
         dim_grid.z, dim_block.x, dim_block.y, dim_block.z);

  CUDA_CHECK(cudaSetDevice(kDeviceId));

  float* h_idata = (float*)malloc(mem_size);
  float* h_cdata = (float*)malloc(mem_size);
  float* h_tdata = (float*)malloc(mem_size);
  float* h_gold = (float*)malloc(mem_size);

  float *d_idata, *d_cdata, *d_tdata;
  CUDA_CHECK(cudaMalloc((void**)&d_idata, mem_size));
  CUDA_CHECK(cudaMalloc((void**)&d_cdata, mem_size));
  CUDA_CHECK(cudaMalloc((void**)&d_tdata, mem_size));

  for (int i = 0; i < kNumRows; ++i) {
    for (int j = 0; j < kNumCols; ++j) {
      h_idata[i * kNumCols + j] = i * kNumCols + j;
    }
  }
  for (int i = 0; i < kNumRows; ++i) {
    for (int j = 0; j < kNumCols; ++j) {
      h_gold[j * kNumRows + i] = h_idata[i * kNumCols + j];
    }
  }

  CUDA_CHECK(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

  CudaEvent start, stop;
  float cost_ms;

  printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

  printf("%25s", "copy");
  CUDA_CHECK(cudaMemset(d_cdata, 0, mem_size));
  // warm up
  copy<<<dim_grid, dim_block>>>(d_cdata, d_idata);

  CUDA_CHECK(cudaEventRecord(start.get()));
  for (int i = 0; i < kNumRuns; i++) {
    copy<<<dim_grid, dim_block>>>(d_cdata, d_idata);
  }
  CUDA_CHECK(cudaEventRecord(stop.get()));
  CUDA_CHECK(cudaEventSynchronize(stop.get()));
  CUDA_CHECK(cudaEventElapsedTime(&cost_ms, start.get(), stop.get()));
  CUDA_CHECK(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
  DataCheck(h_idata, h_cdata, kNumRows * kNumCols, cost_ms);

  printf("%25s", "copySharedMem");
  CUDA_CHECK(cudaMemset(d_cdata, 0, mem_size));
  // warm up
  copySharedMem<<<dim_grid, dim_block>>>(d_cdata, d_idata);

  CUDA_CHECK(cudaEventRecord(start.get()));
  for (int i = 0; i < kNumRuns; i++) {
    copySharedMem<<<dim_grid, dim_block>>>(d_cdata, d_idata);
  }
  CUDA_CHECK(cudaEventRecord(stop.get()));
  CUDA_CHECK(cudaEventSynchronize(stop.get()));
  CUDA_CHECK(cudaEventElapsedTime(&cost_ms, start.get(), stop.get()));
  CUDA_CHECK(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
  DataCheck(h_idata, h_cdata, kNumRows * kNumCols, cost_ms);

  printf("%25s", "transposeNaive");
  CUDA_CHECK(cudaMemset(d_tdata, 0, mem_size));
  // warm up
  transposeNaive<<<dim_grid, dim_block>>>(d_tdata, d_idata);

  CUDA_CHECK(cudaEventRecord(start.get()));
  for (int i = 0; i < kNumRuns; i++) {
    transposeNaive<<<dim_grid, dim_block>>>(d_tdata, d_idata);
  }
  CUDA_CHECK(cudaEventRecord(stop.get()));
  CUDA_CHECK(cudaEventSynchronize(stop.get()));
  CUDA_CHECK(cudaEventElapsedTime(&cost_ms, start.get(), stop.get()));
  CUDA_CHECK(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
  DataCheck(h_gold, h_tdata, kNumRows * kNumCols, cost_ms);

  printf("%25s", "transposeSharedMem");
  CUDA_CHECK(cudaMemset(d_tdata, 0, mem_size));
  // warm up
  transposeSharedMem<<<dim_grid, dim_block>>>(d_tdata, d_idata);

  CUDA_CHECK(cudaEventRecord(start.get()));
  for (int i = 0; i < kNumRuns; i++) {
    transposeSharedMem<<<dim_grid, dim_block>>>(d_tdata, d_idata);
  }
  CUDA_CHECK(cudaEventRecord(stop.get()));
  CUDA_CHECK(cudaEventSynchronize(stop.get()));
  CUDA_CHECK(cudaEventElapsedTime(&cost_ms, start.get(), stop.get()));
  CUDA_CHECK(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
  DataCheck(h_gold, h_tdata, kNumRows * kNumCols, cost_ms);

  printf("%25s", "transposeNoBankConflicts");
  CUDA_CHECK(cudaMemset(d_tdata, 0, mem_size));
  // warm up
  transposeNoBankConflicts<<<dim_grid, dim_block>>>(d_tdata, d_idata);

  CUDA_CHECK(cudaEventRecord(start.get()));
  for (int i = 0; i < kNumRuns; i++) {
    transposeNoBankConflicts<<<dim_grid, dim_block>>>(d_tdata, d_idata);
  }
  CUDA_CHECK(cudaEventRecord(stop.get()));
  CUDA_CHECK(cudaEventSynchronize(stop.get()));
  CUDA_CHECK(cudaEventElapsedTime(&cost_ms, start.get(), stop.get()));
  CUDA_CHECK(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
  DataCheck(h_gold, h_tdata, kNumRows * kNumCols, cost_ms);

  free(h_idata);
  free(h_cdata);
  free(h_tdata);
  free(h_gold);
  CUDA_CHECK(cudaFree(d_idata));
  CUDA_CHECK(cudaFree(d_cdata));
  CUDA_CHECK(cudaFree(d_tdata));
}
