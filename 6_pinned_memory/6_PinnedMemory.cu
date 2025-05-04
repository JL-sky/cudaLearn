#include <iostream>

#include "../utils.hpp"

constexpr int SIZE = 10 * 1024 * 1024;
/*
以下针对host侧，分别使用malloc和cudaHostAlloc进行内存分配进行性能测试
*/
float CudaMallocTest(int size, bool up) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  float elapsed_time = 0;
  int *a;
  a = (int *)malloc(size * sizeof(*a));
  int *dev_a;
  CUDA_CHECK(cudaMalloc(&dev_a, size * sizeof(*dev_a)));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < 100; ++i) {
    if (up) {
      CUDA_CHECK(
          cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice));
    } else {
      CUDA_CHECK(
          cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
    }
  }
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  free(a);
  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_time;
}

float CudaHostAllocTest(int size, bool up) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  float elapsed_time = 0;
  int *a;
  CUDA_CHECK(cudaHostAlloc(&a, size * sizeof(*a), cudaHostAllocDefault));
  int *dev_a;
  CUDA_CHECK(cudaMalloc(&dev_a, size * sizeof(*dev_a)));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < 100; ++i) {
    if (up) {
      CUDA_CHECK(
          cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice));
    } else {
      CUDA_CHECK(
          cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
    }
  }
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  CUDA_CHECK(cudaFreeHost(a));
  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_time;
}

int main() {
  float elapsed_time;
  float MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;
  printf("============ host to device test =============\n");
  elapsed_time = CudaMallocTest(SIZE, true);
  printf("Time using CudaMalloc:%3.1f ms\n", elapsed_time);
  printf("\tMB/s during copy up: %3.1f\n", MB / (elapsed_time / 1000));

  elapsed_time = CudaHostAllocTest(SIZE, true);
  printf("Time using CudaHostAlloc:%3.1f ms\n", elapsed_time);
  printf("\tMB/s during copy up: %3.1f\n\n", MB / (elapsed_time / 1000));

  printf("============ device to host test =============\n");
  elapsed_time = CudaMallocTest(SIZE, false);
  printf("Time using CudaMalloc:%3.1f ms\n", elapsed_time);
  printf("\tMB/s during copy up: %3.1f\n", MB / (elapsed_time / 1000));

  elapsed_time = CudaHostAllocTest(SIZE, false);
  printf("Time using CudaHostAlloc:%3.1f ms\n", elapsed_time);
  printf("\tMB/s during copy up: %3.1f\n", MB / (elapsed_time / 1000));
  return 0;
}