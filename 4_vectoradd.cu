#include <cuda_runtime.h>
#include <stdio.h>

#include <chrono>
#include <cmath>
#include <iostream>

constexpr int N = 33 * 1024 * 1024;

// CPU 向量加法函数
void cpu_add(float* a, float* b, float* c) {
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }
}

// GPU 向量加法核函数，使用二维grid的一维block
__global__ void kernel_add(float* a, float* b, float* c) {
  // int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int bSize = blockDim.x;
  int bIndex = blockIdx.x + blockIdx.y * gridDim.x;
  int tid = bIndex * bSize + threadIdx.x;
  // int tid = (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) +
  // threadIdx.x);
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

int main() {
  float *h_a, *h_b, *h_c;
  float *dev_a, *dev_b, *dev_c;

  // 为 CPU 端数组分配内存
  h_a = new float[N];
  h_b = new float[N];
  h_c = new float[N];
  float* cpu_c = new float[N];

  // 为 GPU 端数组分配内存
  cudaMalloc(&dev_a, N * sizeof(float));
  cudaMalloc(&dev_b, N * sizeof(float));
  cudaMalloc(&dev_c, N * sizeof(float));

  // 初始化 CPU 端数组
  for (int i = 0; i < N; ++i) {
    h_a[i] = i;
    h_b[i] = i * 2;
  }

  // 将数据从 CPU 复制到 GPU
  cudaMemcpy(dev_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice);

  // 记录 CPU 计算开始时间
  auto cpu_start = std::chrono::high_resolution_clock::now();
  cpu_add(h_a, h_b, cpu_c);
  // 记录 CPU 计算结束时间
  auto cpu_stop = std::chrono::high_resolution_clock::now();
  // 计算 CPU 计算时间
  auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                          cpu_stop - cpu_start)
                          .count();
  printf("CPU add time is %f ms\n", cpu_duration);

  // 配置 GPU 线程块和网格
  constexpr int threadsPerBlock = 256;
  int gridSize = ceil(sqrt((N + threadsPerBlock - 1.) / threadsPerBlock));
  dim3 grid(gridSize, gridSize);

  // 创建 CUDA 事件
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 记录 GPU 计算开始时间
  cudaEventRecord(start);
  kernel_add<<<grid, threadsPerBlock>>>(dev_a, dev_b, dev_c);
  // 记录 GPU 计算结束时间
  cudaEventRecord(stop);
  // 同步事件
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  // 计算 GPU 计算时间
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaDeviceSynchronize();

  printf("GPU add time is %f ms\n", milliseconds);

  // 将计算结果从 GPU 复制回 CPU
  cudaMemcpy(h_c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost);

  bool verificationFailed = false;
  for (int i = 0; i < N; ++i) {
    if (fabs(h_c[i] - cpu_c[i]) > 1e-6) {
      printf("Result verification failed at element index %d!\n", i);
      verificationFailed = true;
    }
  }
  if (!verificationFailed) {
    printf("Result right\n");
  }

  // 释放 GPU 端内存
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  // 释放 CPU 端内存
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  delete[] cpu_c;
  // 销毁 CUDA 事件
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}