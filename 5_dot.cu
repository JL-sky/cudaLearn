#include <cuda_runtime.h>
#include <stdio.h>

#include <chrono>
#include <cmath>
#include <iostream>

#include "utils.hpp"

constexpr int N = 32 * 1024;

// GPU 向量点积核函数
template <int THREADPERBLOCK>
__global__ void kernel_dot(double* a, double* b, double* c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // 共享内存是block级的，每一个block内部的所有thread通过共享内存进行通信，因此其大小应该是block的大小
  __shared__ double cache[THREADPERBLOCK];
  int cacheIndex = threadIdx.x;
  double tmp = 0;
  while (tid < N) {
    tmp += a[tid] * b[tid];
    /*
    当线程数小于元素个数时，每个元素不足以分配给一个线程，此时这段代码就会起作用
    假设：
        向量长度 N = 16。
        每个线程块的线程数量 blockDim.x = 4，网格中线程块的数量 gridDim.x = 2
        那么总的线程数量为 blockDim.x * gridDim.x = 8
        每个线程的初始tid分别为0、1、2、3、4、5、6、7

    则对于线程 0：
    1.初始 tid = 0，计算 temp = a[0] * b[0]。
    2.tid 增加 blockDim.x * gridDim.x =8，tid 变为 8，计算 temp += a[8] * b[8]。
    3.tid 再次增加 8，变为 16，此时 tid>= N，循环结束。
    */
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheIndex] = tmp;
  /*
  CUDA 中的同步函数，它会阻塞当前线程块中的所有线程，
  直到该线程块中的所有线程都执行到这一行代码
  确保所有线程都已经将数据从全局内存加载到共享内存中，避免后续操作出现数据不一致的问题。
  */
  __syncthreads();

  for (int index = blockDim.x / 2; index > 0; index >>= 1) {
    if (cacheIndex < index) {
      cache[cacheIndex] += cache[cacheIndex + index];
    }
    __syncthreads();
  }
  // 把每个块的结果存放到块的第一个线程索引中
  if (cacheIndex == 0) {
    c[blockIdx.x] = cache[0];
  }
}

// 检验计算结果的正确性
bool CheckResult(double* out, double groundtruth, int num_blocks) {
  double res = 0;
  for (int i = 0; i < num_blocks; ++i) {
    res += out[i];  // 累加所有block的中间结果
  }
  return fabs(res - groundtruth) < 1e-6;
}
int main() {
  double *h_a, *h_b, *h_c;
  double *dev_a, *dev_b, *dev_c;

  h_a = new double[N];
  h_b = new double[N];
  h_c = new double[N];

  CUDA_CHECK(cudaMalloc(&dev_a, N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dev_b, N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dev_c, N * sizeof(double)));

  double groundtruth = 0.0f;
  for (int i = 0; i < N; ++i) {
    h_a[i] = static_cast<double>(i);
    h_b[i] = static_cast<double>(i * 2);
    groundtruth += h_a[i] * h_b[i];
  }

  CUDA_CHECK(
      cudaMemcpy(dev_a, h_a, N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(dev_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice));

  constexpr int threadsPerBlock = 256;
  int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(numBlocks);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  kernel_dot<threadsPerBlock><<<grid, threadsPerBlock>>>(dev_a, dev_b, dev_c);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU dot product time: %f ms\n", milliseconds);

  cudaMemcpy(h_c, dev_c, numBlocks * sizeof(double), cudaMemcpyDeviceToHost);

  bool isCorrect = CheckResult(h_c, groundtruth, numBlocks);
  if (isCorrect) {
    printf("Result is correct!\n");
  } else {
    printf("Result is incorrect.\n");
    printf("Ground truth: %f\n", groundtruth);
  }

  // 清理资源
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}