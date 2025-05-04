#include <cuda.h>
#include <cuda_runtime.h>

#include "../utils.hpp"

template <int BlockSize>
__global__ void reduce_v0(float* d_in, float* d_out) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int cache_index = threadIdx.x;
  __shared__ float shmem[BlockSize];
  // 把每个block的中线程的数据拷贝到block中
  shmem[cache_index] = d_in[tid];
  __syncthreads();
  for (int index = 1; index < blockDim.x; index <<= 1) {
    // 改进点：替换为位运算
    if ((cache_index & (2 * index - 1)) == 0)
      shmem[cache_index] += shmem[cache_index + index];
    __syncthreads();
  }
  if (cache_index == 0) d_out[blockIdx.x] = shmem[0];
}

bool CheckResult(float* out, float groudtruth, int n) {
  float res = 0;
  for (int i = 0; i < n; i++) {
    res += out[i];
  }
  if (res != groudtruth) {
    return false;
  }
  return true;
}

int main() {
  constexpr int device_index = 0;
  cudaSetDevice(device_index);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_index);
  constexpr int N = 25600000;
  constexpr int block_size = 256;
  int grid_size =
      std::min((N + block_size - 1) / block_size, device_prop.maxGridSize[0]);

  float* in = (float*)malloc(N * sizeof(float));
  float* dev_in;
  CUDA_CHECK(cudaMalloc(&dev_in, N * sizeof(float)));
  float* out = (float*)malloc(grid_size * sizeof(float));
  float* dev_out;
  CUDA_CHECK(cudaMalloc(&dev_out, grid_size * sizeof(float)));
  for (int i = 0; i < N; ++i) {
    in[i] = 1.0f;
  }
  CUDA_CHECK(cudaMemcpy(dev_in, in, N * sizeof(float), cudaMemcpyHostToDevice));
  dim3 block(block_size);
  dim3 grid(grid_size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  reduce_v0<block_size><<<grid, block>>>(dev_in, dev_out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_time = 0;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  CUDA_CHECK(cudaMemcpy(out, dev_out, grid_size * sizeof(float),
                        cudaMemcpyDeviceToHost));
  printf("allocated %d blocks,data counts are %d\n", block_size, N);

  float groudtruth = N * 1.0f;
  bool is_right = CheckResult(out, groudtruth, grid_size);
  if (is_right) {
    printf("the ans is right\n");
  } else {
    printf("the ans is wrong\n");
    // for(int i = 0; i < GridSize;i++){
    // printf("res per block : %lf ",out[i]);
    //}
    // printf("\n");
    printf("groudtruth is: %f \n", groudtruth);
  }
  printf("reduce_v1 latency = %f ms\n", elapsed_time);
  cudaFree(dev_in);
  cudaFree(dev_out);
  free(in);
  free(out);
  return 0;
}