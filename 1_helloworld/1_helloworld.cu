#include <iostream>
// __global__声明一个核函数，核函数是运行在gpu设备上的
__global__ void kernel() { printf("hello world!\n"); }

int main(void) {
  /*
  一个核函数对应一个grid里的某一个block的某一个thread
  一个grid有若干个block
  一个block有若干个thread
  */
  /*
  参数1表示gridSize的大小为1，即grid中有1个block
  参数4表示blockSize的大小为4，即每个block中有4个thread
  */
  kernel<<<1, 4>>>();
  cudaDeviceSynchronize();
  return 0;
}