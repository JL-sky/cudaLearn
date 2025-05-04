![image-20250504110859399](https://raw.githubusercontent.com/jl-sky/imageDatesets/master/2025/05/upgit_20250504_1746328139.png)

![image-20250504110928503](https://raw.githubusercontent.com/jl-sky/imageDatesets/master/2025/05/upgit_20250504_1746328168.png)



```c++
  // 计算一个grid中的thread数量
  int bSize = blockDim.z * blockDim.y * blockDim.x;
  // 计算相对于grid的全局block索引
  int bIndex =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
  // 计算相对于block的全局thread索引
  int tIndex = threadIdx.z * blockDim.x * blockDim.y +
               threadIdx.y * blockDim.x + threadIdx.x;
  // 计算相对于grid的全局thread索引
  int index = bIndex * bSize + tIndex;
```

