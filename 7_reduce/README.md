# reduce_v0

## 算法解读

每个线程处理相邻的两个元素

## 示例说明

假设 `blockDim.x = 8`，`cache_index` 是当前线程对应的共享内存索引，并且 `shmem` 数组初始值为 `[1, 2, 3, 4, 5, 6, 7, 8]`。

#### 第一次循环：`index = 1`

- `index * 2 = 2`
- 满足`cache_index % 2 == 0`的线程会执行加法操作：
  - `cache_index = 0`：`shmem[0] = shmem[0] + shmem[1] = 1 + 2 = 3`
  - `cache_index = 2`：`shmem[2] = shmem[2] + shmem[3] = 3 + 4 = 7`
  - `cache_index = 4`：`shmem[4] = shmem[4] + shmem[5] = 5 + 6 = 11`
  - `cache_index = 6`：`shmem[6] = shmem[6] + shmem[7] = 7 + 8 = 15`
- 此时 `shmem` 数组变为 `[3, 2, 7, 4, 11, 6, 15, 8]`

#### 第二次循环：`index = 2`

- `index * 2 = 4`
- 满足`cache_index % 4 == 0`的线程会执行加法操作：
  - `cache_index = 0`：`shmem[0] = shmem[0] + shmem[2] = 3 + 7 = 10`
  - `cache_index = 4`：`shmem[4] = shmem[4] + shmem[6] = 11 + 15 = 26`
- 此时 `shmem` 数组变为 `[10, 2, 7, 4, 26, 6, 15, 8]`

#### 第三次循环：`index = 4`

- `index * 2 = 8`
- 满足`cache_index % 8 == 0`的线程会执行加法操作：
  - `cache_index = 0`：`shmem[0] = shmem[0] + shmem[4] = 10 + 26 = 36`
- 此时 `shmem` 数组变为 `[36, 2, 7, 4, 26, 6, 15, 8]`

最终，`shmem[0]` 中存储的是数组中所有元素的和。

## 算法缺陷

