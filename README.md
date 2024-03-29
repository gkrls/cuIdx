# cuIdx

cuIdx is a single header CUDA/C++ library that aims to assist with indexing in CUDA kernels.
In particular it provides functions to retrieve:

-   The 1D local index of a thread (within its block)
-   The 1D global index of a thread
-   The 1D local warp index of a thread (within its block)
-   The 1D global warp index of a thread
-   The the index of a thread w.r.t to it's warp (lane)
-   The 1D block size of a thread block

for any number of grid and block dimensions. 

All library functions are meant to be called from within CUDA kernels. 
(Almost) all functions accept template arguments that denote the number of
grid/block dimensions as specified on kernel launch. Non-template versions exist and
can cover any number of grid/block dimensions at the expense of a slightly more costly
computation (3 dimensions are assumed to cover all cases.)

`cuIdx.cuh` should be compiled with NVCC. Minimum supported CUDA Version: **2.1**

## Documentation
Todo

## Example Usage

- Retrieve global thread index
```C++
#include "cuidx.cuh"

using namespace cuidx;

__global__ void vecadd(int *A, int *B, int *C, unsigned len) {
  // auto pos = blockIdx.x * blockDim.x + threadIdx.x;    // vanilla cuda, 1D grid/blocks
  // auto pos = blockIdx.x * blockDim.x * blockDim.y
  //            + threadIdx.y * blockDim.x + threadIdx.x; // vanilla cuda, 1D grid, 2D blocks
  // auto pos = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
  //            + threadIdx.z * blockDim.y * blockDim.x
  //            + threadIdx.y * blockDim.x
  //            + threadIdx.x                             // vanilla cuda, 1D grid, 3D blocks
  
  auto pos = gtid();                                      // <- cuidx
  if (pos < len)
    C[pos] = A[pos] + B[pos];
}

```

- Retrieve global thread index (faster)
```C++
#include "cuidx.cuh"

using namespace cuidx;

__global__ void vecadd(int *A, int *B, int *C, unsigned len) {
  // auto pos = gtid();                                // <- cuidx, assumes 3D grid/blocks
  auto pos = gtid<1,1>();                              // <- cuidx, for 1D grid/blocks (faster)
  if (pos < len)
    C[pos] = A[pos] + B[pos];
}

```

- Warp level
```C++
#include "cuidx.cuh"

using namespace cuidx;

__global__ void kernel(int *A, int *B, int *C, unsigned len) {
  __shared__ int shmem[BLOCK_SIZE / WARPSIZE] = {0};
  int warp_sum = 0;
  ...
  
  __syncwarp();
  
  // vanilla cuda
  auto tid = threadIdx.x * blockDim.x + threadIdx.x; // get tid in 2D block
  if ( tid % 32 == 0)                                // check laneid == 0 (leader)
    shmem[tid / 32] = warp_sum;                      // write at warp index
  
  // cuidx
  if ( wleader())
    shmem[ wid()] = warp_sum;
}

```
