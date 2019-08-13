# cuIdx

cuIdx is a single header CUDA/C++ library that aims to assist with indexing in CUDA kernels.
In particular it provides functions to retrieve:

- The 1D local index of a thread (within its block)
- The 1D global index of a thread
- The 1D local warp index of a thread (within its block)
- The 1D global warp index of a thread
- The the index of a thread w.r.t to it's warp (lane)
- The 1D block size of a thread block

for any number of grid and block dimensions. 

All library functions are meant to be called from within CUDA kernels. 
(Almost) all functions accept template arguments that denote the grid/block 
dimensions as specified on kernel launch. Non-template versions exist and
can cover any grid/block dimensions at the expense of a slightly more costly
computation (as the assume 3 dimensions to cover all cases.)

`cuIdx.cuh` should be compiled with NVCC. Minimum supported CUDA Version: **2.1**

### Documentation
Todo

### Example Usage
```
TODO
```
