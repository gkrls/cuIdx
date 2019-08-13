/*! @file cuidx.cuh */

// MIT License
//
// Copyright (c) 2019 George Karlos
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#ifndef __CUIDX_CUH__
#define __CUIDX_CUH__

#include <cuda_runtime.h>
#include <cstdint>

#if !defined(__NVCC__) && !defined(__CUDACC__)
    #error "cuIdx must be compiled with NVCC"
#endif
#if !defined(CUDART_VERSION)
    #error "cuIdx requires CUDART_VERSION which is Undefined!"
#elif CUDART_VERSION < 2010
    #error "cuIdx minimum supported Cuda version: 2.1"
#endif

#if (CUDART_VERSION >= 7050)
  #define __C11_SUPPORTED 1
#else
  #define __C11_SUPPORTED 0
#endif

#define WARPSIZE 32

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#define __bsz1 ( blockDim.x )
#define __bsz2 ( blockDim.y * blockDim.x )
#define __bsz3 ( blockDim.z * blockDim.y * blockDim.x )
#define __bid1 ( blockIdx.x )
#define __bid2 ( blockIdx.y * gridDim.x                       \
               + blockIdx.x )                        
#define __bid3 ( blockIdx.z * gridDim.y * gridDim.x           \
               + blockIdx.y * gridDim.x                       \
               + blockIdx.x )
#define __tid1 ( threadIdx.x )
#define __tid2 ( threadIdx.y * blockDim.x                     \
               + threadIdx.x )
#define __tid3 ( threadIdx.z * blockDim.y * blockDim.x        \
               + threadIdx.y * blockDim.x                     \
               + threadIdx.x )

#define __gtid11 ( __bid(1) * __bsz(1) + __tid(1))
#define __gtid12 ( __bid(1) * __bsz(2) + __tid(2))
#define __gtid13 ( __bid(1) * __bsz(3) + __tid(3))
#define __gtid21 ( __bid(2) * __bsz(1) + __tid(1))
#define __gtid22 ( __bid(2) * __bsz(2) + __tid(2))
#define __gtid23 ( __bid(2) * __bsz(3) + __tid(3))
#define __gtid31 ( __bid(3) * __bsz(1) + __tid(1))
#define __gtid32 ( __bid(3) * __bsz(2) + __tid(2))
#define __gtid33 ( __bid(3) * __bsz(3) + __tid(3))

#define __wid1 ( __tid(1) / WARPSIZE )
#define __wid2 ( __tid(2) / WARPSIZE )
#define __wid3 ( __tid(3) / WARPSIZE )

#define __gwid11 ( __gtid(1,1) / WARPSIZE)
#define __gwid12 ( __gtid(1,2) / WARPSIZE)
#define __gwid13 ( __gtid(1,3) / WARPSIZE)
#define __gwid21 ( __gtid(2,1) / WARPSIZE)
#define __gwid22 ( __gtid(2,2) / WARPSIZE)
#define __gwid23 ( __gtid(2,3) / WARPSIZE)
#define __gwid31 ( __gtid(3,1) / WARPSIZE)
#define __gwid32 ( __gtid(3,2) / WARPSIZE)
#define __gwid33 ( __gtid(3,3) / WARPSIZE)

#define __bsz( blockdims)           __bsz##blockdims
#define __bid( griddims )           __bid##griddims 
#define __tid( blockdims)           __tid##blockdims
#define __gtid(griddims, blockdims) __gtid##griddims ##blockdims
#define __wid( blockdims)           __wid##blockdims
#define __gwid(griddims, blockdims) __gwid##griddims ##blockdims

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

namespace cuidx
{

/*! @brief Retrieve the number of threads in the the thread-block the calling 
 *         thread belongs to
 *  @tparam blockdims The number of threadblock dimensions specified on kernel 
 *          launch
 *
 *  <b>Example</b>(<b>2D</b>-Blocks)<br/>
 *  @code
 *   uint32_t block_size = cuidx::bsz<2>();
 *  @endcode
 *
 *  @details <code>cuidx::bsz()</code> (without template arguments) works
 *          for all cases. However, when <code>blockdims < 3</code>, passing
 *          a template argument results in a slightly faster computation.
 */
#if __C11_SUPPORTED
template<int blockdims> __device__ __forceinline__ uint32_t bsz(void) = delete;
#else
template<int blockdims> __device__ __forceinline__ uint32_t bsz(void);
#endif

/*! @brief Retrieve the linear index of the thread block, w.r.t to the grid,
 *         the calling thread belongs to
 *  @tparam griddims The number of grid dimensions specified on kernel launch
 *
 *  <b>Example</b>(<b>2D</b>-Grid)<br/>
 *  @code
 *   uint32_t block_id = cuidx::bid<2>();
 *  @endcode
 *
 *  @details <code>cuidx::bid()</code> (without template arguments) works
 *          for all cases. However, when <code>griddims < 3</code>, passing
 *          a template argument results in a slightly faster computation.
 */
#if __C11_SUPPORTED
template<int griddims> __device__ __forceinline__ uint32_t bid(void) = delete;
#else
template<int griddims> __device__ __forceinline__ uint32_t bid(void);
#endif

/*!
 * @brief Retrieve the linear index of the calling thread w.r.t its thread block.
 *        Unique among threads in the same block.
 * 
 * @tparam blockdims The number of block dimensions specified on kernel launch
 * 
 * <b>Example</b>(<b>2D</b>-Block)<br/>
 * @code
 *   uint32_t thread_id = cuidx::tid<2>();
 * @endcode
 *
 * @details <code>cuidx::tid()</code> (without template arguments) works
 *          for all cases. However, when <code>blockdims < 3</code>, passing
 *          a template argument results in a slightly faster computation.
 */
#if __C11_SUPPORTED
template<int blockdims> __device__ __forceinline__ uint32_t tid(void) = delete;
#else
template<int blockdims> __device__ __forceinline__ uint32_t tid(void);
#endif

/*!
 * @brief Retrieve the global, linear index of the calling thread w.r.t the grid.
 *        Unique among all threads in the grid.
 *
 * @tparam griddims The number of grid dimensions specified on kernel launch
 * @tparam blockdims The number of block dimensions specified on kernel launch
 *
 * <b>Example</b>(<b>2D</b>-Grid, <b>3</b>-Blocks)<br/>
 * @code
 *   uint32_t global_id = cuidx::gid<2,3>();
 * @endcode
 * @details <code>cuidx::gtid()</code> (without template arguments) works
 *          for all cases. However, when <code>blockdims < 3 && griddims < 3</code>, 
 *          passing template arguments results in a slightly faster computation.
 */
#if __C11_SUPPORTED
template<int griddims, int blockdims> __device__ __forceinline__ uint32_t gtid(void) = delete;
#else
template<int griddims, int blockdims> __device__ __forceinline__ uint32_t gtid(void);
#endif

/*!
 * @brief Retrieve the index of the warp the calling thread belongs to, w.r.t to 
 *        warps within the thread's block.
 *
 * @tparam blockdims The number of block dimensions specified on kernel launch
 *
 * <b>Example</b>(<b>2D</b>-Blocks)<br/>
 * @code
 *   uint32_t warp_id = cuidx::wid<2>();
 * @endcode
 * @details <code>cuidx::wid()</code> (without template arguments) works
 *          for all cases. However, when <code>blockdims < 3</code>, 
 *          passing template arguments results in a slightly faster computation.
 */
#if __C11_SUPPORTED
template<int blockdims> __device__ __forceinline__ uint32_t wid(void) = delete;
#else
template<int blockdims> __device__ __forceinline__ uint32_t wid(void);
#endif

/*!
 * @brief Retrieve the global index of the warp the calling thread belongs to, w.r.t to 
 *        all warps in the grid
 *
 * @tparam griddims The number of grid dimensions specified on kernel launch
 * @tparam blockdims The number of block dimensions specified on kernel launch
 *
 * <b>Example</b>(<b>2D</b>-Grid, <b>3</b>-Blocks)<br/>
 * @code
 *   uint32_t gwarp_id = cuidx::gwid<2,3>();
 * @endcode
 * @details <code>cuidx::gwid()</code> (without template arguments) works
 *          for all cases. However, when <code>blockdims < 3 && griddims < 3</code>, 
 *          passing template arguments results in a slightly faster computation.
 */
 #if __C11_SUPPORTED
template<int griddims, int blockdims> __device__ __forceinline__ uint32_t gwid(void) = delete;
#else
template<int griddims, int blockdims> __device__ __forceinline__ uint32_t gwid(void);
#endif

/*!
 * @brief Retrieve the warp lane of the calling thread. i.e. The thread's index
 *        within its warp.
 */
__device__ __forceinline__ uint32_t lid(void);

/*!
 * @brief Retrieve the index of the SM the calling thread is assigned to.
 */
__device__ __forceinline__ uint32_t smid(void);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
__device__ __forceinline__ uint32_t bsz(void);
__device__ __forceinline__ uint32_t bid(void);
__device__ __forceinline__ uint32_t tid(void);
__device__ __forceinline__ uint32_t gtid(void);
__device__ __forceinline__ uint32_t wid(void);
__device__ __forceinline__ uint32_t gwid(void);
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


/* specializations */

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<> __device__ __forceinline__ uint32_t bsz<1>(void);
template<> __device__ __forceinline__ uint32_t bsz<2>(void);
template<> __device__ __forceinline__ uint32_t bsz<3>(void);
template<> __device__ __forceinline__ uint32_t bid<1>(void);
template<> __device__ __forceinline__ uint32_t bid<2>(void);
template<> __device__ __forceinline__ uint32_t bid<3>(void);
template<> __device__ __forceinline__ uint32_t tid<1>(void);
template<> __device__ __forceinline__ uint32_t tid<2>(void);
template<> __device__ __forceinline__ uint32_t tid<3>(void);
template<> __device__ __forceinline__ uint32_t gtid<1, 1>(void);
template<> __device__ __forceinline__ uint32_t gtid<1, 2>(void);
template<> __device__ __forceinline__ uint32_t gtid<1, 3>(void);
template<> __device__ __forceinline__ uint32_t gtid<2, 1>(void);
template<> __device__ __forceinline__ uint32_t gtid<2, 2>(void);
template<> __device__ __forceinline__ uint32_t gtid<2, 3>(void);
template<> __device__ __forceinline__ uint32_t gtid<3, 1>(void);
template<> __device__ __forceinline__ uint32_t gtid<3, 2>(void);
template<> __device__ __forceinline__ uint32_t gtid<3, 3>(void);
template<> __device__ __forceinline__ uint32_t wid<1>(void);
template<> __device__ __forceinline__ uint32_t wid<2>(void);
template<> __device__ __forceinline__ uint32_t wid<3>(void);
template<> __device__ __forceinline__ uint32_t gwid<1, 1>(void);
template<> __device__ __forceinline__ uint32_t gwid<1, 2>(void);
template<> __device__ __forceinline__ uint32_t gwid<1, 3>(void);
template<> __device__ __forceinline__ uint32_t gwid<2, 1>(void);
template<> __device__ __forceinline__ uint32_t gwid<2, 2>(void);
template<> __device__ __forceinline__ uint32_t gwid<2, 3>(void);
template<> __device__ __forceinline__ uint32_t gwid<3, 1>(void);
template<> __device__ __forceinline__ uint32_t gwid<3, 2>(void);
template<> __device__ __forceinline__ uint32_t gwid<3, 3>(void);
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

}

/* implementations */

template<> __device__ __forceinline__ uint32_t cuidx::bsz<1>(void) { return __bsz(1); }
template<> __device__ __forceinline__ uint32_t cuidx::bsz<2>(void) { return __bsz(2); }
template<> __device__ __forceinline__ uint32_t cuidx::bsz<3>(void) { return __bsz(3); }

template<> __device__ __forceinline__ uint32_t cuidx::bid<1>(void) { return __bid(1); }
template<> __device__ __forceinline__ uint32_t cuidx::bid<2>(void) { return __bid(2); }
template<> __device__ __forceinline__ uint32_t cuidx::bid<3>(void) { return __bid(3); }

template<> __device__ __forceinline__ uint32_t cuidx::tid<1>(void) { return __tid(1); }
template<> __device__ __forceinline__ uint32_t cuidx::tid<2>(void) { return __tid(2); }
template<> __device__ __forceinline__ uint32_t cuidx::tid<3>(void) { return __tid(3); }

template<> __device__ __forceinline__ uint32_t cuidx::gtid<1, 1>(void) { return __gtid(1,1); }
template<> __device__ __forceinline__ uint32_t cuidx::gtid<1, 2>(void) { return __gtid(1,2); }
template<> __device__ __forceinline__ uint32_t cuidx::gtid<1, 3>(void) { return __gtid(1,3); }
template<> __device__ __forceinline__ uint32_t cuidx::gtid<2, 1>(void) { return __gtid(2,1); }
template<> __device__ __forceinline__ uint32_t cuidx::gtid<2, 2>(void) { return __gtid(2,2); }
template<> __device__ __forceinline__ uint32_t cuidx::gtid<2, 3>(void) { return __gtid(2,3); }
template<> __device__ __forceinline__ uint32_t cuidx::gtid<3, 1>(void) { return __gtid(3,1); }
template<> __device__ __forceinline__ uint32_t cuidx::gtid<3, 2>(void) { return __gtid(3,2); }
template<> __device__ __forceinline__ uint32_t cuidx::gtid<3, 3>(void) { return __gtid(3,3); }

template<> __device__ __forceinline__ uint32_t cuidx::wid<1>(void) { return __wid(1); }
template<> __device__ __forceinline__ uint32_t cuidx::wid<2>(void) { return __wid(2); }
template<> __device__ __forceinline__ uint32_t cuidx::wid<3>(void) { return __wid(3); }

template<> __device__ __forceinline__ uint32_t cuidx::gwid<1, 1>(void) { return __gwid(1,1); }
template<> __device__ __forceinline__ uint32_t cuidx::gwid<1, 2>(void) { return __gwid(1,2); }
template<> __device__ __forceinline__ uint32_t cuidx::gwid<1, 3>(void) { return __gwid(1,3); }
template<> __device__ __forceinline__ uint32_t cuidx::gwid<2, 1>(void) { return __gwid(2,1); }
template<> __device__ __forceinline__ uint32_t cuidx::gwid<2, 2>(void) { return __gwid(2,2); }
template<> __device__ __forceinline__ uint32_t cuidx::gwid<2, 3>(void) { return __gwid(2,3); }
template<> __device__ __forceinline__ uint32_t cuidx::gwid<3, 1>(void) { return __gwid(3,1); }
template<> __device__ __forceinline__ uint32_t cuidx::gwid<3, 2>(void) { return __gwid(3,2); }
template<> __device__ __forceinline__ uint32_t cuidx::gwid<3, 3>(void) { return __gwid(3,3); }

__device__ __forceinline__ uint32_t cuidx::bsz(void)  { return __bsz(3);     }
__device__ __forceinline__ uint32_t cuidx::bid(void)  { return __bid(3);     }
__device__ __forceinline__ uint32_t cuidx::tid(void)  { return __tid(3);     }
__device__ __forceinline__ uint32_t cuidx::gtid(void) { return __gtid(3, 3); }
__device__ __forceinline__ uint32_t cuidx::wid(void)  { return __wid(3);     }
__device__ __forceinline__ uint32_t cuidx::gwid(void) { return __gwid(3, 3); }

__device__ __forceinline__ uint32_t cuidx::lid(void)
{
  uint32_t lane; 
  asm volatile ("mov.u32 %0, %laneid;" : "=r"(lane));
  return lane;
}

__device__ __forceinline__ uint32_t cuidx::smid(void)
{    
  uint32_t sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  return sm;
}

#endif /* __CUIDX_CUH__ */