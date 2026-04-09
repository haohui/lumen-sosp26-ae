#ifndef KERNEL_H_
#define KERNEL_H_

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

constexpr int GEMM_BLOCK_M = 64;
constexpr int GEMM_BLOCK_N = 64;
constexpr int GEMM_BLOCK_K = 16;
constexpr int GEMM_THREADS_X = 16;
constexpr int GEMM_THREADS_Y = 16;

__global__ void gemm_bf16_var_mnk_kernel(const __hip_bfloat16* A,
                                         const __hip_bfloat16* B,
                                         __hip_bfloat16* C,
                                         int M,
                                         int N,
                                         int K);

hipError_t ksearch_launch_gemm_bf16_var_mnk(dim3 grid,
                                            dim3 block,
                                            size_t shared_mem,
                                            hipStream_t stream,
                                            const __hip_bfloat16* A,
                                            const __hip_bfloat16* B,
                                            __hip_bfloat16* C,
                                            int M,
                                            int N,
                                            int K);

#endif  // KERNEL_H_