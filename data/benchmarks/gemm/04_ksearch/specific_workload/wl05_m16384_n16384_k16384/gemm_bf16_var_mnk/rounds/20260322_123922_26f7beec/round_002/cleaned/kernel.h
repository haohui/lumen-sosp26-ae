#ifndef GEMM_BF16_VAR_MNK_KERNEL_H
#define GEMM_BF16_VAR_MNK_KERNEL_H

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

hipError_t ksearch_launch_gemm_bf16_balanced(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const hip_bfloat16* A,
    const hip_bfloat16* B,
    hip_bfloat16* C,
    int M,
    int N,
    int K);

#endif  // GEMM_BF16_VAR_MNK_KERNEL_H