#ifndef KSEARCH_GEMM_BF16_VAR_MNK_BEST_KERNEL_H
#define KSEARCH_GEMM_BF16_VAR_MNK_BEST_KERNEL_H

#include <hip/hip_runtime.h>
#include <cstdint>

void launch_gemm_bf16_var_mnk(
    const void* A,
    const void* B,
    void* C,
    int64_t M,
    int64_t N,
    int64_t K,
    hipStream_t stream);

#endif // KSEARCH_GEMM_BF16_VAR_MNK_BEST_KERNEL_H
