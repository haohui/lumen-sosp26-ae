#ifndef KSEARCH_GEMM_BF16_VAR_MNK_KERNEL_H_
#define KSEARCH_GEMM_BF16_VAR_MNK_KERNEL_H_

#include <cstdint>
#include <hip/hip_runtime.h>

__global__ void gemm_bf16_var_mnk_kernel(
    const uint16_t* A,
    const uint16_t* B,
    uint16_t* C,
    int64_t M,
    int64_t N,
    int64_t K);

__global__ void gemm_bf16_var_mnk_balanced_kernel(
    const uint16_t* A,
    const uint16_t* B,
    uint16_t* C,
    int64_t M,
    int64_t N,
    int64_t K);

hipError_t ksearch_launch_gemm_bf16_var_mnk(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const uint16_t* A,
    const uint16_t* B,
    uint16_t* C,
    int64_t M,
    int64_t N,
    int64_t K);

hipError_t ksearch_launch_gemm_bf16_var_mnk_balanced(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const uint16_t* A,
    const uint16_t* B,
    uint16_t* C,
    int64_t M,
    int64_t N,
    int64_t K);

#endif  // KSEARCH_GEMM_BF16_VAR_MNK_KERNEL_H_