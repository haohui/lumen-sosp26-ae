#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <torch/extension.h>

hipError_t ksearch_launch_gemm_bf16_var_mnk(
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

torch::Tensor run(torch::Tensor A, torch::Tensor B);