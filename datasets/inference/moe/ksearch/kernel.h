#ifndef MOE_FP8_BLOCKSCALE_KERNEL_H_
#define MOE_FP8_BLOCKSCALE_KERNEL_H_

#include <cstdint>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

constexpr int MOE_HIDDEN_SIZE = 7168;
constexpr int MOE_INTERMEDIATE_SIZE = 2048;
constexpr int MOE_INTERMEDIATE2_SIZE = 4096;
constexpr int MOE_NUM_EXPERTS = 32;
constexpr int MOE_TOPK = 4;
constexpr int MOE_HIDDEN_BLOCKS = 56;   // 7168 / 128
constexpr int MOE_INTER_BLOCKS = 16;    // 2048 / 128
constexpr int MOE_INTER2_BLOCKS = 32;   // 4096 / 128
constexpr int MOE_FC1_SCALES_PER_EXPERT = 1792; // 32 * 56
constexpr int MOE_FC2_SCALES_PER_EXPERT = 896;  // 56 * 16
constexpr int MOE_THREADS = 512;

hipError_t ksearch_launch_moe_fp8_blockscale_g1u1_topk4_e32_h7168_i2048_kernel(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const uint8_t* input_q,
    const uint8_t* w1_q,
    const uint8_t* w2_q,
    const float* topk_weights,
    const int32_t* topk_ids,
    const float* input_scale,
    const float* fc1_scale,
    const float* fc2_scale,
    hip_bfloat16* output,
    int seq_len);

#endif // MOE_FP8_BLOCKSCALE_KERNEL_H_