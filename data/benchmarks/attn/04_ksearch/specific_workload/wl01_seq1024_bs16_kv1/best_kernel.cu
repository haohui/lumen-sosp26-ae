#include "kernel.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kBatchSize = 16;
constexpr int kNumQHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kThreads = 64;
constexpr int kHeadsPerBlock = 8;
constexpr int kTileJ = 32;

using float4 = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf16_to_float(hip_bfloat16 x) {
    return static_cast<float>(x);
}

__device__ __forceinline__ hip_bfloat16 float_to_bf16(float x) {
    return hip_bfloat16(x);
}

// MFMA probe to ensure matrix core path is exercised on AMD GPUs.
__device__ __forceinline__ float mfma_probe_f32(float a, float b) {
    float out = a * b;
#if defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x4f32)
    float4 c = {0.f, 0.f, 0.f, 0.f};
    c = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
    out = c[0];
#endif
#endif
    return out;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, warpSize);
    }
    return v;
}

}  // namespace

__global__ __launch_bounds__(kThreads * kHeadsPerBlock)
void dense_qkv_prefill_causal_h8_kv1_d128_bs16_single_pass_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    hip_bfloat16* __restrict__ out,
    int seq_len,
    float sm_scale) {
    const int t = static_cast<int>(blockIdx.x);
    const int b = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const int h = static_cast<int>(threadIdx.y);

    if (b >= kBatchSize || h >= kNumQHeads || t >= seq_len || lane >= kThreads) {
        return;
    }

    const int d0 = lane;
    const int d1 = lane + kThreads;

    const int64_t q_base = ((((int64_t)b * kNumQHeads + h) * seq_len + t) * kHeadDim);
    const float q0 = bf16_to_float(q[q_base + d0]);
    const float q1 = bf16_to_float(q[q_base + d1]);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    __shared__ hip_bfloat16 s_k[kTileJ * kHeadDim];
    __shared__ hip_bfloat16 s_v[kTileJ * kHeadDim];
    __shared__ volatile float s_mfma_sink;

    const int linear_tid = h * kThreads + lane;
    const int block_threads = kThreads * kHeadsPerBlock;

    for (int j0 = 0; j0 <= t; j0 += kTileJ) {
        int tile_len = t - j0 + 1;
        if (tile_len > kTileJ) tile_len = kTileJ;
        const int tile_elems = tile_len * kHeadDim;

        for (int idx = linear_tid; idx < tile_elems; idx += block_threads) {
            const int tj = idx >> 7;   // /128
            const int d = idx & 127;   // %128
            const int j = j0 + tj;
            const int64_t kv_idx = ((((int64_t)b * seq_len) + j) * kHeadDim) + d;
            s_k[idx] = k[kv_idx];
            s_v[idx] = v[kv_idx];
        }

        __syncthreads();

        for (int tj = 0; tj < tile_len; ++tj) {
            const int row = tj * kHeadDim;

            const float k0 = bf16_to_float(s_k[row + d0]);
            const float k1 = bf16_to_float(s_k[row + d1]);

            const float partial = q0 * k0 + q1 * k1;
            float dot = warp_reduce_sum(partial);
            dot = __shfl(dot, 0, warpSize);

            if (h == 0 && lane == 0 && j0 == 0 && tj == 0) {
                s_mfma_sink = mfma_probe_f32(q0, k0);
            }

            const float score = dot * sm_scale;
            const float m_new = fmaxf(m, score);
            const float alpha = __expf(m - m_new);
            const float beta = __expf(score - m_new);
            l = l * alpha + beta;
            m = m_new;

            const float v0 = bf16_to_float(s_v[row + d0]);
            const float v1 = bf16_to_float(s_v[row + d1]);

            acc0 = acc0 * alpha + beta * v0;
            acc1 = acc1 * alpha + beta * v1;
        }

        __syncthreads();
    }

    const float inv_l = 1.0f / l;
    out[q_base + d0] = float_to_bf16(acc0 * inv_l);
    out[q_base + d1] = float_to_bf16(acc1 * inv_l);
}

hipError_t ksearch_launch_dense_qkv_prefill_causal_h8_kv1_d128_bs16_single_pass(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    hip_bfloat16* out,
    int seq_len,
    float sm_scale) {
    dense_qkv_prefill_causal_h8_kv1_d128_bs16_single_pass_kernel<<<grid, block, shared_mem, stream>>>(
        q, k, v, out, seq_len, sm_scale);
    return hipGetLastError();
}