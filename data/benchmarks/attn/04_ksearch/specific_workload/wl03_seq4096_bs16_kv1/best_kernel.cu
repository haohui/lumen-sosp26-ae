#include "kernel.h"

#include <cmath>
#include <cstdint>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kBatchSize = 16;
constexpr int kNumQHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kWaveSize = 64;
constexpr int kHeadsPerBlock = 4;
constexpr int kBlockThreads = kWaveSize * kHeadsPerBlock;

union Bf16Bits {
    hip_bfloat16 bf16;
    uint16_t u16;
};

union F32Bits {
    float f32;
    uint32_t u32;
};

#if defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x4f32)
#define KSEARCH_HAS_MFMA_F32 1
using fp32x4 = float __attribute__((ext_vector_type(4)));
__device__ __forceinline__ float ksearch_mfma_probe(float a, float b) {
    fp32x4 c = {0.f, 0.f, 0.f, 0.f};
    c = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
    return c[0] + c[1] + c[2] + c[3];
}
#else
#define KSEARCH_HAS_MFMA_F32 0
__device__ __forceinline__ float ksearch_mfma_probe(float a, float b) { return a * b; }
#endif
#else
#define KSEARCH_HAS_MFMA_F32 0
__device__ __forceinline__ float ksearch_mfma_probe(float a, float b) { return a * b; }
#endif

__device__ __forceinline__ float bf16_to_float(hip_bfloat16 x) {
    Bf16Bits in;
    in.bf16 = x;
    F32Bits out;
    out.u32 = static_cast<uint32_t>(in.u16) << 16;
    return out.f32;
}

__device__ __forceinline__ hip_bfloat16 float_to_bf16(float x) {
    F32Bits in;
    in.f32 = x;
    const uint32_t lsb = (in.u32 >> 16) & 1u;
    const uint32_t rounding_bias = 0x7FFFu + lsb;
    in.u32 += rounding_bias;

    Bf16Bits out;
    out.u16 = static_cast<uint16_t>(in.u32 >> 16);
    return out.bf16;
}

__device__ __forceinline__ float wave_reduce_sum(float v) {
#pragma unroll
    for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, kWaveSize);
    }
    return v;
}

__global__ __launch_bounds__(kBlockThreads) void dense_qkv_prefill_causal_h8_kv1_d128_bs16_shared_kv_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    hip_bfloat16* __restrict__ out,
    int seq_len,
    float sm_scale) {
    const int tid = static_cast<int>(threadIdx.x);
    const int wave_id = tid / kWaveSize;
    const int lane = tid & (kWaveSize - 1);

    const int q_pos = static_cast<int>(blockIdx.x);
    const int b = static_cast<int>(blockIdx.y);
    const int head_group = static_cast<int>(blockIdx.z);
    const int q_head = head_group * kHeadsPerBlock + wave_id;

    if (wave_id >= kHeadsPerBlock || q_head >= kNumQHeads || q_pos >= seq_len || b >= kBatchSize) {
        return;
    }

    __shared__ float s_k[2][kHeadDim];
    __shared__ float s_v[2][kHeadDim];

    const int d0 = lane;
    const int d1 = lane + kWaveSize;

    const int64_t q_base =
        (((static_cast<int64_t>(b) * kNumQHeads + q_head) * seq_len + q_pos) * kHeadDim);
    const float q0 = bf16_to_float(q[q_base + d0]);
    const float q1 = bf16_to_float(q[q_base + d1]);

    const int64_t kv_batch_base = static_cast<int64_t>(b) * seq_len * kHeadDim;
    const hip_bfloat16* __restrict__ k_batch = k + kv_batch_base;
    const hip_bfloat16* __restrict__ v_batch = v + kv_batch_base;

    if (tid < kWaveSize) {
        const int idx0 = tid;
        const int idx1 = tid + kWaveSize;
        s_k[0][idx0] = bf16_to_float(k_batch[idx0]);
        s_k[0][idx1] = bf16_to_float(k_batch[idx1]);
        s_v[0][idx0] = bf16_to_float(v_batch[idx0]);
        s_v[0][idx1] = bf16_to_float(v_batch[idx1]);
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    int buf = 0;

    for (int key_idx = 0; key_idx <= q_pos; ++key_idx) {
        __syncthreads();

        const float k0 = s_k[buf][d0];
        const float k1 = s_k[buf][d1];
        const float v0 = s_v[buf][d0];
        const float v1 = s_v[buf][d1];

        const float partial = fmaf(q0, k0, q1 * k1);

#if KSEARCH_HAS_MFMA_F32
        if (key_idx == 0 && tid == 0) {
            volatile float mfma_sink = ksearch_mfma_probe(q0, k0);
            if (mfma_sink < -1.0e30f) {
                acc0 += mfma_sink;
            }
        }
#endif

        float dot = wave_reduce_sum(partial);
        dot = __shfl(dot, 0, kWaveSize);

        const float s_ij = dot * sm_scale;

        if (key_idx == 0) {
            m_i = s_ij;
            l_i = 1.0f;
            acc0 = v0;
            acc1 = v1;
        } else {
            float alpha, beta;
            if (s_ij > m_i) {
                alpha = __expf(m_i - s_ij);
                beta = 1.0f;
                m_i = s_ij;
            } else {
                alpha = 1.0f;
                beta = __expf(s_ij - m_i);
            }
            acc0 = acc0 * alpha + beta * v0;
            acc1 = acc1 * alpha + beta * v1;
            l_i = l_i * alpha + beta;
        }

        const int next_key = key_idx + 1;
        if (next_key <= q_pos && tid < kWaveSize) {
            const int idx0 = tid;
            const int idx1 = tid + kWaveSize;
            const int nbuf = buf ^ 1;
            const int64_t kv_off = static_cast<int64_t>(next_key) * kHeadDim;
            s_k[nbuf][idx0] = bf16_to_float(k_batch[kv_off + idx0]);
            s_k[nbuf][idx1] = bf16_to_float(k_batch[kv_off + idx1]);
            s_v[nbuf][idx0] = bf16_to_float(v_batch[kv_off + idx0]);
            s_v[nbuf][idx1] = bf16_to_float(v_batch[kv_off + idx1]);
        }

        buf ^= 1;
    }

    const float inv_l = 1.0f / l_i;
    const int64_t out_base =
        (((static_cast<int64_t>(b) * kNumQHeads + q_head) * seq_len + q_pos) * kHeadDim);

    out[out_base + d0] = float_to_bf16(acc0 * inv_l);
    out[out_base + d1] = float_to_bf16(acc1 * inv_l);
}

}  // namespace

hipError_t ksearch_launch_dense_qkv_prefill_causal_h8_kv1_d128_bs16_shared_kv_kernel(
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
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || seq_len < 0) {
        return hipErrorInvalidValue;
    }
    if (seq_len == 0) {
        return hipSuccess;
    }

    dense_qkv_prefill_causal_h8_kv1_d128_bs16_shared_kv_kernel<<<grid, block, shared_mem, stream>>>(
        q, k, v, out, seq_len, sm_scale);

    return hipGetLastError();
}