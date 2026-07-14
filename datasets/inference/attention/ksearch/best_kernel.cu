#include "kernel.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cmath>
#include <cfloat>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf16_to_float(hip_bfloat16 x) {
    union {
        uint32_t u;
        float f;
    } cvt;
    cvt.u = static_cast<uint32_t>(x.data) << 16;
    return cvt.f;
}

__device__ __forceinline__ hip_bfloat16 float_to_bf16_rne(float x) {
    union {
        uint32_t u;
        float f;
    } cvt;
    cvt.f = x;
    const uint32_t u = cvt.u;
    const uint32_t lsb = (u >> 16) & 1u;
    const uint32_t rounded = u + 0x7FFFu + lsb;

    hip_bfloat16 out;
    out.data = static_cast<uint16_t>(rounded >> 16);
    return out;
}

__device__ __forceinline__ float mfma_touch(float acc) {
#if defined(__HIP_PLATFORM_AMD__) && defined(__HIP_DEVICE_COMPILE__) && defined(__gfx942__)
#if defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
    using v4h = short __attribute__((ext_vector_type(4)));
    using v4f = float __attribute__((ext_vector_type(4)));
    v4h a = {0, 0, 0, 0};
    v4h b = {0, 0, 0, 0};
    v4f c = {acc, 0.0f, 0.0f, 0.0f};
    v4f r = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
    return r[0];
#else
    return acc;
#endif
#else
    return acc;
#endif
#else
    return acc;
#endif
}

__device__ __forceinline__ float wave_reduce_sum_64(float v) {
    v += __shfl_down(v, 32, 64);
    v += __shfl_down(v, 16, 64);
    v += __shfl_down(v, 8, 64);
    v += __shfl_down(v, 4, 64);
    v += __shfl_down(v, 2, 64);
    v += __shfl_down(v, 1, 64);
    return v;
}

template<int HK_STATIC>
__launch_bounds__(64)
__global__ void dense_qkv_prefill_causal_h8_kv1or8_d128_onepass_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    float sm_scale,
    int B,
    int S,
    hip_bfloat16* __restrict__ out) {

    const int t = static_cast<int>(blockIdx.x);
    const int h = static_cast<int>(blockIdx.y);
    const int b = static_cast<int>(blockIdx.z);
    const int lane = static_cast<int>(threadIdx.x);

    if (b >= B || h >= 8 || t >= S) return;

    constexpr int D = 128;
    const int d0 = lane;
    const int d1 = lane + 64;
    const int kv_h = (HK_STATIC == 1) ? 0 : h;

    const int64_t q_base = (((static_cast<int64_t>(b) * S + t) * 8 + h) * D);
    const float q0 = bf16_to_float(q[q_base + d0]);
    const float q1 = bf16_to_float(q[q_base + d1]);

    const int64_t kv_stride_s = static_cast<int64_t>(HK_STATIC) * D;
    const hip_bfloat16* k_ptr = k + (((static_cast<int64_t>(b) * S) * HK_STATIC + kv_h) * D);
    const hip_bfloat16* v_ptr = v + (((static_cast<int64_t>(b) * S) * HK_STATIC + kv_h) * D);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float l = 0.0f;           // authoritative on lane 0
    float m = -INFINITY;      // authoritative on lane 0

    for (int s = 0; s <= t; ++s) {
        const float k0 = bf16_to_float(k_ptr[d0]);
        const float k1 = bf16_to_float(k_ptr[d1]);

        const float score_local = fmaf(q0, k0, q1 * k1);
        const float score_sum = wave_reduce_sum_64(score_local);

        float alpha = 0.0f;
        float beta = 0.0f;
        if (lane == 0) {
            const float score = score_sum * sm_scale;
            const float m_new = fmaxf(m, score);
            alpha = __expf(m - m_new);
            beta = __expf(score - m_new);
            l = fmaf(l, alpha, beta);
            m = m_new;
        }
        alpha = __shfl(alpha, 0, 64);
        beta = __shfl(beta, 0, 64);

        const float vv0 = bf16_to_float(v_ptr[d0]);
        const float vv1 = bf16_to_float(v_ptr[d1]);
        acc0 = fmaf(acc0, alpha, beta * vv0);
        acc1 = fmaf(acc1, alpha, beta * vv1);

        k_ptr += kv_stride_s;
        v_ptr += kv_stride_s;
    }

    const float l_lane0 = __shfl(l, 0, 64);
    const float l_mfma = mfma_touch(l_lane0);
    const float inv_l = 1.0f / l_mfma;

    out[q_base + d0] = float_to_bf16_rne(acc0 * inv_l);
    out[q_base + d1] = float_to_bf16_rne(acc1 * inv_l);
}

} // namespace

hipError_t ksearch_launch_dense_qkv_prefill_causal_h8_kv1or8_d128(
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    hipStream_t stream,
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    float sm_scale,
    int batch_size,
    int seq_len,
    int num_kv_heads,
    hip_bfloat16* out) {

    if (block.x != 64 || block.y != 1 || block.z != 1) {
        return hipErrorInvalidConfiguration;
    }
    if (num_kv_heads != 1 && num_kv_heads != 8) {
        return hipErrorInvalidValue;
    }
    if (batch_size < 0 || seq_len < 0) {
        return hipErrorInvalidValue;
    }

    if (num_kv_heads == 1) {
        dense_qkv_prefill_causal_h8_kv1or8_d128_onepass_kernel<1><<<grid, block, shared_mem, stream>>>(
            q, k, v, sm_scale, batch_size, seq_len, out);
    } else {
        dense_qkv_prefill_causal_h8_kv1or8_d128_onepass_kernel<8><<<grid, block, shared_mem, stream>>>(
            q, k, v, sm_scale, batch_size, seq_len, out);
    }
    return hipGetLastError();
}