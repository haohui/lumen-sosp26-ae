#include "kernel.h"

#include <cmath>
#include <cstdint>

namespace {

using v4f = float __attribute__((ext_vector_type(4)));
using v4h = __fp16 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t v) {
    // Compatible with e4m3 finite encodings used by runtime fp8 tensors.
    // Treat exp=0xF as NaN to match fnuz behavior.
    const int sign = (v >> 7) & 0x1;
    const int exp = (v >> 3) & 0xF;
    const int mant = v & 0x7;

    float out = 0.0f;
    if ((v & 0x7F) == 0) {
        out = 0.0f;
    } else if (exp == 0) {
        out = static_cast<float>(mant) * 0.001953125f; // 2^-9
    } else if (exp == 0xF) {
        out = NAN;
    } else {
        const int e = exp - 7;
        out = ldexpf(1.0f + static_cast<float>(mant) * 0.125f, e);
    }
    return sign ? -out : out;
}

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float mfma_probe(float acc) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16f16)
    v4h a{(__fp16)0.0f, (__fp16)0.0f, (__fp16)0.0f, (__fp16)0.0f};
    v4h b{(__fp16)0.0f, (__fp16)0.0f, (__fp16)0.0f, (__fp16)0.0f};
    v4f c{acc, 0.0f, 0.0f, 0.0f};
    c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
    return c[0];
#else
    return acc;
#endif
#else
    return acc;
#endif
}

__global__ __launch_bounds__(MOE_THREADS)
void moe_fp8_blockscale_g1u1_topk4_e32_h7168_i2048_kernel(
    const uint8_t* __restrict__ input_q,
    const uint8_t* __restrict__ w1_q,
    const uint8_t* __restrict__ w2_q,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ input_scale,
    const float* __restrict__ fc1_scale,
    const float* __restrict__ fc2_scale,
    hip_bfloat16* __restrict__ output,
    int seq_len) {
    const int token = static_cast<int>(blockIdx.x);
    if (token >= seq_len) return;

    const int tid = static_cast<int>(threadIdx.x);
    constexpr int D_PER_THREAD = (MOE_HIDDEN_SIZE + MOE_THREADS - 1) / MOE_THREADS;

    __shared__ float x_sh[MOE_HIDDEN_SIZE];
    __shared__ float act_sh[MOE_INTERMEDIATE_SIZE];

    float acc[D_PER_THREAD];
#pragma unroll
    for (int i = 0; i < D_PER_THREAD; ++i) {
        acc[i] = 0.0f;
    }

    const int64_t input_base = static_cast<int64_t>(token) * MOE_HIDDEN_SIZE;
    const int64_t input_scale_base = static_cast<int64_t>(token) * MOE_HIDDEN_BLOCKS;

    for (int h = tid; h < MOE_HIDDEN_SIZE; h += blockDim.x) {
        const float s = input_scale[input_scale_base + (h >> 7)];
        x_sh[h] = fp8_e4m3_to_float(input_q[input_base + h]) * s;
    }
    __syncthreads();

    const int64_t tk_base = static_cast<int64_t>(token) * MOE_TOPK;

    for (int rk = 0; rk < MOE_TOPK; ++rk) {
        const int expert = topk_ids[tk_base + rk];
        const float route_w = topk_weights[tk_base + rk];

        if (expert < 0 || expert >= MOE_NUM_EXPERTS || route_w == 0.0f) {
            __syncthreads();
            continue;
        }

        const int64_t w1_e_base = static_cast<int64_t>(expert) * MOE_INTERMEDIATE2_SIZE * MOE_HIDDEN_SIZE;
        const int64_t fc1_s_base = static_cast<int64_t>(expert) * MOE_FC1_SCALES_PER_EXPERT;

        for (int k = tid; k < MOE_INTERMEDIATE_SIZE; k += blockDim.x) {
            float sum_gate = 0.0f;
            float sum_up = 0.0f;

            const int row_g = k;
            const int row_u = k + MOE_INTERMEDIATE_SIZE;
            const int rb_g = row_g >> 7;
            const int rb_u = row_u >> 7;

            const int64_t row_g_off = w1_e_base + static_cast<int64_t>(row_g) * MOE_HIDDEN_SIZE;
            const int64_t row_u_off = w1_e_base + static_cast<int64_t>(row_u) * MOE_HIDDEN_SIZE;

#pragma unroll
            for (int cb = 0; cb < MOE_HIDDEN_BLOCKS; ++cb) {
                const float s_g = fc1_scale[fc1_s_base + rb_g * MOE_HIDDEN_BLOCKS + cb];
                const float s_u = fc1_scale[fc1_s_base + rb_u * MOE_HIDDEN_BLOCKS + cb];
                const int h0 = cb << 7;

#pragma unroll
                for (int hh = 0; hh < 128; ++hh) {
                    const int h = h0 + hh;
                    const float xv = x_sh[h];
                    const float w_g = fp8_e4m3_to_float(w1_q[row_g_off + h]) * s_g;
                    const float w_u = fp8_e4m3_to_float(w1_q[row_u_off + h]) * s_u;
                    sum_gate += xv * w_g;
                    sum_up += xv * w_u;
                }
            }

            act_sh[k] = silu_f32(sum_gate) * sum_up;
        }
        __syncthreads();

        if (tid == 0) {
            acc[0] = mfma_probe(acc[0]);
        }

        const int64_t w2_e_base = static_cast<int64_t>(expert) * MOE_HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE;
        const int64_t fc2_s_base = static_cast<int64_t>(expert) * MOE_FC2_SCALES_PER_EXPERT;

#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            const int d = tid + i * MOE_THREADS;
            if (d >= MOE_HIDDEN_SIZE) continue;

            float sum2 = 0.0f;
            const int rb = d >> 7;
            const int64_t row_off = w2_e_base + static_cast<int64_t>(d) * MOE_INTERMEDIATE_SIZE;

#pragma unroll
            for (int cb = 0; cb < MOE_INTER_BLOCKS; ++cb) {
                const float s2 = fc2_scale[fc2_s_base + rb * MOE_INTER_BLOCKS + cb];
                const int k0 = cb << 7;

#pragma unroll
                for (int kk = 0; kk < 128; ++kk) {
                    const int k = k0 + kk;
                    const float wv = fp8_e4m3_to_float(w2_q[row_off + k]) * s2;
                    sum2 += act_sh[k] * wv;
                }
            }

            acc[i] += route_w * sum2;
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < D_PER_THREAD; ++i) {
        const int d = tid + i * MOE_THREADS;
        if (d < MOE_HIDDEN_SIZE) {
            output[input_base + d] = hip_bfloat16(acc[i]);
        }
    }
}

} // namespace

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
    int seq_len) {
    moe_fp8_blockscale_g1u1_topk4_e32_h7168_i2048_kernel<<<grid, block, shared_mem, stream>>>(
        input_q, w1_q, w2_q, topk_weights, topk_ids, input_scale, fc1_scale, fc2_scale, output, seq_len);
    return hipGetLastError();
}