#include "kernel.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using v4f = float __attribute__((ext_vector_type(4)));
using v4h = __fp16 __attribute__((ext_vector_type(4)));

__device__ __constant__ float kFp8E4M3FnuzLut[256];

__host__ inline float host_fp8_e4m3fnuz_to_float(uint8_t v) {
    const int sign = (v >> 7) & 0x1;
    const int exp = (v >> 3) & 0xF;
    const int mant = v & 0x7;

    float out = 0.0f;
    if ((v & 0x7F) == 0) {
        out = 0.0f;
    } else if (exp == 0) {
        out = static_cast<float>(mant) * 0.001953125f; // 2^-9
    } else if (exp == 0xF) {
        out = std::numeric_limits<float>::quiet_NaN();
    } else {
        const int e = exp - 7;
        out = std::ldexp(1.0f + static_cast<float>(mant) * 0.125f, e);
    }
    return sign ? -out : out;
}

hipError_t ensure_fp8_lut_initialized() {
    static bool initialized = false;
    if (initialized) return hipSuccess;

    float lut[256];
    for (int i = 0; i < 256; ++i) {
        lut[i] = host_fp8_e4m3fnuz_to_float(static_cast<uint8_t>(i));
    }

    hipError_t st = hipMemcpyToSymbol(HIP_SYMBOL(kFp8E4M3FnuzLut), lut, sizeof(lut), 0, hipMemcpyHostToDevice);
    if (st == hipSuccess) {
        initialized = true;
    }
    return st;
}

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t v) {
    return kFp8E4M3FnuzLut[v];
}

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
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
    __shared__ float in_scale_sh[MOE_HIDDEN_BLOCKS];
    __shared__ float fc1_scale_sh[MOE_FC1_SCALES_PER_EXPERT];
    __shared__ float fc2_scale_sh[MOE_FC2_SCALES_PER_EXPERT];
    __shared__ int topk_id_sh[MOE_TOPK];
    __shared__ float topk_w_sh[MOE_TOPK];

    float acc[D_PER_THREAD];
#pragma unroll
    for (int i = 0; i < D_PER_THREAD; ++i) {
        acc[i] = 0.0f;
    }

    const int64_t input_base = static_cast<int64_t>(token) * MOE_HIDDEN_SIZE;
    const int64_t input_scale_base = static_cast<int64_t>(token) * MOE_HIDDEN_BLOCKS;

    for (int i = tid; i < MOE_HIDDEN_BLOCKS; i += blockDim.x) {
        in_scale_sh[i] = input_scale[input_scale_base + i];
    }
    __syncthreads();

    for (int h = tid; h < MOE_HIDDEN_SIZE; h += blockDim.x) {
        const float s = in_scale_sh[h >> 7];
        x_sh[h] = fp8_e4m3_to_float(input_q[input_base + h]) * s;
    }

    const int64_t tk_base = static_cast<int64_t>(token) * MOE_TOPK;
    if (tid < MOE_TOPK) {
        topk_id_sh[tid] = topk_ids[tk_base + tid];
        topk_w_sh[tid] = topk_weights[tk_base + tid];
    }
    __syncthreads();

    for (int rk = 0; rk < MOE_TOPK; ++rk) {
        const int expert = topk_id_sh[rk];
        const float route_w = topk_w_sh[rk];
        const bool route_valid = (expert >= 0 && expert < MOE_NUM_EXPERTS && route_w != 0.0f);

        if (route_valid) {
            const int64_t fc1_s_base = static_cast<int64_t>(expert) * MOE_FC1_SCALES_PER_EXPERT;
            const int64_t fc2_s_base = static_cast<int64_t>(expert) * MOE_FC2_SCALES_PER_EXPERT;

            for (int i = tid; i < MOE_FC1_SCALES_PER_EXPERT; i += blockDim.x) {
                fc1_scale_sh[i] = fc1_scale[fc1_s_base + i];
            }
            for (int i = tid; i < MOE_FC2_SCALES_PER_EXPERT; i += blockDim.x) {
                fc2_scale_sh[i] = fc2_scale[fc2_s_base + i];
            }
        }
        __syncthreads();

        if (route_valid) {
            const int64_t w1_e_base = static_cast<int64_t>(expert) * MOE_INTERMEDIATE2_SIZE * MOE_HIDDEN_SIZE;

            for (int k = tid; k < MOE_INTERMEDIATE_SIZE; k += blockDim.x) {
                float sum_gate = 0.0f;
                float sum_up = 0.0f;

                const int row_g = k;
                const int row_u = k + MOE_INTERMEDIATE_SIZE;
                const int rb_g = row_g >> 7;
                const int rb_u = row_u >> 7;

                const int64_t row_g_off = w1_e_base + static_cast<int64_t>(row_g) * MOE_HIDDEN_SIZE;
                const int64_t row_u_off = w1_e_base + static_cast<int64_t>(row_u) * MOE_HIDDEN_SIZE;
                const uint8_t* __restrict__ row_g_ptr = w1_q + row_g_off;
                const uint8_t* __restrict__ row_u_ptr = w1_q + row_u_off;

                const float* __restrict__ s_g_row = fc1_scale_sh + rb_g * MOE_HIDDEN_BLOCKS;
                const float* __restrict__ s_u_row = fc1_scale_sh + rb_u * MOE_HIDDEN_BLOCKS;

                for (int cb = 0; cb < MOE_HIDDEN_BLOCKS; ++cb) {
                    const float s_g = s_g_row[cb];
                    const float s_u = s_u_row[cb];
                    const int h0 = cb << 7;

#pragma unroll 8
                    for (int hh = 0; hh < 128; ++hh) {
                        const int h = h0 + hh;
                        const float xv = x_sh[h];
                        const float w_g = fp8_e4m3_to_float(row_g_ptr[h]) * s_g;
                        const float w_u = fp8_e4m3_to_float(row_u_ptr[h]) * s_u;
                        sum_gate = fmaf(xv, w_g, sum_gate);
                        sum_up = fmaf(xv, w_u, sum_up);
                    }
                }

                act_sh[k] = silu_f32(sum_gate) * sum_up;
            }
        }

        __syncthreads();

        if (route_valid) {
            const int64_t w2_e_base = static_cast<int64_t>(expert) * MOE_HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE;

#pragma unroll
            for (int i = 0; i < D_PER_THREAD; ++i) {
                const int d = tid + i * MOE_THREADS;
                if (d >= MOE_HIDDEN_SIZE) continue;

                float sum2 = 0.0f;
                const int rb = d >> 7;
                const int64_t row_off = w2_e_base + static_cast<int64_t>(d) * MOE_INTERMEDIATE_SIZE;
                const uint8_t* __restrict__ row_ptr = w2_q + row_off;
                const float* __restrict__ fc2_row_scale = fc2_scale_sh + rb * MOE_INTER_BLOCKS;

#pragma unroll
                for (int cb = 0; cb < MOE_INTER_BLOCKS; ++cb) {
                    const float s2 = fc2_row_scale[cb];
                    const int k0 = cb << 7;

#pragma unroll 8
                    for (int kk = 0; kk < 128; ++kk) {
                        const int k = k0 + kk;
                        const float wv = fp8_e4m3_to_float(row_ptr[k]) * s2;
                        sum2 = fmaf(act_sh[k], wv, sum2);
                    }
                }

                acc[i] = fmaf(route_w, sum2, acc[i]);
            }

            if (tid == 0) {
                acc[0] = mfma_probe(acc[0]);
            }
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
    hipError_t st = ensure_fp8_lut_initialized();
    if (st != hipSuccess) return st;

    moe_fp8_blockscale_g1u1_topk4_e32_h7168_i2048_kernel<<<grid, block, shared_mem, stream>>>(
        input_q, w1_q, w2_q, topk_weights, topk_ids, input_scale, fc1_scale, fc2_scale, output, seq_len);
    return hipGetLastError();
}