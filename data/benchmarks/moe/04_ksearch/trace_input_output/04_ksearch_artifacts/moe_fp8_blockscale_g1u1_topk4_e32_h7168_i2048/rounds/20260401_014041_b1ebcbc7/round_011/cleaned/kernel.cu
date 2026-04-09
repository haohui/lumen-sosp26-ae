#include "kernel.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using v4f = float __attribute__((ext_vector_type(4)));
using v4h = __fp16 __attribute__((ext_vector_type(4)));

static_assert(MOE_HIDDEN_SIZE % MOE_THREADS == 0, "MOE_HIDDEN_SIZE must be divisible by MOE_THREADS");
static_assert(MOE_INTERMEDIATE_SIZE % MOE_THREADS == 0, "MOE_INTERMEDIATE_SIZE must be divisible by MOE_THREADS");
static_assert(MOE_THREADS == 512, "Kernel is specialized for 512 threads");
static_assert((MOE_INTERMEDIATE_SIZE / MOE_THREADS) == 4, "Expected 4 activation tiles for 2048/512");

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

__device__ __forceinline__ float wave_broadcast0(float v) {
    return __shfl(v, 0);
}

__device__ __forceinline__ void unpack4_fp8_lut(
    uint32_t p,
    const float* __restrict__ lut,
    float& f0,
    float& f1,
    float& f2,
    float& f3) {
    f0 = lut[(p) & 0xFFu];
    f1 = lut[(p >> 8) & 0xFFu];
    f2 = lut[(p >> 16) & 0xFFu];
    f3 = lut[(p >> 24) & 0xFFu];
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

    constexpr int D_PER_THREAD = MOE_HIDDEN_SIZE / MOE_THREADS; // 14
    constexpr int ACT_TILES = MOE_INTERMEDIATE_SIZE / MOE_THREADS; // 4

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 63;

    __shared__ float x_sh[MOE_HIDDEN_SIZE];
    __shared__ float act_tile_sh[MOE_THREADS];
    __shared__ float in_scale_sh[MOE_HIDDEN_BLOCKS];
    __shared__ int topk_id_sh[MOE_TOPK];
    __shared__ float topk_w_sh[MOE_TOPK];
    __shared__ float fp8_lut_sh[256];

    float acc[D_PER_THREAD];
#pragma unroll
    for (int i = 0; i < D_PER_THREAD; ++i) {
        acc[i] = 0.0f;
    }

    const int64_t input_base = static_cast<int64_t>(token) * MOE_HIDDEN_SIZE;
    const int64_t input_scale_base = static_cast<int64_t>(token) * MOE_HIDDEN_BLOCKS;
    const int64_t tk_base = static_cast<int64_t>(token) * MOE_TOPK;

    for (int i = tid; i < 256; i += blockDim.x) {
        fp8_lut_sh[i] = kFp8E4M3FnuzLut[i];
    }
    for (int i = tid; i < MOE_HIDDEN_BLOCKS; i += blockDim.x) {
        in_scale_sh[i] = input_scale[input_scale_base + i];
    }
    if (tid < MOE_TOPK) {
        topk_id_sh[tid] = topk_ids[tk_base + tid];
        topk_w_sh[tid] = topk_weights[tk_base + tid];
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < D_PER_THREAD; ++i) {
        const int h = tid + i * MOE_THREADS;
        x_sh[h] = fp8_lut_sh[input_q[input_base + h]] * in_scale_sh[h >> 7];
    }
    __syncthreads();

#pragma unroll
    for (int rk = 0; rk < MOE_TOPK; ++rk) {
        const int expert = topk_id_sh[rk];
        const float route_w = topk_w_sh[rk];
        const bool valid_route = (expert >= 0 && expert < MOE_NUM_EXPERTS && route_w != 0.0f);

        if (!valid_route) {
            continue;
        }

        const int64_t w1_e_base = static_cast<int64_t>(expert) * MOE_INTERMEDIATE2_SIZE * MOE_HIDDEN_SIZE;
        const int64_t w2_e_base = static_cast<int64_t>(expert) * MOE_HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE;
        const int64_t fc1_s_base = static_cast<int64_t>(expert) * MOE_FC1_SCALES_PER_EXPERT;
        const int64_t fc2_s_base = static_cast<int64_t>(expert) * MOE_FC2_SCALES_PER_EXPERT;

#pragma unroll
        for (int tile = 0; tile < ACT_TILES; ++tile) {
            const int k = tile * MOE_THREADS + tid; // 0..2047
            const int row_g = k;
            const int row_u = k + MOE_INTERMEDIATE_SIZE;
            const int rb_g = row_g >> 7;
            const int rb_u = row_u >> 7;

            const uint8_t* __restrict__ row_g_ptr =
                w1_q + w1_e_base + static_cast<int64_t>(row_g) * MOE_HIDDEN_SIZE;
            const uint8_t* __restrict__ row_u_ptr =
                w1_q + w1_e_base + static_cast<int64_t>(row_u) * MOE_HIDDEN_SIZE;

            float sum_gate = 0.0f;
            float sum_up = 0.0f;

            for (int cb = 0; cb < MOE_HIDDEN_BLOCKS; ++cb) {
                float s_g = 0.0f;
                float s_u = 0.0f;
                if (lane == 0) {
                    s_g = fc1_scale[fc1_s_base + rb_g * MOE_HIDDEN_BLOCKS + cb];
                    s_u = fc1_scale[fc1_s_base + rb_u * MOE_HIDDEN_BLOCKS + cb];
                }
                s_g = wave_broadcast0(s_g);
                s_u = wave_broadcast0(s_u);

                const int h0 = cb << 7;
                float dot_g = 0.0f;
                float dot_u = 0.0f;

#pragma unroll
                for (int hh = 0; hh < 128; hh += 4) {
                    const int h = h0 + hh;

                    const float x0 = x_sh[h];
                    const float x1 = x_sh[h + 1];
                    const float x2 = x_sh[h + 2];
                    const float x3 = x_sh[h + 3];

                    const uint32_t pg = *reinterpret_cast<const uint32_t*>(row_g_ptr + h);
                    const uint32_t pu = *reinterpret_cast<const uint32_t*>(row_u_ptr + h);

                    float wg0, wg1, wg2, wg3;
                    float wu0, wu1, wu2, wu3;
                    unpack4_fp8_lut(pg, fp8_lut_sh, wg0, wg1, wg2, wg3);
                    unpack4_fp8_lut(pu, fp8_lut_sh, wu0, wu1, wu2, wu3);

                    dot_g = fmaf(x0, wg0, dot_g);
                    dot_g = fmaf(x1, wg1, dot_g);
                    dot_g = fmaf(x2, wg2, dot_g);
                    dot_g = fmaf(x3, wg3, dot_g);

                    dot_u = fmaf(x0, wu0, dot_u);
                    dot_u = fmaf(x1, wu1, dot_u);
                    dot_u = fmaf(x2, wu2, dot_u);
                    dot_u = fmaf(x3, wu3, dot_u);
                }

                sum_gate = fmaf(s_g, dot_g, sum_gate);
                sum_up = fmaf(s_u, dot_u, sum_up);
            }

            act_tile_sh[tid] = silu_f32(sum_gate) * sum_up;
            __syncthreads();

#pragma unroll
            for (int i = 0; i < D_PER_THREAD; ++i) {
                const int d = tid + i * MOE_THREADS;
                const int rb = d >> 7;
                const uint8_t* __restrict__ row_ptr =
                    w2_q + w2_e_base + static_cast<int64_t>(d) * MOE_INTERMEDIATE_SIZE;

                float sum_tile = 0.0f;

#pragma unroll
                for (int cb4 = 0; cb4 < 4; ++cb4) {
                    const int cb_global = (tile << 2) + cb4;

                    float s2 = 0.0f;
                    if (lane == 0) {
                        s2 = fc2_scale[fc2_s_base + rb * MOE_INTER_BLOCKS + cb_global];
                    }
                    s2 = wave_broadcast0(s2);

                    const int k0_local = cb4 << 7;
                    const int k0_global = (tile << 9) + k0_local;

                    float dot2 = 0.0f;
#pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        const int kl = k0_local + kk;
                        const int kg = k0_global + kk;

                        const uint32_t pw = *reinterpret_cast<const uint32_t*>(row_ptr + kg);

                        float w0, w1, w2v, w3;
                        unpack4_fp8_lut(pw, fp8_lut_sh, w0, w1, w2v, w3);

                        dot2 = fmaf(act_tile_sh[kl], w0, dot2);
                        dot2 = fmaf(act_tile_sh[kl + 1], w1, dot2);
                        dot2 = fmaf(act_tile_sh[kl + 2], w2v, dot2);
                        dot2 = fmaf(act_tile_sh[kl + 3], w3, dot2);
                    }

                    sum_tile = fmaf(s2, dot2, sum_tile);
                }

                acc[i] = fmaf(route_w, sum_tile, acc[i]);
            }

            __syncthreads();
        }
    }

    if (tid == 0) {
        acc[0] = mfma_probe(acc[0]);
    }

#pragma unroll
    for (int i = 0; i < D_PER_THREAD; ++i) {
        const int d = tid + i * MOE_THREADS;
        output[input_base + d] = hip_bfloat16(acc[i]);
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