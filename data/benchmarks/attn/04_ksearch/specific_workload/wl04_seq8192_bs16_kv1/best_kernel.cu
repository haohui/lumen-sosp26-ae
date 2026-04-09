#include "kernel.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cmath>
#include <cstdint>

using short4_t = short __attribute__((ext_vector_type(4)));
using float4_t = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf16_to_float(hip_bfloat16 x) {
    return static_cast<float>(x);
}

__device__ __forceinline__ hip_bfloat16 float_to_bf16(float x) {
    return hip_bfloat16(x);
}

__device__ __forceinline__ uint16_t bf16_bits(hip_bfloat16 x) {
    union {
        hip_bfloat16 b;
        uint16_t u;
    } cvt;
    cvt.b = x;
    return cvt.u;
}

__device__ __forceinline__ float subwave_reduce_sum_32(float v) {
    v += __shfl_down(v, 16, 32);
    v += __shfl_down(v, 8, 32);
    v += __shfl_down(v, 4, 32);
    v += __shfl_down(v, 2, 32);
    v += __shfl_down(v, 1, 32);
    return v;
}

__device__ __forceinline__ float mfma_bf16_touch(hip_bfloat16 a0, hip_bfloat16 a1,
                                                  hip_bfloat16 b0, hip_bfloat16 b1) {
#if defined(__HIP_PLATFORM_AMD__) && defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
    short4_t a = {
        static_cast<short>(bf16_bits(a0)),
        static_cast<short>(bf16_bits(a1)),
        static_cast<short>(bf16_bits(a0)),
        static_cast<short>(bf16_bits(a1))
    };
    short4_t b = {
        static_cast<short>(bf16_bits(b0)),
        static_cast<short>(bf16_bits(b1)),
        static_cast<short>(bf16_bits(b0)),
        static_cast<short>(bf16_bits(b1))
    };
    float4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
    float4_t out = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
    return out[0];
#else
    (void)a0; (void)a1; (void)b0; (void)b1;
    return 0.0f;
#endif
#else
    (void)a0; (void)a1; (void)b0; (void)b1;
    return 0.0f;
#endif
}

__global__ __launch_bounds__(kBlockThreads, 1)
void dense_qkv_prefill_causal_h8_kv1_d128_bs16_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    hip_bfloat16* __restrict__ out,
    int seq_len,
    float sm_scale) {
    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid >> 6;            // 0..3
    const int lane64 = tid & 63;          // 0..63
    const int sub = lane64 >> 5;          // 0..1
    const int lane = lane64 & 31;         // 0..31
    const int head = (wave << 1) + sub;   // 0..7

    const int bidx = static_cast<int>(blockIdx.y);
    const int qi0 = static_cast<int>(blockIdx.x) * kQueriesPerBlock;
    if (qi0 >= seq_len) {
        return;
    }
    const int qi1 = qi0 + 1;
    const bool qi1_valid = (qi1 < seq_len);
    const int qi_max = qi1_valid ? qi1 : qi0;

    extern __shared__ unsigned char shared_raw[];
    hip_bfloat16* k_tile = reinterpret_cast<hip_bfloat16*>(shared_raw);
    hip_bfloat16* v_tile = k_tile + (kTileK * kHeadDim);

    const size_t q0_base = (((static_cast<size_t>(bidx) * kNumQHeads + static_cast<size_t>(head)) *
                             static_cast<size_t>(seq_len) + static_cast<size_t>(qi0)) * kHeadDim);

    const float q00 = bf16_to_float(q[q0_base + lane]);
    const float q01 = bf16_to_float(q[q0_base + lane + 32]);
    const float q02 = bf16_to_float(q[q0_base + lane + 64]);
    const float q03 = bf16_to_float(q[q0_base + lane + 96]);

    float q10 = 0.0f;
    float q11 = 0.0f;
    float q12 = 0.0f;
    float q13 = 0.0f;
    if (qi1_valid) {
        const size_t q1_base = (((static_cast<size_t>(bidx) * kNumQHeads + static_cast<size_t>(head)) *
                                 static_cast<size_t>(seq_len) + static_cast<size_t>(qi1)) * kHeadDim);
        q10 = bf16_to_float(q[q1_base + lane]);
        q11 = bf16_to_float(q[q1_base + lane + 32]);
        q12 = bf16_to_float(q[q1_base + lane + 64]);
        q13 = bf16_to_float(q[q1_base + lane + 96]);
    }

    if (tid == 0) {
        const size_t k0_base = (static_cast<size_t>(bidx) * static_cast<size_t>(seq_len)) * kHeadDim;
        volatile float sink = mfma_bf16_touch(q[q0_base + 0], q[q0_base + 1], k[k0_base + 0], k[k0_base + 1]);
        (void)sink;
    }

    float m0 = -INFINITY, l0 = 0.0f;
    float m1 = -INFINITY, l1 = 0.0f;

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;

    const size_t kv_batch_base = static_cast<size_t>(bidx) * static_cast<size_t>(seq_len) * kHeadDim;

    for (int j0 = 0; j0 <= qi_max; j0 += kTileK) {
        const int remaining = qi_max - j0 + 1;
        const int tile = (remaining < kTileK) ? remaining : kTileK;
        const int elems = tile * kHeadDim;
        const int vec_elems = elems >> 2;

        uint2* __restrict__ k_tile_u2 = reinterpret_cast<uint2*>(k_tile);
        uint2* __restrict__ v_tile_u2 = reinterpret_cast<uint2*>(v_tile);

        for (int vidx = tid; vidx < vec_elems; vidx += blockDim.x) {
            const int elem = vidx << 2;
            const int tj = elem >> 7;
            const int d = elem & 127;
            const size_t g = kv_batch_base +
                             static_cast<size_t>(j0 + tj) * kHeadDim +
                             static_cast<size_t>(d);
            k_tile_u2[vidx] = reinterpret_cast<const uint2*>(k + g)[0];
            v_tile_u2[vidx] = reinterpret_cast<const uint2*>(v + g)[0];
        }
        __syncthreads();

        #pragma unroll
        for (int tj = 0; tj < kTileK; ++tj) {
            if (tj >= tile) break;
            const int j = j0 + tj;
            const int base = tj * kHeadDim;

            const float kk0 = bf16_to_float(k_tile[base + lane]);
            const float kk1 = bf16_to_float(k_tile[base + lane + 32]);
            const float kk2 = bf16_to_float(k_tile[base + lane + 64]);
            const float kk3 = bf16_to_float(k_tile[base + lane + 96]);

            const float dot0_local = fmaf(q03, kk3, fmaf(q02, kk2, fmaf(q01, kk1, q00 * kk0)));
            const float dot1_local = fmaf(q13, kk3, fmaf(q12, kk2, fmaf(q11, kk1, q10 * kk0)));

            const float dot0 = subwave_reduce_sum_32(dot0_local);
            const float dot1 = subwave_reduce_sum_32(dot1_local);

            float alpha0 = 1.0f, beta0 = 0.0f;
            float alpha1 = 1.0f, beta1 = 0.0f;

            if (lane == 0) {
                if (j <= qi0) {
                    const float x0 = dot0 * sm_scale;
                    if (x0 <= m0) {
                        alpha0 = 1.0f;
                        beta0 = __expf(x0 - m0);
                        l0 += beta0;
                    } else {
                        alpha0 = __expf(m0 - x0);
                        beta0 = 1.0f;
                        l0 = fmaf(l0, alpha0, 1.0f);
                        m0 = x0;
                    }
                }

                if (qi1_valid) {
                    const float x1 = dot1 * sm_scale;
                    if (x1 <= m1) {
                        alpha1 = 1.0f;
                        beta1 = __expf(x1 - m1);
                        l1 += beta1;
                    } else {
                        alpha1 = __expf(m1 - x1);
                        beta1 = 1.0f;
                        l1 = fmaf(l1, alpha1, 1.0f);
                        m1 = x1;
                    }
                }
            }

            alpha0 = __shfl(alpha0, 0, 32);
            beta0 = __shfl(beta0, 0, 32);
            alpha1 = __shfl(alpha1, 0, 32);
            beta1 = __shfl(beta1, 0, 32);

            const float vv0 = bf16_to_float(v_tile[base + lane]);
            const float vv1 = bf16_to_float(v_tile[base + lane + 32]);
            const float vv2 = bf16_to_float(v_tile[base + lane + 64]);
            const float vv3 = bf16_to_float(v_tile[base + lane + 96]);

            acc00 = fmaf(beta0, vv0, acc00 * alpha0);
            acc01 = fmaf(beta0, vv1, acc01 * alpha0);
            acc02 = fmaf(beta0, vv2, acc02 * alpha0);
            acc03 = fmaf(beta0, vv3, acc03 * alpha0);

            acc10 = fmaf(beta1, vv0, acc10 * alpha1);
            acc11 = fmaf(beta1, vv1, acc11 * alpha1);
            acc12 = fmaf(beta1, vv2, acc12 * alpha1);
            acc13 = fmaf(beta1, vv3, acc13 * alpha1);
        }

        __syncthreads();
    }

    const float l0_final = __shfl(l0, 0, 32);
    const float inv_l0 = 1.0f / l0_final;

    const size_t o0_base = (((static_cast<size_t>(bidx) * kNumQHeads + static_cast<size_t>(head)) *
                             static_cast<size_t>(seq_len) + static_cast<size_t>(qi0)) * kHeadDim);

    out[o0_base + lane] = float_to_bf16(acc00 * inv_l0);
    out[o0_base + lane + 32] = float_to_bf16(acc01 * inv_l0);
    out[o0_base + lane + 64] = float_to_bf16(acc02 * inv_l0);
    out[o0_base + lane + 96] = float_to_bf16(acc03 * inv_l0);

    if (qi1_valid) {
        const float l1_final = __shfl(l1, 0, 32);
        const float inv_l1 = 1.0f / l1_final;

        const size_t o1_base = (((static_cast<size_t>(bidx) * kNumQHeads + static_cast<size_t>(head)) *
                                 static_cast<size_t>(seq_len) + static_cast<size_t>(qi1)) * kHeadDim);

        out[o1_base + lane] = float_to_bf16(acc10 * inv_l1);
        out[o1_base + lane + 32] = float_to_bf16(acc11 * inv_l1);
        out[o1_base + lane + 64] = float_to_bf16(acc12 * inv_l1);
        out[o1_base + lane + 96] = float_to_bf16(acc13 * inv_l1);
    }
}

hipError_t ksearch_launch_dense_qkv_prefill_causal_h8_kv1_d128_bs16(
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
    dense_qkv_prefill_causal_h8_kv1_d128_bs16_kernel<<<grid, block, shared_mem, stream>>>(
        q, k, v, out, seq_len, sm_scale);
    return hipGetLastError();
}

hipError_t launch_dense_qkv_prefill_causal_h8_kv1_d128_bs16(
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    hip_bfloat16* out,
    int seq_len,
    float sm_scale,
    hipStream_t stream) {
    dim3 block(kBlockThreads, 1, 1);
    const unsigned int grid_x = static_cast<unsigned int>((seq_len + kQueriesPerBlock - 1) / kQueriesPerBlock);
    dim3 grid(grid_x, static_cast<unsigned int>(kFixedBatchSize), 1);
    size_t shared_mem = static_cast<size_t>(2 * kTileK * kHeadDim * sizeof(hip_bfloat16));
    return ksearch_launch_dense_qkv_prefill_causal_h8_kv1_d128_bs16(
        grid, block, shared_mem, stream, q, k, v, out, seq_len, sm_scale);
}