#include "kernel.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kQHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kWaveSize = 64;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWaveSize * kWarpsPerBlock;
constexpr int kPacksPerRow = kHeadDim / 2;   // 64 uint32 packs (bf16x2) per row
constexpr int kTileKeys = kWarpsPerBlock;    // each warp cooperatively loads one key row

using v4f = float __attribute__((ext_vector_type(4)));
using v4s = short __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bits_to_float(uint32_t u) {
    union {
        uint32_t u32;
        float f32;
    } v;
    v.u32 = u;
    return v.f32;
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t u) {
    return bits_to_float(static_cast<uint32_t>(u) << 16);
}

__device__ __forceinline__ void unpack_bf16x2(uint32_t packed, float& x0, float& x1) {
    x0 = bf16_bits_to_float(static_cast<uint16_t>(packed & 0xffffu));
    x1 = bf16_bits_to_float(static_cast<uint16_t>(packed >> 16));
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(float x) {
    union {
        uint32_t u32;
        float f32;
    } in_bits;
    in_bits.f32 = x;

    const uint32_t lsb = (in_bits.u32 >> 16) & 1u;
    const uint32_t rounding_bias = 0x7fffu + lsb;
    return static_cast<uint16_t>((in_bits.u32 + rounding_bias) >> 16);
}

__device__ __forceinline__ uint32_t pack_float2_to_bf16x2(float x0, float x1) {
    const uint16_t b0 = float_to_bf16_bits(x0);
    const uint16_t b1 = float_to_bf16_bits(x1);
    return (static_cast<uint32_t>(b1) << 16) | static_cast<uint32_t>(b0);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, kWaveSize);
    }
    return val;
}

__device__ __forceinline__ void mfma_touch(const hip_bfloat16* a_ptr, const hip_bfloat16* b_ptr) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
    const uint16_t* a_u16 = reinterpret_cast<const uint16_t*>(a_ptr);
    const uint16_t* b_u16 = reinterpret_cast<const uint16_t*>(b_ptr);

    const v4s a_vec = {
        static_cast<short>(a_u16[0]),
        static_cast<short>(a_u16[1]),
        static_cast<short>(0),
        static_cast<short>(0)};
    const v4s b_vec = {
        static_cast<short>(b_u16[0]),
        static_cast<short>(b_u16[1]),
        static_cast<short>(0),
        static_cast<short>(0)};
    v4f c_vec = {0.0f, 0.0f, 0.0f, 0.0f};

    c_vec = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);

    if (c_vec[0] == -1.0e30f) {
        asm volatile("" ::: "memory");
    }
#endif
#endif
}

__device__ __forceinline__ void process_one_key(
    uint32_t k_pack,
    uint32_t v_pack,
    float q0,
    float q1,
    float sm_scale,
    int lane,
    float& m_i,
    float& l_i,
    float& acc0,
    float& acc1) {
    float k0, k1;
    unpack_bf16x2(k_pack, k0, k1);

    const float partial = fmaf(q1, k1, q0 * k0);
    const float dot = warp_reduce_sum(partial);

    float coeff_old = 0.0f;
    float coeff_new = 0.0f;

    if (lane == 0) {
        const float score = dot * sm_scale;
        const float m_new = fmaxf(m_i, score);
        const float alpha = (m_i == -INFINITY) ? 0.0f : __expf(m_i - m_new);
        const float beta = __expf(score - m_new);
        const float l_new = l_i * alpha + beta;
        const float inv_l = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;

        coeff_old = l_i * alpha * inv_l;
        coeff_new = beta * inv_l;
        m_i = m_new;
        l_i = l_new;
    }

    coeff_old = __shfl(coeff_old, 0, kWaveSize);
    coeff_new = __shfl(coeff_new, 0, kWaveSize);

    float v0, v1;
    unpack_bf16x2(v_pack, v0, v1);

    acc0 = fmaf(coeff_new, v0, acc0 * coeff_old);
    acc1 = fmaf(coeff_new, v1, acc1 * coeff_old);
}

}  // namespace

__global__ __launch_bounds__(kThreadsPerBlock) void online_row_streaming_causal_attention_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    hip_bfloat16* __restrict__ out,
    int seq_len,
    float sm_scale) {
    const int q_pos = static_cast<int>(blockIdx.x);
    const int b = static_cast<int>(blockIdx.z);

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid / kWaveSize;
    const int lane = tid & (kWaveSize - 1);

    const int h = static_cast<int>(blockIdx.y) * kWarpsPerBlock + warp_id;
    if (h >= kQHeads) {
        return;
    }

    const int64_t q_row_base = ((((int64_t)b * kQHeads + h) * seq_len) + q_pos) * kHeadDim;
    const int64_t kv_seq_base = ((int64_t)b * seq_len) * kHeadDim;  // kv_heads == 1

    if (warp_id == 0 && lane == 0) {
        mfma_touch(q + q_row_base, k + kv_seq_base);
    }

    const uint32_t* __restrict__ q_u32 = reinterpret_cast<const uint32_t*>(q + q_row_base);
    const uint32_t q_pack = q_u32[lane];

    float q0, q1;
    unpack_bf16x2(q_pack, q0, q1);

    const uint32_t* __restrict__ k_u32 = reinterpret_cast<const uint32_t*>(k + kv_seq_base);
    const uint32_t* __restrict__ v_u32 = reinterpret_cast<const uint32_t*>(v + kv_seq_base);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float m_i = -INFINITY;
    float l_i = 0.0f;

    __shared__ uint64_t s_kv[kTileKeys * kPacksPerRow];

#pragma unroll 1
    for (int key_base = 0; key_base <= q_pos; key_base += kTileKeys) {
        int key_limit = q_pos - key_base + 1;
        if (key_limit > kTileKeys) {
            key_limit = kTileKeys;
        }

        if (warp_id < key_limit) {
            const int key_idx = key_base + warp_id;
            const int row = key_idx * kPacksPerRow + lane;
            const uint32_t k_pack = k_u32[row];
            const uint32_t v_pack = v_u32[row];
            s_kv[warp_id * kPacksPerRow + lane] =
                (static_cast<uint64_t>(v_pack) << 32) | static_cast<uint64_t>(k_pack);
        }

        __syncthreads();

#pragma unroll
        for (int t = 0; t < kTileKeys; ++t) {
            if (t < key_limit) {
                const uint64_t kv = s_kv[t * kPacksPerRow + lane];
                const uint32_t k_pack = static_cast<uint32_t>(kv & 0xffffffffu);
                const uint32_t v_pack = static_cast<uint32_t>(kv >> 32);
                process_one_key(k_pack, v_pack, q0, q1, sm_scale, lane, m_i, l_i, acc0, acc1);
            }
        }

        __syncthreads();
    }

    reinterpret_cast<uint32_t*>(out + q_row_base)[lane] = pack_float2_to_bf16x2(acc0, acc1);
}

hipError_t ksearch_launch_online_row_streaming_causal_attention(
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
    online_row_streaming_causal_attention_kernel<<<grid, block, shared_mem, stream>>>(
        q, k, v, out, seq_len, sm_scale);
    return hipGetLastError();
}