import os
import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

causal_attn_cpp_source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

__device__ __forceinline__ float bf16_to_float_u16(const uint16_t x) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = static_cast<uint32_t>(x) << 16;
    return v.f;
}

__device__ __forceinline__ uint16_t float_to_bf16_u16_rn(const float x) {
    union {
        uint32_t u;
        float f;
    } v;
    v.f = x;
    uint32_t lsb = (v.u >> 16) & 1u;
    uint32_t rounding_bias = 0x7FFFu + lsb;
    return static_cast<uint16_t>((v.u + rounding_bias) >> 16);
}

__device__ __forceinline__ float wave_reduce_sum(float val) {
    // AMD wavefront: 64 lanes
    val += __shfl_xor(val, 32);
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
    return val;
}

__global__ void causal_attn_bf16_kernel(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    uint16_t* __restrict__ O,
    int B,
    int S,
    int Hq,
    int Hkv,
    int D,
    float scale,
    int groups
) {
    constexpr int WAVE = 64;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (WAVE - 1);
    const int warp_id = tid / WAVE;
    const int num_warps = static_cast<int>(blockDim.x) / WAVE;

    const int bh = static_cast<int>(blockIdx.x);
    const int q_idx = static_cast<int>(blockIdx.y);

    if (q_idx >= S) return;
    const int b = bh / Hq;
    const int h = bh % Hq;
    if (b >= B) return;

    const int kv_h = h / groups;

    const int64_t q_base = (((int64_t)b * S + q_idx) * Hq + h) * D;

    const int d0 = lane;
    const int d1 = lane + WAVE;
    const int d2 = lane + 2 * WAVE;
    const int d3 = lane + 3 * WAVE;

    const float q0 = (d0 < D) ? bf16_to_float_u16(Q[q_base + d0]) : 0.0f;
    const float q1 = (d1 < D) ? bf16_to_float_u16(Q[q_base + d1]) : 0.0f;
    const float q2 = (d2 < D) ? bf16_to_float_u16(Q[q_base + d2]) : 0.0f;
    const float q3 = (d3 < D) ? bf16_to_float_u16(Q[q_base + d3]) : 0.0f;

    float m = -1.0e30f;
    float l = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int k_idx = warp_id; k_idx <= q_idx; k_idx += num_warps) {
        const int64_t k_base = (((int64_t)b * S + k_idx) * Hkv + kv_h) * D;

        float dot = 0.0f;
        if (d0 < D) dot += q0 * bf16_to_float_u16(K[k_base + d0]);
        if (d1 < D) dot += q1 * bf16_to_float_u16(K[k_base + d1]);
        if (d2 < D) dot += q2 * bf16_to_float_u16(K[k_base + d2]);
        if (d3 < D) dot += q3 * bf16_to_float_u16(K[k_base + d3]);

        dot = wave_reduce_sum(dot);
        const float score = __shfl(dot, 0) * scale;

        float vv0 = 0.0f, vv1 = 0.0f, vv2 = 0.0f, vv3 = 0.0f;
        if (d0 < D) vv0 = bf16_to_float_u16(V[k_base + d0]);
        if (d1 < D) vv1 = bf16_to_float_u16(V[k_base + d1]);
        if (d2 < D) vv2 = bf16_to_float_u16(V[k_base + d2]);
        if (d3 < D) vv3 = bf16_to_float_u16(V[k_base + d3]);

        if (score > m) {
            const float alpha = expf(m - score);
            l = l * alpha + 1.0f;
            acc0 = acc0 * alpha + vv0;
            acc1 = acc1 * alpha + vv1;
            acc2 = acc2 * alpha + vv2;
            acc3 = acc3 * alpha + vv3;
            m = score;
        } else {
            const float w = expf(score - m);
            l += w;
            acc0 += w * vv0;
            acc1 += w * vv1;
            acc2 += w * vv2;
            acc3 += w * vv3;
        }
    }

    extern __shared__ float smem[];
    float* s_acc = smem;                               // num_warps * D
    float* s_m = s_acc + num_warps * D;               // num_warps
    float* s_l = s_m + num_warps;                     // num_warps
    float* s_global = s_l + num_warps;                // 2 floats: global_m, global_l

    if (d0 < D) s_acc[warp_id * D + d0] = acc0;
    if (d1 < D) s_acc[warp_id * D + d1] = acc1;
    if (d2 < D) s_acc[warp_id * D + d2] = acc2;
    if (d3 < D) s_acc[warp_id * D + d3] = acc3;

    if (lane == 0) {
        s_m[warp_id] = m;
        s_l[warp_id] = l;
    }

    __syncthreads();

    if (tid == 0) {
        float global_m = -1.0e30f;
        for (int w = 0; w < num_warps; ++w) {
            if (s_m[w] > global_m) global_m = s_m[w];
        }

        float global_l = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            global_l += s_l[w] * expf(s_m[w] - global_m);
        }

        s_global[0] = global_m;
        s_global[1] = global_l;
    }

    __syncthreads();

    if (warp_id == 0) {
        const float global_m = s_global[0];
        const float global_l = s_global[1];

        if (global_l > 0.0f) {
            for (int d = lane; d < D; d += WAVE) {
                float acc = 0.0f;
                for (int w = 0; w < num_warps; ++w) {
                    acc += s_acc[w * D + d] * expf(s_m[w] - global_m);
                }
                const float out_f = acc / global_l;
                O[q_base + d] = float_to_bf16_u16_rn(out_f);
            }
        } else {
            for (int d = lane; d < D; d += WAVE) {
                O[q_base + d] = float_to_bf16_u16_rn(0.0f);
            }
        }
    }
}

torch::Tensor causal_attention_bf16_hip(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q/K/V must be HIP tensors");
    TORCH_CHECK(Q.scalar_type() == at::kBFloat16, "Q must be bfloat16");
    TORCH_CHECK(K.scalar_type() == at::kBFloat16, "K must be bfloat16");
    TORCH_CHECK(V.scalar_type() == at::kBFloat16, "V must be bfloat16");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "Q/K/V must be [B, S, H, D]");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "Q/K/V must be contiguous");

    const int64_t B64 = Q.size(0);
    const int64_t S64 = Q.size(1);
    const int64_t Hq64 = Q.size(2);
    const int64_t D64 = Q.size(3);

    TORCH_CHECK(K.size(0) == B64 && V.size(0) == B64, "B mismatch");
    TORCH_CHECK(K.size(1) == S64 && V.size(1) == S64, "S mismatch");
    TORCH_CHECK(K.size(3) == D64 && V.size(3) == D64, "D mismatch");
    TORCH_CHECK(K.size(2) == V.size(2), "K/V head mismatch");

    const int64_t Hkv64 = K.size(2);
    TORCH_CHECK(Hq64 % Hkv64 == 0, "Hq must be divisible by Hkv");
    TORCH_CHECK(D64 <= 256, "Head dim D > 256 is not supported by this kernel");

    const int B = static_cast<int>(B64);
    const int S = static_cast<int>(S64);
    const int Hq = static_cast<int>(Hq64);
    const int Hkv = static_cast<int>(Hkv64);
    const int D = static_cast<int>(D64);
    const int groups = Hq / Hkv;

    auto O = torch::empty_like(Q);

    const dim3 block(256, 1, 1);           // 4 wavefronts
    const dim3 grid(static_cast<unsigned int>(B * Hq),
                    static_cast<unsigned int>(S),
                    1u);
    const int num_warps = 4;
    const size_t shared_bytes = static_cast<size_t>(
        (num_warps * D + 2 * num_warps + 2) * sizeof(float)
    );

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    hipLaunchKernelGGL(
        causal_attn_bf16_kernel,
        grid,
        block,
        shared_bytes,
        stream,
        reinterpret_cast<const uint16_t*>(Q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(K.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(V.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t*>(O.data_ptr<at::BFloat16>()),
        B, S, Hq, Hkv, D, scale, groups
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "causal_attn_bf16_kernel launch failed: ", hipGetErrorString(err));

    return O;
}
"""

causal_attn_ext = load_inline(
    name="causal_attn_bf16_hip_ext_v1",
    cpp_sources=causal_attn_cpp_source,
    functions=["causal_attention_bf16_hip"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.attn = causal_attn_ext

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # External and internal layout kept as [B, S, H, D].
        return self.attn.causal_attention_bf16_hip(
            Q.contiguous(), K.contiguous(), V.contiguous()
        )


batch_size = 16
num_q_heads = 8
num_kv_heads = 1
sequence_length = int(os.getenv("ATTN_SEQ_LEN", "1024"))
head_dim = 128
supported_sequence_lengths = (1024, 2048, 4096, 8192, 16384)


def get_inputs():
    Q = torch.randn(batch_size, sequence_length, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(batch_size, sequence_length, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(batch_size, sequence_length, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    return [Q, K, V]


def get_init_inputs():
    return []


ANTI_HACK_MANIFEST = {
    "forbidden_api_used": [],
    "main_compute_kernels": ["causal_attn_bf16_kernel"],
    "fallback_path": False,
}

PERF_MANIFEST = {
    "launch_config": {
        "grid": "(B*Hq, S, 1)",
        "block": "(256, 1, 1)",
        "num_warps": 4,
    },
    "tile_sizes": {
        "BLOCK_M": 1,                 # one query position per block
        "BLOCK_N": "warp-strided keys across 4 wavefronts",
        "BLOCK_K": 128,               # typical head_dim for this model
    },
    "expected_parallelism": "Each [B,H,query] row is computed cooperatively by 4 wavefronts (256 threads), with keys partitioned across wavefronts and reduced in shared memory.",
}
