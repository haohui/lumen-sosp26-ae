import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <limits>

template <typename T>
__device__ __forceinline__ T wave_reduce_sum(T v) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset);
    }
    return v;
}

// One-wave kernel (64 threads): each lane handles up to 2 dims (d, d+64), D <= 128.
template<int BLOCK_THREADS>
__global__ void sdpa_causal_bf16_wave64_kernel(
    const at::BFloat16* __restrict__ Q,
    const at::BFloat16* __restrict__ K,
    const at::BFloat16* __restrict__ V,
    at::BFloat16* __restrict__ O,
    int B, int Hq, int Hk, int S, int D,
    float scale
) {
    static_assert(BLOCK_THREADS == 64, "This kernel is specialized for one wave (64 threads).");
    const int tid = threadIdx.x;

    const long long block_id = (long long)blockIdx.x;
    const int i = (int)(block_id % S);
    const long long t1 = block_id / S;
    const int h = (int)(t1 % Hq);
    const int b = (int)(t1 / Hq);
    if (b >= B) return;

    const int kvh = (Hk == 1) ? 0 : (h % Hk);

    const long long q_base = (((long long)b * Hq + h) * S + i) * D;
    const long long kv_head_base = ((long long)b * Hk + kvh) * S * D;

    const int d0 = tid;
    const int d1 = tid + 64;

    const bool a0 = (d0 < D);
    const bool a1 = (d1 < D);

    // Cache Q in registers once (reused across all j for fixed i).
    float q0 = a0 ? (float)Q[q_base + d0] : 0.0f;
    float q1 = a1 ? (float)Q[q_base + d1] : 0.0f;

    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int j = 0; j <= i; ++j) {
        const long long kv_row = kv_head_base + (long long)j * D;

        float part = 0.0f;
        if (a0) part += q0 * (float)K[kv_row + d0];
        if (a1) part += q1 * (float)K[kv_row + d1];

        float dot = wave_reduce_sum(part);
        float score = __shfl(dot, 0) * scale;  // broadcast lane-0 sum to full wave

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float beta  = expf(score - m_new);
        l = l * alpha + beta;
        m = m_new;

        if (a0) {
            float v0 = (float)V[kv_row + d0];
            acc0 = acc0 * alpha + beta * v0;
        }
        if (a1) {
            float v1 = (float)V[kv_row + d1];
            acc1 = acc1 * alpha + beta * v1;
        }
    }

    float inv_l = 1.0f / l;
    if (a0) O[q_base + d0] = (at::BFloat16)(acc0 * inv_l);
    if (a1) O[q_base + d1] = (at::BFloat16)(acc1 * inv_l);
}

torch::Tensor sdpa_causal_bf16(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a HIP tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a HIP tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a HIP tensor");

    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bfloat16");
    TORCH_CHECK(K.dtype() == torch::kBFloat16, "K must be bfloat16");
    TORCH_CHECK(V.dtype() == torch::kBFloat16, "V must be bfloat16");

    TORCH_CHECK(Q.dim() == 4, "Q must be [B, Hq, S, D]");
    TORCH_CHECK(K.dim() == 4, "K must be [B, Hk, S, D]");
    TORCH_CHECK(V.dim() == 4, "V must be [B, Hk, S, D]");

    const int64_t B  = Q.size(0);
    const int64_t Hq = Q.size(1);
    const int64_t S  = Q.size(2);
    const int64_t D  = Q.size(3);

    TORCH_CHECK(K.size(0) == B && V.size(0) == B, "Batch mismatch");
    TORCH_CHECK(K.size(2) == S && V.size(2) == S, "Sequence mismatch");
    TORCH_CHECK(K.size(3) == D && V.size(3) == D, "Head-dim mismatch");

    const int64_t Hk = K.size(1);
    TORCH_CHECK(V.size(1) == Hk, "K/V head mismatch");
    TORCH_CHECK(Hk >= 1, "Hk must be >= 1");
    TORCH_CHECK(D > 0 && D <= 128, "This kernel supports 1 <= D <= 128");

    auto Qc = Q.contiguous();
    auto Kc = K.contiguous();
    auto Vc = V.contiguous();
    auto O  = torch::zeros_like(Qc);

    constexpr int BLOCK = 64;
    const long long total_blocks = B * Hq * S;
    dim3 grid((unsigned int)total_blocks);
    dim3 block(BLOCK);

    const float scale = 1.0f / std::sqrt((float)D);

    sdpa_causal_bf16_wave64_kernel<BLOCK><<<grid, block>>>(
        reinterpret_cast<const at::BFloat16*>(Qc.data_ptr<at::BFloat16>()),
        reinterpret_cast<const at::BFloat16*>(Kc.data_ptr<at::BFloat16>()),
        reinterpret_cast<const at::BFloat16*>(Vc.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(O.data_ptr<at::BFloat16>()),
        (int)B, (int)Hq, (int)Hk, (int)S, (int)D, scale
    );

    auto err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "sdpa_causal_bf16 kernel launch failed: ", hipGetErrorString(err));

    return O;
}
"""

cpp_src = r"""
torch::Tensor sdpa_causal_bf16(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

_sdpa_ext = load_inline(
    name="sdpa_causal_bf16_wave64_ext",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["sdpa_causal_bf16"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sdpa = _sdpa_ext

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.sdpa.sdpa_causal_bf16(Q, K, V)
