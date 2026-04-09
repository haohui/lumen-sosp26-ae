import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <limits>

__device__ __forceinline__ float wave_reduce_sum_f32(float v) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset);
    }
    return v;
}

template<bool TWO_DIMS>
__global__ __launch_bounds__(64, 8) void sdpa_causal_bf16_wave64_kernel_opt(
    const at::BFloat16* __restrict__ Q,
    const at::BFloat16* __restrict__ K,
    const at::BFloat16* __restrict__ V,
    at::BFloat16* __restrict__ O,
    int B, int Hq, int Hk, int S, int D,
    float scale
) {
    const int tid = threadIdx.x;
    const int i = (int)blockIdx.x;   // sequence position
    const int h = (int)blockIdx.y;   // query head
    const int b = (int)blockIdx.z;   // batch

    if (b >= B || h >= Hq || i >= S) return;

    const int kvh = (Hk == 1) ? 0 : (h % Hk);

    const long long q_base = (((long long)b * Hq + h) * S + i) * D;
    const long long kv_head_base = ((long long)b * Hk + kvh) * (long long)S * D;

    const int d0 = tid;
    const bool a0 = (d0 < D);

    float q0 = a0 ? (float)Q[q_base + d0] * scale : 0.0f;

    int d1 = 0;
    bool a1 = false;
    float q1 = 0.0f;
    if constexpr (TWO_DIMS) {
        d1 = tid + 64;
        a1 = (d1 < D);
        q1 = a1 ? (float)Q[q_base + d1] * scale : 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const at::BFloat16* k_ptr = K + kv_head_base;
    const at::BFloat16* v_ptr = V + kv_head_base;

    for (int j = 0; j <= i; ++j) {
        float part = 0.0f;
        if (a0) part += q0 * (float)k_ptr[d0];
        if constexpr (TWO_DIMS) {
            if (a1) part += q1 * (float)k_ptr[d1];
        }

        float dot = wave_reduce_sum_f32(part);
        float score = __shfl(dot, 0);

        float m_new = fmaxf(m, score);
        float alpha = __expf(m - m_new);
        float beta  = __expf(score - m_new);
        l = l * alpha + beta;
        m = m_new;

        if (a0) acc0 = acc0 * alpha + beta * (float)v_ptr[d0];
        if constexpr (TWO_DIMS) {
            if (a1) acc1 = acc1 * alpha + beta * (float)v_ptr[d1];
        }

        k_ptr += D;
        v_ptr += D;
    }

    float inv_l = 1.0f / l;
    if (a0) O[q_base + d0] = (at::BFloat16)(acc0 * inv_l);
    if constexpr (TWO_DIMS) {
        if (a1) O[q_base + d1] = (at::BFloat16)(acc1 * inv_l);
    }
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
    TORCH_CHECK(S >= 1, "S must be >= 1");
    TORCH_CHECK(Hq >= 1 && B >= 1, "B/Hq must be >= 1");

    auto Qc = Q.contiguous();
    auto Kc = K.contiguous();
    auto Vc = V.contiguous();
    auto O  = torch::zeros_like(Qc);

    constexpr int BLOCK = 64;
    dim3 block(BLOCK);
    dim3 grid((unsigned int)S, (unsigned int)Hq, (unsigned int)B);

    const float scale = 1.0f / std::sqrt((float)D);

    if (D <= 64) {
        sdpa_causal_bf16_wave64_kernel_opt<false><<<grid, block>>>(
            reinterpret_cast<const at::BFloat16*>(Qc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const at::BFloat16*>(Kc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const at::BFloat16*>(Vc.data_ptr<at::BFloat16>()),
            reinterpret_cast<at::BFloat16*>(O.data_ptr<at::BFloat16>()),
            (int)B, (int)Hq, (int)Hk, (int)S, (int)D, scale
        );
    } else {
        sdpa_causal_bf16_wave64_kernel_opt<true><<<grid, block>>>(
            reinterpret_cast<const at::BFloat16*>(Qc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const at::BFloat16*>(Kc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const at::BFloat16*>(Vc.data_ptr<at::BFloat16>()),
            reinterpret_cast<at::BFloat16*>(O.data_ptr<at::BFloat16>()),
            (int)B, (int)Hq, (int)Hk, (int)S, (int)D, scale
        );
    }

    auto err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "sdpa_causal_bf16 kernel launch failed: ", hipGetErrorString(err));

    return O;
}
"""

cpp_src = r"""
torch::Tensor sdpa_causal_bf16(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

_sdpa_ext = load_inline(
    name="sdpa_causal_bf16_wave64_opt_ext",
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
