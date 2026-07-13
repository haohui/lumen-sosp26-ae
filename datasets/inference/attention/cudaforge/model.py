import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <limits>

__device__ __forceinline__ float wave_reduce_sum_64(float v) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, 64);
    }
    return __shfl(v, 0, 64);
}

__global__ __launch_bounds__(128, 4) void sdpa_causal_bf16_d64_kernel_w2(
    const at::BFloat16* __restrict__ Q,
    const at::BFloat16* __restrict__ K,
    const at::BFloat16* __restrict__ V,
    at::BFloat16* __restrict__ O,
    int B, int Hq, int Hk, int S,
    float scale
) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6; // 0..1

    const int i = (int)blockIdx.x * 2 + wave;
    const int h = (int)blockIdx.y;
    const int b = (int)blockIdx.z;

    if (b >= B || h >= Hq || i >= S) return;

    const int kvh = (Hk == 1) ? 0 : (h % Hk);

    const long long q_base = (((long long)b * S + i) * Hq + h) * 64LL;
    const long long kv_head_base = (((long long)b * S) * Hk + kvh) * 64LL;

    const float q = (float)Q[q_base + lane] * scale;

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;

    const at::BFloat16* k_ptr = K + kv_head_base;
    const at::BFloat16* v_ptr = V + kv_head_base;

    int j = 0;
    for (; j + 1 <= i; j += 2) {
        float part0 = q * (float)k_ptr[lane];
        float score0 = wave_reduce_sum_64(part0);

        float m_new0 = fmaxf(m, score0);
        float alpha0 = expf(m - m_new0);
        float beta0  = expf(score0 - m_new0);

        l = l * alpha0 + beta0;
        m = m_new0;
        acc = acc * alpha0 + beta0 * (float)v_ptr[lane];

        k_ptr += (long long)Hk * 64LL;
        v_ptr += (long long)Hk * 64LL;

        float part1 = q * (float)k_ptr[lane];
        float score1 = wave_reduce_sum_64(part1);

        float m_new1 = fmaxf(m, score1);
        float alpha1 = expf(m - m_new1);
        float beta1  = expf(score1 - m_new1);

        l = l * alpha1 + beta1;
        m = m_new1;
        acc = acc * alpha1 + beta1 * (float)v_ptr[lane];

        k_ptr += (long long)Hk * 64LL;
        v_ptr += (long long)Hk * 64LL;
    }

    if (j <= i) {
        float part = q * (float)k_ptr[lane];
        float score = wave_reduce_sum_64(part);

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float beta  = expf(score - m_new);

        l = l * alpha + beta;
        acc = acc * alpha + beta * (float)v_ptr[lane];
    }

    O[q_base + lane] = (at::BFloat16)(acc / l);
}

__global__ __launch_bounds__(128, 4) void sdpa_causal_bf16_d128_kernel_w2(
    const at::BFloat16* __restrict__ Q,
    const at::BFloat16* __restrict__ K,
    const at::BFloat16* __restrict__ V,
    at::BFloat16* __restrict__ O,
    int B, int Hq, int Hk, int S,
    float scale
) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6; // 0..1

    const int i = (int)blockIdx.x * 2 + wave;
    const int h = (int)blockIdx.y;
    const int b = (int)blockIdx.z;

    if (b >= B || h >= Hq || i >= S) return;

    const int kvh = (Hk == 1) ? 0 : (h % Hk);

    const long long q_base = (((long long)b * S + i) * Hq + h) * 128LL;
    const long long kv_head_base = (((long long)b * S) * Hk + kvh) * 128LL;

    const int d0 = lane;
    const int d1 = lane + 64;

    const float q0 = (float)Q[q_base + d0] * scale;
    const float q1 = (float)Q[q_base + d1] * scale;

    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const at::BFloat16* k_ptr = K + kv_head_base;
    const at::BFloat16* v_ptr = V + kv_head_base;

    int j = 0;
    for (; j + 1 <= i; j += 2) {
        float part0 = q0 * (float)k_ptr[d0] + q1 * (float)k_ptr[d1];
        float score0 = wave_reduce_sum_64(part0);

        float m_new0 = fmaxf(m, score0);
        float alpha0 = expf(m - m_new0);
        float beta0  = expf(score0 - m_new0);

        l = l * alpha0 + beta0;
        m = m_new0;
        acc0 = acc0 * alpha0 + beta0 * (float)v_ptr[d0];
        acc1 = acc1 * alpha0 + beta0 * (float)v_ptr[d1];

        k_ptr += (long long)Hk * 128LL;
        v_ptr += (long long)Hk * 128LL;

        float part1 = q0 * (float)k_ptr[d0] + q1 * (float)k_ptr[d1];
        float score1 = wave_reduce_sum_64(part1);

        float m_new1 = fmaxf(m, score1);
        float alpha1 = expf(m - m_new1);
        float beta1  = expf(score1 - m_new1);

        l = l * alpha1 + beta1;
        m = m_new1;
        acc0 = acc0 * alpha1 + beta1 * (float)v_ptr[d0];
        acc1 = acc1 * alpha1 + beta1 * (float)v_ptr[d1];

        k_ptr += (long long)Hk * 128LL;
        v_ptr += (long long)Hk * 128LL;
    }

    if (j <= i) {
        float part = q0 * (float)k_ptr[d0] + q1 * (float)k_ptr[d1];
        float score = wave_reduce_sum_64(part);

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float beta  = expf(score - m_new);

        l = l * alpha + beta;
        acc0 = acc0 * alpha + beta * (float)v_ptr[d0];
        acc1 = acc1 * alpha + beta * (float)v_ptr[d1];
    }

    float inv_l = 1.0f / l;
    O[q_base + d0] = (at::BFloat16)(acc0 * inv_l);
    O[q_base + d1] = (at::BFloat16)(acc1 * inv_l);
}

template<bool TWO_DIMS>
__global__ __launch_bounds__(128, 4) void sdpa_causal_bf16_generic_kernel_w2(
    const at::BFloat16* __restrict__ Q,
    const at::BFloat16* __restrict__ K,
    const at::BFloat16* __restrict__ V,
    at::BFloat16* __restrict__ O,
    int B, int Hq, int Hk, int S, int D,
    float scale
) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6; // 0..1

    const int i = (int)blockIdx.x * 2 + wave;
    const int h = (int)blockIdx.y;
    const int b = (int)blockIdx.z;

    if (b >= B || h >= Hq || i >= S) return;

    const int kvh = (Hk == 1) ? 0 : (h % Hk);

    const long long q_base = (((long long)b * S + i) * Hq + h) * (long long)D;
    const long long kv_head_base = (((long long)b * S) * Hk + kvh) * (long long)D;

    const int d0 = lane;
    const bool a0 = (d0 < D);
    const float q0 = a0 ? ((float)Q[q_base + d0] * scale) : 0.0f;

    int d1 = 0;
    bool a1 = false;
    float q1 = 0.0f;
    if constexpr (TWO_DIMS) {
        d1 = lane + 64;
        a1 = (d1 < D);
        q1 = a1 ? ((float)Q[q_base + d1] * scale) : 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const at::BFloat16* k_ptr = K + kv_head_base;
    const at::BFloat16* v_ptr = V + kv_head_base;

    int j = 0;
    for (; j + 1 <= i; j += 2) {
        float part0 = 0.0f;
        if (a0) part0 += q0 * (float)k_ptr[d0];
        if constexpr (TWO_DIMS) { if (a1) part0 += q1 * (float)k_ptr[d1]; }

        float score0 = wave_reduce_sum_64(part0);

        float m_new0 = fmaxf(m, score0);
        float alpha0 = expf(m - m_new0);
        float beta0  = expf(score0 - m_new0);

        l = l * alpha0 + beta0;
        m = m_new0;
        if (a0) acc0 = acc0 * alpha0 + beta0 * (float)v_ptr[d0];
        if constexpr (TWO_DIMS) { if (a1) acc1 = acc1 * alpha0 + beta0 * (float)v_ptr[d1]; }

        k_ptr += (long long)Hk * D;
        v_ptr += (long long)Hk * D;

        float part1 = 0.0f;
        if (a0) part1 += q0 * (float)k_ptr[d0];
        if constexpr (TWO_DIMS) { if (a1) part1 += q1 * (float)k_ptr[d1]; }

        float score1 = wave_reduce_sum_64(part1);

        float m_new1 = fmaxf(m, score1);
        float alpha1 = expf(m - m_new1);
        float beta1  = expf(score1 - m_new1);

        l = l * alpha1 + beta1;
        m = m_new1;
        if (a0) acc0 = acc0 * alpha1 + beta1 * (float)v_ptr[d0];
        if constexpr (TWO_DIMS) { if (a1) acc1 = acc1 * alpha1 + beta1 * (float)v_ptr[d1]; }

        k_ptr += (long long)Hk * D;
        v_ptr += (long long)Hk * D;
    }

    if (j <= i) {
        float part = 0.0f;
        if (a0) part += q0 * (float)k_ptr[d0];
        if constexpr (TWO_DIMS) { if (a1) part += q1 * (float)k_ptr[d1]; }

        float score = wave_reduce_sum_64(part);

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float beta  = expf(score - m_new);

        l = l * alpha + beta;
        if (a0) acc0 = acc0 * alpha + beta * (float)v_ptr[d0];
        if constexpr (TWO_DIMS) { if (a1) acc1 = acc1 * alpha + beta * (float)v_ptr[d1]; }
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

    TORCH_CHECK(Q.dim() == 4, "Q must be [B, S, Hq, D]");
    TORCH_CHECK(K.dim() == 4, "K must be [B, S, Hk, D]");
    TORCH_CHECK(V.dim() == 4, "V must be [B, S, Hk, D]");

    const int64_t B  = Q.size(0);
    const int64_t S  = Q.size(1);
    const int64_t Hq = Q.size(2);
    const int64_t D  = Q.size(3);

    TORCH_CHECK(K.size(0) == B && V.size(0) == B, "Batch mismatch");
    TORCH_CHECK(K.size(1) == S && V.size(1) == S, "Sequence mismatch");
    TORCH_CHECK(K.size(3) == D && V.size(3) == D, "Head-dim mismatch");

    const int64_t Hk = K.size(2);
    TORCH_CHECK(V.size(2) == Hk, "K/V head mismatch");
    TORCH_CHECK(Hk >= 1, "Hk must be >= 1");
    TORCH_CHECK(D > 0 && D <= 128, "This kernel supports 1 <= D <= 128");
    TORCH_CHECK(S >= 1, "S must be >= 1");
    TORCH_CHECK(Hq >= 1 && B >= 1, "B/Hq must be >= 1");

    auto Qc = Q.contiguous();
    auto Kc = K.contiguous();
    auto Vc = V.contiguous();
    auto O  = torch::empty_like(Qc);

    constexpr int BLOCK = 128; // 2 wavefronts per block
    dim3 block(BLOCK);
    dim3 grid((unsigned int)((S + 1) / 2), (unsigned int)Hq, (unsigned int)B);

    const float scale = 1.0f / std::sqrt((float)D);

    const at::BFloat16* Qp = reinterpret_cast<const at::BFloat16*>(Qc.data_ptr<at::BFloat16>());
    const at::BFloat16* Kp = reinterpret_cast<const at::BFloat16*>(Kc.data_ptr<at::BFloat16>());
    const at::BFloat16* Vp = reinterpret_cast<const at::BFloat16*>(Vc.data_ptr<at::BFloat16>());
    at::BFloat16* Op = reinterpret_cast<at::BFloat16*>(O.data_ptr<at::BFloat16>());

    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (D == 64) {
        sdpa_causal_bf16_d64_kernel_w2<<<grid, block, 0, stream>>>(Qp, Kp, Vp, Op, (int)B, (int)Hq, (int)Hk, (int)S, scale);
    } else if (D == 128) {
        sdpa_causal_bf16_d128_kernel_w2<<<grid, block, 0, stream>>>(Qp, Kp, Vp, Op, (int)B, (int)Hq, (int)Hk, (int)S, scale);
    } else if (D <= 64) {
        sdpa_causal_bf16_generic_kernel_w2<false><<<grid, block, 0, stream>>>(Qp, Kp, Vp, Op, (int)B, (int)Hq, (int)Hk, (int)S, (int)D, scale);
    } else {
        sdpa_causal_bf16_generic_kernel_w2<true><<<grid, block, 0, stream>>>(Qp, Kp, Vp, Op, (int)B, (int)Hq, (int)Hk, (int)S, (int)D, scale);
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
    name="sdpa_causal_bf16_mi300x_wave64_w2_bshd_ext",
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
