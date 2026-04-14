# <complete ModelNew code>
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cpp_src = r"""
torch::Tensor fa2_attention_bf16(torch::Tensor q, torch::Tensor k, torch::Tensor v);
"""

_hip_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdint>

template<int BK>
__global__ void fa2_bf16_kernel(
    const at::BFloat16* __restrict__ q,
    const at::BFloat16* __restrict__ k,
    const at::BFloat16* __restrict__ v,
    at::BFloat16* __restrict__ o,
    int B, int S, int Hq, int Hk, int D, int groups, float scale
) {
    const int tid = threadIdx.x;
    const int linear = blockIdx.x;

    const int t = linear % S;
    const int tmp = linear / S;
    const int hq = tmp % Hq;
    const int b = tmp / Hq;
    if (b >= B) return;

    const int hk = hq / groups;

    extern __shared__ float smem[];
    float* red = smem;                  // [blockDim.x]
    float* scores = smem + blockDim.x;  // [BK]
    float* sc = scores + BK;            // [5] => m, l, m_new, alpha, tile_sum

    const int64_t q_base = (((int64_t)b * S + t) * Hq + hq) * D;
    const float qv = (tid < D) ? static_cast<float>(q[q_base + tid]) : 0.0f;
    float acc = 0.0f;

    if (tid == 0) {
        sc[0] = -INFINITY; // m
        sc[1] = 0.0f;      // l
    }
    __syncthreads();

    for (int k0 = 0; k0 <= t; k0 += BK) {
        int tile_n = (t + 1 - k0);
        if (tile_n > BK) tile_n = BK;

        float tile_max = -INFINITY;

        // Tiled QK
        for (int kk = 0; kk < tile_n; ++kk) {
            const int ks = k0 + kk;
            const int64_t k_base = (((int64_t)b * S + ks) * Hk + hk) * D;
            const float kv = (tid < D) ? static_cast<float>(k[k_base + tid]) : 0.0f;

            red[tid] = qv * kv;
            __syncthreads();

            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) red[tid] += red[tid + stride];
                __syncthreads();
            }

            if (tid == 0) {
                float s = red[0] * scale;
                scores[kk] = s;
                tile_max = fmaxf(tile_max, s);
            }
            __syncthreads();
        }

        // Online softmax update
        if (tid == 0) {
            const float m_prev = sc[0];
            const float l_prev = sc[1];
            const float m_new = fmaxf(m_prev, tile_max);
            const float alpha = (isinf(m_prev) && m_prev < 0.0f) ? 0.0f : expf(m_prev - m_new);

            float tile_sum = 0.0f;
            for (int kk = 0; kk < tile_n; ++kk) {
                tile_sum += expf(scores[kk] - m_new);
            }

            sc[2] = m_new;
            sc[3] = alpha;
            sc[4] = tile_sum;
            (void)l_prev;
        }
        __syncthreads();

        const float m_new = sc[2];
        const float alpha = sc[3];

        // Tiled PV accumulation
        if (tid < D) {
            float tile_acc = 0.0f;
            for (int kk = 0; kk < tile_n; ++kk) {
                const int ks = k0 + kk;
                const int64_t v_base = (((int64_t)b * S + ks) * Hk + hk) * D;
                const float vv = static_cast<float>(v[v_base + tid]);
                const float p = expf(scores[kk] - m_new);
                tile_acc += p * vv;
            }
            acc = acc * alpha + tile_acc;
        }

        if (tid == 0) {
            const float l_prev = sc[1];
            const float l_new = l_prev * sc[3] + sc[4];
            sc[0] = sc[2];
            sc[1] = l_new;
        }
        __syncthreads();
    }

    if (tid < D) {
        const float l_final = sc[1];
        const float outv = acc / l_final;
        const int64_t o_base = (((int64_t)b * S + t) * Hq + hq) * D;
        o[o_base + tid] = static_cast<at::BFloat16>(outv);
    }
}

torch::Tensor fa2_attention_bf16(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA(HIP) tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be torch.bfloat16");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be torch.bfloat16");
    TORCH_CHECK(v.scalar_type() == at::kBFloat16, "v must be torch.bfloat16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be [B,S,H,D]");

    auto q_ = q.contiguous();
    auto k_ = k.contiguous();
    auto v_ = v.contiguous();

    const int B = (int)q_.size(0);
    const int S = (int)q_.size(1);
    const int Hq = (int)q_.size(2);
    const int D = (int)q_.size(3);

    TORCH_CHECK(k_.size(0) == B && v_.size(0) == B, "Batch mismatch");
    TORCH_CHECK(k_.size(1) == S && v_.size(1) == S, "Sequence mismatch");
    TORCH_CHECK(k_.size(3) == D && v_.size(3) == D, "Head dim mismatch");

    const int Hk = (int)k_.size(2);
    const int Hv = (int)v_.size(2);
    TORCH_CHECK(Hk == Hv, "K/V head count mismatch");
    TORCH_CHECK(Hq % Hk == 0, "Hq must be divisible by Hk");
    TORCH_CHECK(D > 0 && D <= 256, "Supported head_dim in this kernel: 1..256");

    const int groups = Hq / Hk;
    auto out = torch::empty({B, S, Hq, D}, q_.options());

    int threads = 1;
    while (threads < D) threads <<= 1;
    if (threads < 64) threads = 64;
    if (threads > 256) threads = 256;

    constexpr int BK = 64;
    const int64_t total = (int64_t)B * S * Hq;
    dim3 grid((unsigned int)total);
    dim3 block(threads);
    const size_t shmem = (threads + BK + 5) * sizeof(float);

    const float scale = 1.0f / std::sqrt((float)D);

    c10::cuda::CUDAGuard device_guard(q_.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    fa2_bf16_kernel<BK><<<grid, block, shmem, stream>>>(
        q_.data_ptr<at::BFloat16>(),
        k_.data_ptr<at::BFloat16>(),
        v_.data_ptr<at::BFloat16>(),
        out.data_ptr<at::BFloat16>(),
        B, S, Hq, Hk, D, groups, scale
    );

    return out;
}
"""

_fa2_ext = load_inline(
    name="fa2_bshd_bf16_rocm_ext",
    cpp_sources=_cpp_src,
    cuda_sources=_hip_src,
    functions=["fa2_attention_bf16"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return _fa2_ext.fa2_attention_bf16(Q, K, V)


batch_size = 16
num_q_heads = 8
num_kv_heads = 1
sequence_length = int(os.getenv("ATTN_SEQ_LEN", "1024"))
head_dim = 128
supported_sequence_lengths = (1024, 2048, 4096, 8192, 16384)

def get_inputs():
    device = "cuda"
    Q = torch.randn(batch_size, sequence_length, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    K = torch.randn(batch_size, sequence_length, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    V = torch.randn(batch_size, sequence_length, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    return [Q, K, V]

def get_init_inputs():
    return []
