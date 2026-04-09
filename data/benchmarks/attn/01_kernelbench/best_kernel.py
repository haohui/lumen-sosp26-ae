import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Use HIP compiler for AMD GPUs
os.environ["CXX"] = "hipcc"

attention_cpp_source = r"""
#include <torch/extension.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <cmath>

template<int HEAD_DIM>
__global__ void causal_mqa_bf16_kernel(
    const at::BFloat16* __restrict__ q,
    const at::BFloat16* __restrict__ k,
    const at::BFloat16* __restrict__ v,
    at::BFloat16* __restrict__ o,
    int B, int Hq, int Hkv, int S) {
  const int tid = threadIdx.x;
  const int q_pos = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;

  if (tid >= HEAD_DIM) return;

  const int kv_h = (Hkv == 1) ? 0 : h;

  const int q_base = (((b * Hq + h) * S + q_pos) * HEAD_DIM);
  const int kv_head_base = ((b * Hkv + kv_h) * S * HEAD_DIM);
  const int o_base = q_base;

  const float q_i = static_cast<float>(q[q_base + tid]);
  float acc = 0.0f;

  float m = -1.0e30f;
  float l = 0.0f;
  const float inv_sqrt_d = rsqrtf(static_cast<float>(HEAD_DIM));

  __shared__ float red[HEAD_DIM];
  __shared__ float scale_prev;
  __shared__ float scale_curr;

  for (int key = 0; key <= q_pos; ++key) {
    const int kv_offset = kv_head_base + key * HEAD_DIM;

    red[tid] = q_i * static_cast<float>(k[kv_offset + tid]);
    __syncthreads();

    for (int stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        red[tid] += red[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float score = red[0] * inv_sqrt_d;
      const float m_new = fmaxf(m, score);
      const float alpha = expf(m - m_new);
      const float beta = expf(score - m_new);
      const float l_new = l * alpha + beta;
      const float inv_l_new = 1.0f / l_new;

      scale_prev = (l == 0.0f) ? 0.0f : (l * alpha) * inv_l_new;
      scale_curr = beta * inv_l_new;

      m = m_new;
      l = l_new;
    }
    __syncthreads();

    const float v_i = static_cast<float>(v[kv_offset + tid]);
    acc = acc * scale_prev + v_i * scale_curr;
    __syncthreads();
  }

  o[o_base + tid] = at::BFloat16(acc);
}

torch::Tensor causal_mqa_attention_hip(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V) {
  TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA/HIP tensor");
  TORCH_CHECK(K.is_cuda(), "K must be a CUDA/HIP tensor");
  TORCH_CHECK(V.is_cuda(), "V must be a CUDA/HIP tensor");

  TORCH_CHECK(Q.scalar_type() == torch::kBFloat16, "Q must be BF16");
  TORCH_CHECK(K.scalar_type() == torch::kBFloat16, "K must be BF16");
  TORCH_CHECK(V.scalar_type() == torch::kBFloat16, "V must be BF16");

  TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "Q, K, V must be rank-4");
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

  const int B = static_cast<int>(Q.size(0));
  const int Hq = static_cast<int>(Q.size(1));
  const int S = static_cast<int>(Q.size(2));
  const int D = static_cast<int>(Q.size(3));

  TORCH_CHECK(D == 128, "This kernel is specialized for head_dim=128");

  TORCH_CHECK(K.size(0) == B && V.size(0) == B, "Batch size mismatch");
  TORCH_CHECK(K.size(2) == S && V.size(2) == S, "Sequence length mismatch");
  TORCH_CHECK(K.size(3) == D && V.size(3) == D, "Head dim mismatch");

  const int Hkv = static_cast<int>(K.size(1));
  TORCH_CHECK(static_cast<int>(V.size(1)) == Hkv, "K/V head mismatch");
  TORCH_CHECK(Hkv == 1 || Hkv == Hq, "K/V heads must be 1 or equal to Q heads");

  auto O = torch::empty_like(Q);

  const dim3 blocks(S, Hq, B);
  const dim3 threads(128);

  causal_mqa_bf16_kernel<128><<<blocks, threads, 0, at::hip::getCurrentHIPStreamMasqueradingAsCUDA()>>>(
      Q.data_ptr<at::BFloat16>(),
      K.data_ptr<at::BFloat16>(),
      V.data_ptr<at::BFloat16>(),
      O.data_ptr<at::BFloat16>(),
      B, Hq, Hkv, S);

  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "causal_mqa_bf16_kernel launch failed: ", hipGetErrorString(err));

  return O;
}
"""

attention_ext = load_inline(
    name="causal_mqa_bf16_ext_v2_streamfix",
    cpp_sources=attention_cpp_source,
    functions=["causal_mqa_attention_hip"],
    extra_cflags=["-O3", "-ffast-math"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.attn = attention_ext

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.attn.causal_mqa_attention_hip(Q, K, V)


batch_size = 16
num_q_heads = 8
num_kv_heads = 1
sequence_length = int(os.getenv("ATTN_SEQ_LEN", "1024"))
head_dim = 128
supported_sequence_lengths = (1024, 2048, 4096, 8192, 16384)


def get_inputs():
    Q = torch.randn(batch_size, num_q_heads, sequence_length, head_dim, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(batch_size, num_kv_heads, sequence_length, head_dim, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(batch_size, num_kv_heads, sequence_length, head_dim, dtype=torch.bfloat16, device="cuda")
    return [Q, K, V]


def get_init_inputs():
    return []
