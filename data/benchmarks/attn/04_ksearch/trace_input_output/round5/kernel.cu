#include "kernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kHeadDim = 128;
constexpr int kBlockQ = 4;
constexpr int kWave = 64;

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset, kWave);
  }
  return v;
}

__global__ __launch_bounds__(kWave * kBlockQ, 4)
void dense_qkv_prefill_causal_h8_kv1or8_d128_kernel(
    const c10::BFloat16* __restrict__ q,
    const c10::BFloat16* __restrict__ k,
    const c10::BFloat16* __restrict__ v,
    c10::BFloat16* __restrict__ out,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    float sm_scale) {
  const int lane = threadIdx.x;      // [0, 63]
  const int q_local = threadIdx.y;   // [0, 3]

  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int q_start = blockIdx.x * kBlockQ;
  const int qi = q_start + q_local;

  if (lane >= kWave || q_local >= kBlockQ || b >= batch_size || h >= num_q_heads) {
    return;
  }

  const bool valid_q = (qi < seq_len);
  const int kh = (num_kv_heads == 1) ? 0 : h;
  const int tile_q_end = q_start + (kBlockQ - 1);
  const int max_k = (seq_len - 1 < tile_q_end) ? (seq_len - 1) : tile_q_end;

  __shared__ float k_tile[kHeadDim];
  __shared__ float v_tile[kHeadDim];

  const int d0 = lane;
  const int d1 = lane + kWave;

  const int64_t q_head_base =
      (static_cast<int64_t>(b) * num_q_heads + h) * static_cast<int64_t>(seq_len) * kHeadDim;
  const int64_t kv_head_base =
      (static_cast<int64_t>(b) * num_kv_heads + kh) * static_cast<int64_t>(seq_len) * kHeadDim;

  float q0 = 0.0f;
  float q1 = 0.0f;
  if (valid_q) {
    const int64_t q_row = q_head_base + static_cast<int64_t>(qi) * kHeadDim;
    q0 = static_cast<float>(q[q_row + d0]);
    q1 = static_cast<float>(q[q_row + d1]);
  }

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;

  if (max_k >= 0) {
    for (int kj = 0; kj <= max_k; ++kj) {
      if (q_local == 0) {
        const int64_t kv_row = kv_head_base + static_cast<int64_t>(kj) * kHeadDim;
        k_tile[d0] = static_cast<float>(k[kv_row + d0]);
        k_tile[d1] = static_cast<float>(k[kv_row + d1]);
        v_tile[d0] = static_cast<float>(v[kv_row + d0]);
        v_tile[d1] = static_cast<float>(v[kv_row + d1]);
      }
      __syncthreads();

      const bool active = valid_q && (kj <= qi);

      float dot = 0.0f;
      if (active) {
        dot = q0 * k_tile[d0] + q1 * k_tile[d1];
      }
      dot = wave_sum(dot);

      float alpha = 1.0f;
      float beta = 0.0f;
      if (lane == 0 && active) {
        const float score = dot * sm_scale;
        const float m_new = fmaxf(m, score);
        alpha = isfinite(m) ? expf(m - m_new) : 0.0f;
        beta = expf(score - m_new);
        m = m_new;
        l = l * alpha + beta;
      }

      alpha = __shfl(alpha, 0, kWave);
      beta = __shfl(beta, 0, kWave);

      if (valid_q) {
        acc0 = acc0 * alpha + beta * v_tile[d0];
        acc1 = acc1 * alpha + beta * v_tile[d1];
      }

      __syncthreads();
    }
  }

  if (valid_q) {
    const float denom = __shfl(l, 0, kWave);
    const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

    const int64_t out_row = q_head_base + static_cast<int64_t>(qi) * kHeadDim;
    out[out_row + d0] = c10::BFloat16(acc0 * inv_denom);
    out[out_row + d1] = c10::BFloat16(acc1 * inv_denom);
  }
}

}  // namespace

void launch_dense_qkv_prefill_causal_h8_kv1or8_d128(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    float sm_scale,
    at::Tensor& out) {
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA/ROCm device");
  TORCH_CHECK(k.is_cuda(), "k must be on CUDA/ROCm device");
  TORCH_CHECK(v.is_cuda(), "v must be on CUDA/ROCm device");
  TORCH_CHECK(out.is_cuda(), "out must be on CUDA/ROCm device");

  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bfloat16");
  TORCH_CHECK(v.scalar_type() == at::kBFloat16, "v must be bfloat16");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");

  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && out.dim() == 4, "all tensors must be 4D");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch mismatch across q/k/v");
  TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2), "seq_len mismatch across q/k/v");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "head_dim mismatch across q/k/v");
  TORCH_CHECK(q.size(1) == 8, "num_q_heads must be 8");
  TORCH_CHECK(k.size(1) == 1 || k.size(1) == 8, "num_kv_heads must be 1 or 8");
  TORCH_CHECK(v.size(1) == k.size(1), "k/v num_kv_heads mismatch");
  TORCH_CHECK(q.size(3) == kHeadDim, "head_dim must be 128");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have same shape as q");

  c10::cuda::CUDAGuard device_guard(q.device());

  const int batch_size = static_cast<int>(q.size(0));
  const int num_q_heads = static_cast<int>(q.size(1));
  const int seq_len = static_cast<int>(q.size(2));
  const int num_kv_heads = static_cast<int>(k.size(1));

  if (batch_size == 0 || seq_len == 0) {
    return;
  }

  const auto* q_ptr = q.data_ptr<c10::BFloat16>();
  const auto* k_ptr = k.data_ptr<c10::BFloat16>();
  const auto* v_ptr = v.data_ptr<c10::BFloat16>();
  auto* out_ptr = out.data_ptr<c10::BFloat16>();

  dim3 block(kWave, kBlockQ, 1);
  dim3 grid((seq_len + kBlockQ - 1) / kBlockQ, num_q_heads, batch_size);

  const auto stream = at::cuda::getDefaultCUDAStream(q.get_device());
  dense_qkv_prefill_causal_h8_kv1or8_d128_kernel<<<grid, block, 0, stream.stream()>>>(
      q_ptr, k_ptr, v_ptr, out_ptr, batch_size, num_q_heads, num_kv_heads, seq_len, sm_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}