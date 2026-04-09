#include "kernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kHeadDim = 128;
constexpr int kBlockQ = 8;
constexpr int kBlockK = 16;
constexpr int kWave = 64;
constexpr int kNumQHeads = 8;

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset, kWave);
  }
  return v;
}

template <bool kGqaOne>
__global__ __launch_bounds__(kWave * kBlockQ, 2)
void dense_qkv_prefill_causal_h8_kv1or8_d128_kernel(
    const c10::BFloat16* __restrict__ q,
    const c10::BFloat16* __restrict__ k,
    const c10::BFloat16* __restrict__ v,
    c10::BFloat16* __restrict__ out,
    int batch_size,
    int num_kv_heads,
    int seq_len,
    float sm_scale) {
  const int lane = threadIdx.x;      // [0, 63]
  const int q_local = threadIdx.y;   // [0, kBlockQ-1]
  const int tid = q_local * kWave + lane;

  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int q_start = blockIdx.x * kBlockQ;
  const int qi = q_start + q_local;

  if (b >= batch_size || h >= kNumQHeads) {
    return;
  }

  const bool valid_q = (qi < seq_len);
  const int kh = kGqaOne ? 0 : h;
  const int tile_q_end = q_start + (kBlockQ - 1);
  const int max_k = (seq_len - 1 < tile_q_end) ? (seq_len - 1) : tile_q_end;

  __shared__ float k_tile[kBlockK * kHeadDim];
  __shared__ float v_tile[kBlockK * kHeadDim];

  const int d0 = lane;
  const int d1 = lane + kWave;

  const int64_t q_head_base =
      (static_cast<int64_t>(b) * kNumQHeads + h) * static_cast<int64_t>(seq_len) * kHeadDim;
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
    constexpr int kTileElems = kBlockK * kHeadDim;
    constexpr int kBlockThreads = kWave * kBlockQ;

    for (int tile_start = 0; tile_start <= max_k; tile_start += kBlockK) {
      for (int idx = tid; idx < kTileElems; idx += kBlockThreads) {
        const int kk = idx / kHeadDim;
        const int d = idx - kk * kHeadDim;
        const int kj = tile_start + kk;
        float kval = 0.0f;
        float vval = 0.0f;
        if (kj <= max_k) {
          const int64_t kv_row = kv_head_base + static_cast<int64_t>(kj) * kHeadDim;
          kval = static_cast<float>(k[kv_row + d]);
          vval = static_cast<float>(v[kv_row + d]);
        }
        k_tile[idx] = kval;
        v_tile[idx] = vval;
      }
      __syncthreads();

#pragma unroll
      for (int kk = 0; kk < kBlockK; ++kk) {
        const int kj = tile_start + kk;
        const bool active = valid_q && (kj <= qi) && (kj <= max_k);
        const int tile_off = kk * kHeadDim;

        float dot = 0.0f;
        if (active) {
          dot = q0 * k_tile[tile_off + d0] + q1 * k_tile[tile_off + d1];
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
          acc0 = acc0 * alpha + beta * v_tile[tile_off + d0];
          acc1 = acc1 * alpha + beta * v_tile[tile_off + d1];
        }
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

  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() && q.device() == out.device(),
              "q/k/v/out must be on the same device");

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
  TORCH_CHECK(q.size(1) == kNumQHeads, "num_q_heads must be 8");
  TORCH_CHECK(k.size(1) == 1 || k.size(1) == 8, "num_kv_heads must be 1 or 8");
  TORCH_CHECK(v.size(1) == k.size(1), "k/v num_kv_heads mismatch");
  TORCH_CHECK(q.size(3) == kHeadDim, "head_dim must be 128");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have same shape as q");

  c10::cuda::CUDAGuard device_guard(q.device());

  const int batch_size = static_cast<int>(q.size(0));
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
  dim3 grid((seq_len + kBlockQ - 1) / kBlockQ, kNumQHeads, batch_size);

  const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

  if (num_kv_heads == 1) {
    dense_qkv_prefill_causal_h8_kv1or8_d128_kernel<true>
        <<<grid, block, 0, stream.stream()>>>(
            q_ptr, k_ptr, v_ptr, out_ptr, batch_size, num_kv_heads, seq_len, sm_scale);
  } else {
    dense_qkv_prefill_causal_h8_kv1or8_d128_kernel<false>
        <<<grid, block, 0, stream.stream()>>>(
            q_ptr, k_ptr, v_ptr, out_ptr, batch_size, num_kv_heads, seq_len, sm_scale);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}