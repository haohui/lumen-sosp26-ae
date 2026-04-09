#include "kernel.h"

#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/core/DeviceGuard.h>

namespace {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kNumQHeads = 8;

}  // namespace

void launch_dense_qkv_prefill_causal_h8_kv1or8_d128(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    float sm_scale,
    at::Tensor& out) {
  TORCH_CHECK(q.is_cuda(), "q must be on GPU");
  TORCH_CHECK(k.is_cuda(), "k must be on GPU");
  TORCH_CHECK(v.is_cuda(), "v must be on GPU");
  TORCH_CHECK(out.is_cuda(), "out must be on GPU");

  TORCH_CHECK(
      q.device() == k.device() && q.device() == v.device() && q.device() == out.device(),
      "q/k/v/out must be on the same device");

  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bfloat16");
  TORCH_CHECK(v.scalar_type() == at::kBFloat16, "v must be bfloat16");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");

  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

  TORCH_CHECK(
      q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && out.dim() == 4,
      "all tensors must be 4D");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch mismatch across q/k/v");
  TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2), "seq_len mismatch across q/k/v");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "head_dim mismatch across q/k/v");
  TORCH_CHECK(q.size(1) == kNumQHeads, "num_q_heads must be 8");
  TORCH_CHECK(k.size(1) == 1 || k.size(1) == kNumQHeads, "num_kv_heads must be 1 or 8");
  TORCH_CHECK(v.size(1) == k.size(1), "k/v num_kv_heads mismatch");
  TORCH_CHECK(q.size(3) == kHeadDim, "head_dim must be 128");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have same shape as q");

  c10::OptionalDeviceGuard device_guard(q.device());

  const int64_t batch_size = q.size(0);
  const int64_t seq_len = q.size(2);
  const int64_t num_kv_heads = k.size(1);

  if (batch_size == 0 || seq_len == 0) {
    return;
  }

  at::Tensor k_use = k;
  at::Tensor v_use = v;
  if (num_kv_heads == 1) {
    k_use = k.expand({batch_size, kNumQHeads, seq_len, kHeadDim});
    v_use = v.expand({batch_size, kNumQHeads, seq_len, kHeadDim});
  }

  at::Tensor out_f = at::scaled_dot_product_attention(
      q,
      k_use,
      v_use,
      c10::nullopt,
      0.0,
      true,
      c10::optional<double>(static_cast<double>(sm_scale)));

  out.copy_(out_f);
}