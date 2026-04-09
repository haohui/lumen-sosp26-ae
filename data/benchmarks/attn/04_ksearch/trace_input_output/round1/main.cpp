#include "kernel.h"

#include <ATen/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <vector>

torch::Tensor run(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale) {
  at::NoGradGuard no_grad;

  TORCH_CHECK(q.dim() == 4, "q must be 4D [B, Hq, S, D]");
  TORCH_CHECK(k.dim() == 4, "k must be 4D [B, Hkv, S, D]");
  TORCH_CHECK(v.dim() == 4, "v must be 4D [B, Hkv, S, D]");

  const int64_t bq = q.size(0);
  const int64_t hq = q.size(1);
  const int64_t sq = q.size(2);
  const int64_t dq = q.size(3);

  const int64_t bk = k.size(0);
  const int64_t hk = k.size(1);
  const int64_t sk = k.size(2);
  const int64_t dk = k.size(3);

  const int64_t bv = v.size(0);
  const int64_t hv = v.size(1);
  const int64_t sv = v.size(2);
  const int64_t dv = v.size(3);

  TORCH_CHECK(bq == bk && bq == bv, "Batch size mismatch across q/k/v");
  TORCH_CHECK(sq == sk && sq == sv, "Sequence length mismatch across q/k/v");
  TORCH_CHECK(dq == dk && dq == dv, "Head dim mismatch across q/k/v");

  TORCH_CHECK(hq == 8, "num_q_heads must be 8");
  TORCH_CHECK(hk == 1 || hk == 8, "num_kv_heads must be 1 or 8");
  TORCH_CHECK(hv == hk, "k and v num_kv_heads must match");
  TORCH_CHECK(dq == 128, "head_dim must be 128");

  torch::Device device = q.is_cuda() ? q.device() : torch::Device(torch::kCUDA, 0);
  at::cuda::CUDAGuard device_guard(device);

  auto qd = q.to(device, torch::kBFloat16, false, false).contiguous();
  auto kd = k.to(device, torch::kBFloat16, false, false).contiguous();
  auto vd = v.to(device, torch::kBFloat16, false, false).contiguous();

  auto out = torch::empty({bq, hq, sq, dq}, qd.options().dtype(torch::kBFloat16));

  launch_dense_qkv_prefill_causal_h8_kv1or8_d128(qd, kd, vd, static_cast<float>(sm_scale), out);

  return out.to(torch::kCPU);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "run",
      &run,
      py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("sm_scale"),
      "Dense causal prefill attention (Hq=8, Hkv in {1,8}, D=128) with query-slab single-pass kernel");
}