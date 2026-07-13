#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <climits>
#include <cstdint>
#include <vector>

#include "kernel.h"

torch::Tensor run(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be rank-4 tensors");

  const bool return_cpu = !q.is_cuda();
  torch::Device target_device = q.is_cuda() ? q.device() : torch::Device(torch::kCUDA, 0);

  auto q_dev = q.to(target_device, torch::kBFloat16).contiguous();
  auto k_dev = k.to(target_device, torch::kBFloat16).contiguous();
  auto v_dev = v.to(target_device, torch::kBFloat16).contiguous();

  TORCH_CHECK(q_dev.is_cuda() && k_dev.is_cuda() && v_dev.is_cuda(), "All tensors must be on HIP device after conversion");
  TORCH_CHECK(q_dev.scalar_type() == torch::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(k_dev.scalar_type() == torch::kBFloat16, "k must be bfloat16");
  TORCH_CHECK(v_dev.scalar_type() == torch::kBFloat16, "v must be bfloat16");

  const auto bq = q_dev.size(0);
  const auto sq = q_dev.size(1);
  const auto hq = q_dev.size(2);
  const auto dq = q_dev.size(3);

  const auto bk = k_dev.size(0);
  const auto sk = k_dev.size(1);
  const auto hk = k_dev.size(2);
  const auto dk = k_dev.size(3);

  const auto bv = v_dev.size(0);
  const auto sv = v_dev.size(1);
  const auto hv = v_dev.size(2);
  const auto dv = v_dev.size(3);

  TORCH_CHECK(bq == bk && bq == bv, "batch size mismatch");
  TORCH_CHECK(sq == sk && sq == sv, "sequence length mismatch");
  TORCH_CHECK(dq == dk && dq == dv, "head dim mismatch");
  TORCH_CHECK(hq == 8, "num_q_heads must be 8");
  TORCH_CHECK(dq == 128, "head_dim must be 128");
  TORCH_CHECK(hk == hv, "k/v head count mismatch");
  TORCH_CHECK(hk == 1 || hk == 8, "num_kv_heads must be 1 or 8");

  TORCH_CHECK(bq <= static_cast<int64_t>(INT32_MAX), "batch too large");
  TORCH_CHECK(sq <= static_cast<int64_t>(INT32_MAX), "seq_len too large");

  auto out_dev = torch::empty({bq, sq, hq, dq}, q_dev.options().dtype(torch::kBFloat16));

  int device_index = target_device.index();
  if (device_index < 0) {
    device_index = 0;
  }
  hipError_t set_dev_err = hipSetDevice(device_index);
  TORCH_CHECK(set_dev_err == hipSuccess, "hipSetDevice failed: ", hipGetErrorString(set_dev_err));

  constexpr int rows_per_block = 4;
  dim3 block(64 * rows_per_block, 1, 1);
  dim3 grid((static_cast<unsigned int>(sq) + rows_per_block - 1) / rows_per_block,
            8,
            static_cast<unsigned int>(bq));

  hipStream_t stream = c10::hip::getCurrentHIPStream(device_index).stream();

  const uint16_t* q_ptr = reinterpret_cast<const uint16_t*>(q_dev.data_ptr<c10::BFloat16>());
  const uint16_t* k_ptr = reinterpret_cast<const uint16_t*>(k_dev.data_ptr<c10::BFloat16>());
  const uint16_t* v_ptr = reinterpret_cast<const uint16_t*>(v_dev.data_ptr<c10::BFloat16>());
  uint16_t* o_ptr = reinterpret_cast<uint16_t*>(out_dev.data_ptr<c10::BFloat16>());

  hipError_t launch_err = ksearch_launch_dense_qkv_prefill_causal_h8_kv1or8_d128(
      grid,
      block,
      0,
      stream,
      q_ptr,
      k_ptr,
      v_ptr,
      o_ptr,
      static_cast<int>(sq),
      static_cast<int>(hk),
      static_cast<float>(sm_scale));

  TORCH_CHECK(launch_err == hipSuccess, "Kernel launch failed: ", hipGetErrorString(launch_err));

  if (return_cpu) {
    return out_dev.cpu();
  }
  return out_dev;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("sm_scale"));
}
