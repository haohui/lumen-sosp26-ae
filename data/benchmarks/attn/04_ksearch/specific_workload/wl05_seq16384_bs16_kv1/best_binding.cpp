#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>

#include "kernel.h"
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;

torch::Tensor run(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be rank-4 tensors");

  const int64_t bq = q.size(0), hq = q.size(1), sq = q.size(2), dq = q.size(3);
  const int64_t bk = k.size(0), hk = k.size(1), sk = k.size(2), dk = k.size(3);
  const int64_t bv = v.size(0), hv = v.size(1), sv = v.size(2), dv = v.size(3);

  TORCH_CHECK(bq == 16, "batch_size must be 16");
  TORCH_CHECK(hq == 8, "num_q_heads must be 8");
  TORCH_CHECK(hk == 1 && hv == 1, "num_kv_heads must be 1");
  TORCH_CHECK(dq == 128 && dk == 128 && dv == 128, "head_dim must be 128");
  TORCH_CHECK(bq == bk && bk == bv, "batch sizes must match");
  TORCH_CHECK(sq == sk && sk == sv, "sequence lengths must match");
  TORCH_CHECK(dq == dk && dk == dv, "head dims must match");

  const bool q_was_cuda = q.is_cuda();

  if (sq == 0) {
    auto opts = q.options().dtype(torch::kBFloat16);
    if (!q_was_cuda) {
      opts = opts.device(torch::kCPU);
    }
    return torch::empty({bq, hq, sq, dq}, opts);
  }

  int dev_index = 0;
  if (q.is_cuda()) {
    dev_index = q.get_device();
  } else if (k.is_cuda()) {
    dev_index = k.get_device();
  } else if (v.is_cuda()) {
    dev_index = v.get_device();
  } else {
    hipError_t get_dev_err = hipGetDevice(&dev_index);
    TORCH_CHECK(get_dev_err == hipSuccess, "hipGetDevice failed: ", hipGetErrorString(get_dev_err));
  }

  int cur_dev = -1;
  hipError_t cur_dev_err = hipGetDevice(&cur_dev);
  TORCH_CHECK(cur_dev_err == hipSuccess, "hipGetDevice failed: ", hipGetErrorString(cur_dev_err));
  if (cur_dev != dev_index) {
    hipError_t set_dev_err = hipSetDevice(dev_index);
    TORCH_CHECK(set_dev_err == hipSuccess, "hipSetDevice failed: ", hipGetErrorString(set_dev_err));
  }

  auto dev_opts = torch::TensorOptions()
                      .device(torch::Device(torch::kCUDA, dev_index))
                      .dtype(torch::kBFloat16);

  auto prepare_tensor = [&](const torch::Tensor& t) -> torch::Tensor {
    torch::Tensor x = t;
    if (!x.is_cuda() || x.get_device() != dev_index || x.scalar_type() != torch::kBFloat16) {
      x = x.to(dev_opts, /*non_blocking=*/true, /*copy=*/false);
    }
    if (!x.is_contiguous()) {
      x = x.contiguous();
    }
    return x;
  };

  auto q_dev = prepare_tensor(q);
  auto k_dev = prepare_tensor(k);
  auto v_dev = prepare_tensor(v);
  auto out_dev = torch::empty({bq, hq, sq, dq}, dev_opts);

  static_assert(sizeof(c10::BFloat16) == sizeof(hip_bfloat16), "BFloat16 size mismatch");

  const hip_bfloat16* q_ptr =
      reinterpret_cast<const hip_bfloat16*>(q_dev.data_ptr<c10::BFloat16>());
  const hip_bfloat16* k_ptr =
      reinterpret_cast<const hip_bfloat16*>(k_dev.data_ptr<c10::BFloat16>());
  const hip_bfloat16* v_ptr =
      reinterpret_cast<const hip_bfloat16*>(v_dev.data_ptr<c10::BFloat16>());
  hip_bfloat16* out_ptr =
      reinterpret_cast<hip_bfloat16*>(out_dev.data_ptr<c10::BFloat16>());

  dim3 grid(static_cast<unsigned int>((sq + 3) / 4), static_cast<unsigned int>(bq), 1);
  dim3 block(512, 1, 1);

  hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  hipError_t launch_err = ksearch_launch_dense_qkv_prefill_causal_h8_kv1_d128_bs16(
      grid,
      block,
      0,
      stream,
      q_ptr,
      k_ptr,
      v_ptr,
      out_ptr,
      static_cast<int>(sq),
      static_cast<float>(sm_scale));

  TORCH_CHECK(launch_err == hipSuccess, "Kernel launch failed: ", hipGetErrorString(launch_err));

  if (q_was_cuda) {
    return out_dev;
  }
  return out_dev.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kBFloat16));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("sm_scale"));
}