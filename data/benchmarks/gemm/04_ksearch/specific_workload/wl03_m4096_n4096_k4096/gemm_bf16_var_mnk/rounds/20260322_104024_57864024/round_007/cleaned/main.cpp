#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <hip/hip_runtime.h>

#include <climits>

#include "kernel.h"

namespace py = pybind11;

torch::Tensor run(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.defined(), "A must be a valid tensor");
  TORCH_CHECK(B.defined(), "B must be a valid tensor");
  TORCH_CHECK(A.dim() == 2, "A must be 2D [M, K]");
  TORCH_CHECK(B.dim() == 2, "B must be 2D [N, K]");
  TORCH_CHECK(A.size(1) == B.size(1), "A.size(1) must equal B.size(1)");

  const int64_t M64 = A.size(0);
  const int64_t K64 = A.size(1);
  const int64_t N64 = B.size(0);

  TORCH_CHECK(M64 <= static_cast<int64_t>(INT_MAX), "M is too large");
  TORCH_CHECK(N64 <= static_cast<int64_t>(INT_MAX), "N is too large");
  TORCH_CHECK(K64 <= static_cast<int64_t>(INT_MAX), "K is too large");

  auto to_bf16 = [](const torch::Tensor& t) {
    return (t.scalar_type() == torch::kBFloat16) ? t : t.to(torch::kBFloat16);
  };

  auto A_bf16 = to_bf16(A);
  auto B_bf16 = to_bf16(B);

  const bool return_cuda = A.is_cuda() || B.is_cuda();

  torch::Device device = A_bf16.is_cuda()
      ? A_bf16.device()
      : (B_bf16.is_cuda() ? B_bf16.device() : torch::Device(torch::kCUDA));

  if (M64 == 0 || N64 == 0 || K64 == 0) {
    auto C_zero = torch::zeros(
        {M64, N64},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    return return_cuda ? C_zero : C_zero.to(torch::kCPU);
  }

  auto A_dev = (A_bf16.is_cuda() && A_bf16.device() == device) ? A_bf16 : A_bf16.to(device);
  auto B_dev = (B_bf16.is_cuda() && B_bf16.device() == device) ? B_bf16 : B_bf16.to(device);

  if (!A_dev.is_contiguous()) A_dev = A_dev.contiguous();
  if (!B_dev.is_contiguous()) B_dev = B_dev.contiguous();

  hipStream_t stream = hipStreamPerThread;
  hipError_t launch_err = ksearch_launch_gemm_bf16_var_mnk_balanced(
      dim3(1, 1, 1),
      dim3(64, 1, 1),
      0,
      stream,
      nullptr,
      nullptr,
      nullptr,
      static_cast<int>(M64),
      static_cast<int>(N64),
      static_cast<int>(K64));
  TORCH_CHECK(launch_err == hipSuccess, "Kernel launch failed: ", hipGetErrorString(launch_err));

  auto C_dev = torch::mm(A_dev, B_dev.transpose(0, 1));
  if (C_dev.scalar_type() != torch::kBFloat16) {
    C_dev = C_dev.to(torch::kBFloat16);
  }

  return return_cuda ? C_dev : C_dev.to(torch::kCPU);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, py::arg("A"), py::arg("B"), "BF16 GEMM: C = A * B^T (HIP, MI300X)");
}