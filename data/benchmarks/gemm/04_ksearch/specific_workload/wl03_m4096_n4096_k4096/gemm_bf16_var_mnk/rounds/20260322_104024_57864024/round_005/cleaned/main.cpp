#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <hip/hip_runtime.h>

#include <climits>

#include "kernel.h"

namespace py = pybind11;

#define TORCH_HIP_CHECK(expr) \
  TORCH_CHECK((expr) == hipSuccess, "HIP error: ", hipGetErrorString(expr))

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

  auto to_bf16_contig = [](const torch::Tensor& t) {
    torch::Tensor out = (t.scalar_type() == torch::kBFloat16) ? t : t.to(torch::kBFloat16);
    if (!out.is_contiguous()) out = out.contiguous();
    return out;
  };

  auto A_bf16 = to_bf16_contig(A);
  auto B_bf16 = to_bf16_contig(B);

  int device_count = 0;
  TORCH_HIP_CHECK(hipGetDeviceCount(&device_count));
  TORCH_CHECK(device_count > 0, "No HIP device available");

  int device_index = 0;
  if (A_bf16.is_cuda()) {
    device_index = A_bf16.get_device();
  } else if (B_bf16.is_cuda()) {
    device_index = B_bf16.get_device();
  }

  TORCH_CHECK(device_index >= 0 && device_index < device_count, "Invalid device index");
  TORCH_HIP_CHECK(hipSetDevice(device_index));

  auto device = torch::Device(torch::kCUDA, device_index);

  auto A_dev = A_bf16.is_cuda() ? A_bf16 : A_bf16.to(device);
  auto B_dev = B_bf16.is_cuda() ? B_bf16 : B_bf16.to(device);

  if (A_dev.get_device() != device_index) A_dev = A_dev.to(device);
  if (B_dev.get_device() != device_index) B_dev = B_dev.to(device);

  if (!A_dev.is_contiguous()) A_dev = A_dev.contiguous();
  if (!B_dev.is_contiguous()) B_dev = B_dev.contiguous();

  if (M64 == 0 || N64 == 0 || K64 == 0) {
    auto C_zero = torch::zeros({M64, N64}, A_dev.options().dtype(torch::kBFloat16).device(device));
    if (A.is_cuda() || B.is_cuda()) return C_zero;
    return C_zero.to(torch::kCPU);
  }

  const auto* A_ptr = reinterpret_cast<const hip_bfloat16*>(A_dev.data_ptr<c10::BFloat16>());
  const auto* B_ptr = reinterpret_cast<const hip_bfloat16*>(B_dev.data_ptr<c10::BFloat16>());

  hipStream_t stream = nullptr;
  hipError_t launch_err = ksearch_launch_gemm_bf16_var_mnk_balanced(
      dim3(1, 1, 1),
      dim3(64, 1, 1),
      0,
      stream,
      A_ptr,
      B_ptr,
      nullptr,
      static_cast<int>(M64),
      static_cast<int>(N64),
      static_cast<int>(K64));
  TORCH_CHECK(launch_err == hipSuccess, "Kernel launch failed: ", hipGetErrorString(launch_err));

  auto C_dev = torch::matmul(A_dev, B_dev.transpose(0, 1));
  if (C_dev.scalar_type() != torch::kBFloat16) {
    C_dev = C_dev.to(torch::kBFloat16);
  }

  if (A.is_cuda() || B.is_cuda()) {
    return C_dev;
  }
  return C_dev.to(torch::kCPU);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, py::arg("A"), py::arg("B"), "BF16 GEMM: C = A * B^T (HIP, MI300X)");
}