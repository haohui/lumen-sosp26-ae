#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <hip/hip_runtime.h>
#include <c10/util/BFloat16.h>

#include "kernel.h"

namespace py = pybind11;

static inline void check_hip(hipError_t err, const char* msg) {
  TORCH_CHECK(err == hipSuccess, msg, ": ", hipGetErrorString(err));
}

torch::Tensor run(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.defined(), "A must be defined");
  TORCH_CHECK(B.defined(), "B must be defined");
  TORCH_CHECK(A.dim() == 2, "A must be 2D, got dim=", A.dim());
  TORCH_CHECK(B.dim() == 2, "B must be 2D, got dim=", B.dim());

  auto A_bf16 = (A.scalar_type() == torch::kBFloat16) ? A : A.to(torch::kBFloat16);
  auto B_bf16 = (B.scalar_type() == torch::kBFloat16) ? B : B.to(torch::kBFloat16);

  TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1),
              "K mismatch: A.shape[1]=", A_bf16.size(1),
              " vs B.shape[1]=", B_bf16.size(1));

  int device_count = 0;
  check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount failed");
  TORCH_CHECK(device_count > 0, "No HIP devices available");

  int device_index = 0;
  if (A_bf16.device().is_cuda()) {
    device_index = A_bf16.get_device();
  } else if (B_bf16.device().is_cuda()) {
    device_index = B_bf16.get_device();
  } else {
    check_hip(hipGetDevice(&device_index), "hipGetDevice failed");
  }

  if (A_bf16.device().is_cuda() && B_bf16.device().is_cuda()) {
    TORCH_CHECK(A_bf16.get_device() == B_bf16.get_device(),
                "A and B must be on the same CUDA/HIP device");
  }

  check_hip(hipSetDevice(device_index), "hipSetDevice failed");
  torch::Device device(torch::kCUDA, device_index);

  auto A_gpu = A_bf16.device().is_cuda() ? A_bf16 : A_bf16.to(device);
  auto B_gpu = B_bf16.device().is_cuda() ? B_bf16 : B_bf16.to(device);

  A_gpu = A_gpu.contiguous();
  B_gpu = B_gpu.contiguous();

  const int64_t M = A_gpu.size(0);
  const int64_t K = A_gpu.size(1);
  const int64_t N = B_gpu.size(0);

  auto C_gpu = torch::zeros({M, N}, A_gpu.options().dtype(torch::kBFloat16).device(device));

  if (M == 0 || N == 0 || K == 0) {
    const bool return_cpu = !(A.device().is_cuda() || B.device().is_cuda());
    return return_cpu ? C_gpu.to(torch::kCPU) : C_gpu;
  }

  const auto* A_ptr = reinterpret_cast<const hip_bfloat16*>(A_gpu.data_ptr<c10::BFloat16>());
  const auto* B_ptr = reinterpret_cast<const hip_bfloat16*>(B_gpu.data_ptr<c10::BFloat16>());
  auto* C_ptr = reinterpret_cast<hip_bfloat16*>(C_gpu.data_ptr<c10::BFloat16>());

  constexpr int BM = 64;
  constexpr int BN = 64;
  dim3 block(16, 16, 1);
  dim3 grid(
      static_cast<unsigned int>((N + BN - 1) / BN),
      static_cast<unsigned int>((M + BM - 1) / BM),
      1);

  hipError_t err = ksearch_launch_gemm_bf16_var_mnk_large(
      grid, block, 0, nullptr, A_ptr, B_ptr, C_ptr,
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
  check_hip(err, "Kernel launch failed");

  const bool return_cpu = !(A.device().is_cuda() || B.device().is_cuda());
  if (return_cpu) {
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
    return C_gpu.to(torch::kCPU);
  }

  return C_gpu;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, py::arg("A"), py::arg("B"), "bf16 GEMM: C = A @ B^T (MI300X HIP)");
}