#include <torch/extension.h>
#include <hip/hip_runtime.h>

#include "kernel.h"

namespace py = pybind11;

static inline void check_hip(hipError_t err, const char* where) {
  TORCH_CHECK(err == hipSuccess, where, " failed with HIP error: ", hipGetErrorString(err));
}

torch::Tensor run(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor of shape [M, K]");
  TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor of shape [N, K]");

  torch::Tensor A_t = A.contiguous();
  torch::Tensor B_t = B.contiguous();

  if (A_t.scalar_type() != torch::kBFloat16) {
    A_t = A_t.to(torch::kBFloat16);
  }
  if (B_t.scalar_type() != torch::kBFloat16) {
    B_t = B_t.to(torch::kBFloat16);
  }

  const int64_t M64 = A_t.size(0);
  const int64_t K64 = A_t.size(1);
  const int64_t N64 = B_t.size(0);

  TORCH_CHECK(B_t.size(1) == K64, "Inner K dimension mismatch: A.shape[1] must equal B.shape[1]");

  torch::Device target_device = A_t.is_cuda()
      ? A_t.device()
      : (B_t.is_cuda() ? B_t.device() : torch::Device(torch::kCUDA, 0));

  if (!A_t.is_cuda() || A_t.device() != target_device) {
    A_t = A_t.to(target_device);
  }
  if (!B_t.is_cuda() || B_t.device() != target_device) {
    B_t = B_t.to(target_device);
  }

  A_t = A_t.contiguous();
  B_t = B_t.contiguous();

  auto C_t = torch::empty({M64, N64}, A_t.options().dtype(torch::kBFloat16));

  const int M = static_cast<int>(M64);
  const int N = static_cast<int>(N64);
  const int K = static_cast<int>(K64);

  const dim3 block(16, 16, 1);
  const dim3 grid((N + 31) / 32, (M + 31) / 32, 1);

  const auto* A_ptr = reinterpret_cast<const __hip_bfloat16*>(A_t.data_ptr<c10::BFloat16>());
  const auto* B_ptr = reinterpret_cast<const __hip_bfloat16*>(B_t.data_ptr<c10::BFloat16>());
  auto* C_ptr = reinterpret_cast<__hip_bfloat16*>(C_t.data_ptr<c10::BFloat16>());

  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(pre-launch)");
  check_hip(
      ksearch_launch_gemm_bf16_var_mnk(grid, block, 0, nullptr, A_ptr, B_ptr, C_ptr, M, N, K),
      "ksearch_launch_gemm_bf16_var_mnk");
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(post-launch)");

  return C_t.cpu();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, py::arg("A"), py::arg("B"), "BF16 GEMM: C = A @ B^T");
}