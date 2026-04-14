#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernel.h"

namespace py = pybind11;

namespace {
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int TM = 4;
constexpr int TN = 4;
}  // namespace

static inline void check_hip(hipError_t err, const char* msg) {
  TORCH_CHECK(err == hipSuccess, msg, " (", hipGetErrorString(err), ")");
}

torch::Tensor run(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.dim() == 2, "A must be 2D, got dim=", A.dim());
  TORCH_CHECK(B.dim() == 2, "B must be 2D, got dim=", B.dim());

  const int64_t M = A.size(0);
  const int64_t K = A.size(1);
  TORCH_CHECK(B.size(1) == K, "K mismatch: A.shape[1]=", K, ", B.shape[1]=", B.size(1));
  const int64_t N = B.size(0);

  auto A_bf16 = (A.scalar_type() == torch::kBFloat16) ? A : A.to(torch::kBFloat16);
  auto B_bf16 = (B.scalar_type() == torch::kBFloat16) ? B : B.to(torch::kBFloat16);

  const bool any_cuda = A_bf16.is_cuda() || B_bf16.is_cuda();
  if (!any_cuda) {
    return torch::matmul(A_bf16, B_bf16.transpose(-1, -2));
  }

  torch::Device device = A_bf16.is_cuda() ? A_bf16.device() : B_bf16.device();
  TORCH_CHECK(device.is_cuda(), "HIP device is required");

  int device_index = device.has_index() ? device.index() : 0;
  int current_device = 0;
  check_hip(hipGetDevice(&current_device), "hipGetDevice failed");
  if (current_device != device_index) {
    check_hip(hipSetDevice(device_index), "hipSetDevice failed");
  }

  if (!A_bf16.is_cuda() || A_bf16.get_device() != device_index) {
    A_bf16 = A_bf16.to(device);
  }
  if (!B_bf16.is_cuda() || B_bf16.get_device() != device_index) {
    B_bf16 = B_bf16.to(device);
  }

  A_bf16 = A_bf16.contiguous();
  B_bf16 = B_bf16.contiguous();

  auto C = torch::empty({M, N}, A_bf16.options().dtype(torch::kBFloat16));

  if (M == 0 || N == 0) {
    return C;
  }

  const uint16_t* A_ptr = reinterpret_cast<const uint16_t*>(A_bf16.data_ptr<c10::BFloat16>());
  const uint16_t* B_ptr = reinterpret_cast<const uint16_t*>(B_bf16.data_ptr<c10::BFloat16>());
  uint16_t* C_ptr = reinterpret_cast<uint16_t*>(C.data_ptr<c10::BFloat16>());

  dim3 block(BN / TN, BM / TM, 1);
  dim3 grid(
      static_cast<uint32_t>((N + BN - 1) / BN),
      static_cast<uint32_t>((M + BM - 1) / BM),
      1);

  hipStream_t stream = c10::hip::getCurrentHIPStream();
  check_hip(
      ksearch_launch_gemm_bf16_var_mnk(grid, block, 0, stream, A_ptr, B_ptr, C_ptr, M, N, K),
      "ksearch_launch_gemm_bf16_var_mnk failed");

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, py::arg("A"), py::arg("B"), "BF16 GEMM: C = A * B^T");
}
