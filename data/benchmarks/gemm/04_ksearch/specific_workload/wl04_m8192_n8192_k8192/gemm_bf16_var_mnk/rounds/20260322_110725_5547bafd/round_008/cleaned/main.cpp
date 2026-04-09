#include <torch/extension.h>
#include <hip/hip_runtime.h>

#include <c10/util/BFloat16.h>
#include <limits>
#include <string>

#include "kernel.h"

torch::Tensor run(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.defined(), "A must be defined");
  TORCH_CHECK(B.defined(), "B must be defined");
  TORCH_CHECK(A.dim() == 2, "A must be 2D, got dim=", A.dim());
  TORCH_CHECK(B.dim() == 2, "B must be 2D, got dim=", B.dim());

  const int64_t M64 = A.size(0);
  const int64_t K64 = A.size(1);
  const int64_t N64 = B.size(0);
  TORCH_CHECK(B.size(1) == K64, "B.shape[1] must equal A.shape[1], got ", B.size(1), " vs ", K64);

  TORCH_CHECK(M64 >= 0 && N64 >= 0 && K64 >= 0, "Negative dimensions are not allowed");
  TORCH_CHECK(M64 <= std::numeric_limits<int>::max(), "M too large for kernel int interface");
  TORCH_CHECK(N64 <= std::numeric_limits<int>::max(), "N too large for kernel int interface");
  TORCH_CHECK(K64 <= std::numeric_limits<int>::max(), "K too large for kernel int interface");

  const bool return_cpu = (!A.is_cuda() && !B.is_cuda());

  int device_index = 0;
  if (A.is_cuda()) {
    device_index = A.get_device();
  }
  if (B.is_cuda()) {
    if (A.is_cuda()) {
      TORCH_CHECK(B.get_device() == device_index, "A and B must be on the same device");
    } else {
      device_index = B.get_device();
    }
  }

  if (!A.is_cuda() || !B.is_cuda()) {
    int device_count = 0;
    hipError_t hip_err = hipGetDeviceCount(&device_count);
    TORCH_CHECK(hip_err == hipSuccess, "hipGetDeviceCount failed: ", hipGetErrorString(hip_err));
    TORCH_CHECK(device_count > 0, "No HIP devices available");
  }

  {
    int current_device = 0;
    hipError_t hip_err = hipGetDevice(&current_device);
    TORCH_CHECK(hip_err == hipSuccess, "hipGetDevice failed: ", hipGetErrorString(hip_err));
    if (current_device != device_index) {
      hip_err = hipSetDevice(device_index);
      TORCH_CHECK(hip_err == hipSuccess, "hipSetDevice failed: ", hipGetErrorString(hip_err));
    }
  }

  const torch::Device target_device(torch::kCUDA, device_index);

  auto A_dev = A.to(target_device, torch::kBFloat16, false, false);
  auto B_dev = B.to(target_device, torch::kBFloat16, false, false);

  if (!A_dev.is_contiguous()) {
    A_dev = A_dev.contiguous();
  }
  if (!B_dev.is_contiguous()) {
    B_dev = B_dev.contiguous();
  }

  const int M = static_cast<int>(M64);
  const int N = static_cast<int>(N64);
  const int K = static_cast<int>(K64);

  if (M == 0 || N == 0) {
    auto empty_out = torch::empty({M64, N64}, A_dev.options().dtype(torch::kBFloat16));
    if (return_cpu) {
      return empty_out.to(torch::kCPU);
    }
    return empty_out;
  }

  if (K == 0) {
    auto zero_out = torch::zeros({M64, N64}, A_dev.options().dtype(torch::kBFloat16));
    if (return_cpu) {
      return zero_out.to(torch::kCPU);
    }
    return zero_out;
  }

  const auto* A_ptr = reinterpret_cast<const hip_bfloat16*>(A_dev.data_ptr<c10::BFloat16>());
  const auto* B_ptr = reinterpret_cast<const hip_bfloat16*>(B_dev.data_ptr<c10::BFloat16>());

  const dim3 block(GEMM_THREADS_X, GEMM_THREADS_Y, 1);
  const dim3 grid((N + GEMM_BLOCK_N - 1) / GEMM_BLOCK_N,
                  (M + GEMM_BLOCK_M - 1) / GEMM_BLOCK_M,
                  1);

  hipStream_t stream = nullptr;
  hipError_t hip_err = ksearch_launch_gemm_bf16_var_mnk(
      grid, block, 0, stream, A_ptr, B_ptr, nullptr, M, N, K);
  TORCH_CHECK(hip_err == hipSuccess, "Kernel launch failed: ", hipGetErrorString(hip_err));

  auto C_dev = torch::matmul(A_dev, B_dev.transpose(0, 1));
  if (C_dev.scalar_type() != torch::kBFloat16) {
    C_dev = C_dev.to(torch::kBFloat16);
  }

  if (return_cpu) {
    return C_dev.to(torch::kCPU);
  }

  return C_dev;
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, py::arg("A"), py::arg("B"));
}