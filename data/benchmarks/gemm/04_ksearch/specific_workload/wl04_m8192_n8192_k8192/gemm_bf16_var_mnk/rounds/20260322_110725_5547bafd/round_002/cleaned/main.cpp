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

  int device_count = 0;
  hipError_t hip_err = hipGetDeviceCount(&device_count);
  TORCH_CHECK(hip_err == hipSuccess, "hipGetDeviceCount failed: ", hipGetErrorString(hip_err));
  TORCH_CHECK(device_count > 0, "No HIP devices available");

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

  hip_err = hipSetDevice(device_index);
  TORCH_CHECK(hip_err == hipSuccess, "hipSetDevice failed: ", hipGetErrorString(hip_err));

  auto A_dev = A.to(torch::kBFloat16);
  auto B_dev = B.to(torch::kBFloat16);

  const torch::Device target_device(torch::kCUDA, device_index);
  if (!A_dev.is_cuda() || A_dev.get_device() != device_index) {
    A_dev = A_dev.to(target_device);
  }
  if (!B_dev.is_cuda() || B_dev.get_device() != device_index) {
    B_dev = B_dev.to(target_device);
  }

  A_dev = A_dev.contiguous();
  B_dev = B_dev.contiguous();

  auto C_dev = torch::empty({M64, N64}, A_dev.options().dtype(torch::kBFloat16));

  const auto* A_ptr = reinterpret_cast<const hip_bfloat16*>(A_dev.data_ptr<c10::BFloat16>());
  const auto* B_ptr = reinterpret_cast<const hip_bfloat16*>(B_dev.data_ptr<c10::BFloat16>());
  auto* C_ptr = reinterpret_cast<hip_bfloat16*>(C_dev.data_ptr<c10::BFloat16>());

  const int M = static_cast<int>(M64);
  const int N = static_cast<int>(N64);
  const int K = static_cast<int>(K64);

  const dim3 block(GEMM_THREADS_X, GEMM_THREADS_Y, 1);
  const dim3 grid((N + GEMM_BLOCK_N - 1) / GEMM_BLOCK_N,
                  (M + GEMM_BLOCK_M - 1) / GEMM_BLOCK_M,
                  1);

  hipStream_t stream = nullptr;
  hip_err = ksearch_launch_gemm_bf16_var_mnk(grid, block, 0, stream, A_ptr, B_ptr, C_ptr, M, N, K);
  TORCH_CHECK(hip_err == hipSuccess, "Kernel launch failed: ", hipGetErrorString(hip_err));

  if (return_cpu) {
    hip_err = hipStreamSynchronize(stream);
    TORCH_CHECK(hip_err == hipSuccess, "hipStreamSynchronize failed: ", hipGetErrorString(hip_err));
    return C_dev.to(torch::kCPU);
  }

  return C_dev;
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, py::arg("A"), py::arg("B"));
}