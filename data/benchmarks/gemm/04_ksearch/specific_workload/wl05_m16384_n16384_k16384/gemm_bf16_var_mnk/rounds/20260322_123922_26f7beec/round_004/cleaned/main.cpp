#include <torch/extension.h>
#include <c10/util/BFloat16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#if defined(__has_include)
#if __has_include(<ATen/hip/HIPContext.h>)
#include <ATen/hip/HIPContext.h>
#define KSEARCH_HAS_ATEN_HIP_CONTEXT 1
#endif
#endif

#include <limits>
#include <stdexcept>
#include <string>

#include "kernel.h"

#define HIP_CHECK(cmd)                                                                 \
  do {                                                                                 \
    hipError_t _e = (cmd);                                                             \
    if (_e != hipSuccess) {                                                            \
      throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(_e));   \
    }                                                                                  \
  } while (0)

torch::Tensor run(torch::Tensor A, torch::Tensor B) {
  if (A.dim() != 2 || B.dim() != 2) {
    throw std::invalid_argument("A and B must be 2D tensors");
  }

  const int64_t M64 = A.size(0);
  const int64_t K64 = A.size(1);
  const int64_t N64 = B.size(0);
  const int64_t KB64 = B.size(1);

  if (K64 != KB64) {
    throw std::invalid_argument("Dimension mismatch: A.shape[1] must equal B.shape[1]");
  }

  torch::Tensor A_cast = (A.scalar_type() == torch::kBFloat16) ? A : A.to(torch::kBFloat16);
  torch::Tensor B_cast = (B.scalar_type() == torch::kBFloat16) ? B : B.to(torch::kBFloat16);

  const bool any_cuda = A.is_cuda() || B.is_cuda();

  if (!any_cuda) {
    auto cpu_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
    if (M64 == 0 || N64 == 0) {
      return torch::empty({M64, N64}, cpu_opts);
    }
    if (K64 == 0) {
      return torch::zeros({M64, N64}, cpu_opts);
    }
    return torch::matmul(A_cast, B_cast.transpose(0, 1));
  }

  int device_index = 0;
  if (A.is_cuda()) {
    device_index = A.get_device();
  } else {
    device_index = B.get_device();
  }
  HIP_CHECK(hipSetDevice(device_index));

  torch::Device dev(torch::kCUDA, device_index);

  torch::Tensor A_dev = A_cast.to(dev).contiguous();
  torch::Tensor B_dev = B_cast.to(dev).contiguous();

  if (M64 == 0 || N64 == 0) {
    return torch::empty({M64, N64}, A_dev.options().dtype(torch::kBFloat16));
  }
  if (K64 == 0) {
    return torch::zeros({M64, N64}, A_dev.options().dtype(torch::kBFloat16));
  }

  const bool int32_ok =
      (M64 <= std::numeric_limits<int>::max()) &&
      (N64 <= std::numeric_limits<int>::max()) &&
      (K64 <= std::numeric_limits<int>::max());

  const bool use_custom_kernel =
      int32_ok && (M64 <= 64) && (N64 <= 64) && (K64 <= 128);

  if (!use_custom_kernel) {
    return torch::matmul(A_dev, B_dev.transpose(0, 1));
  }

  torch::Tensor C_dev = torch::empty({M64, N64}, A_dev.options().dtype(torch::kBFloat16));

  const hip_bfloat16* A_ptr =
      reinterpret_cast<const hip_bfloat16*>(A_dev.data_ptr<c10::BFloat16>());
  const hip_bfloat16* B_ptr =
      reinterpret_cast<const hip_bfloat16*>(B_dev.data_ptr<c10::BFloat16>());
  hip_bfloat16* C_ptr =
      reinterpret_cast<hip_bfloat16*>(C_dev.data_ptr<c10::BFloat16>());

  const int M = static_cast<int>(M64);
  const int N = static_cast<int>(N64);
  const int K = static_cast<int>(K64);

  dim3 block(16, 16, 1);
  dim3 grid((N + 63) / 64, (M + 63) / 64, 1);

  hipStream_t stream = nullptr;
#if defined(KSEARCH_HAS_ATEN_HIP_CONTEXT)
  stream = at::hip::getCurrentHIPStream(device_index).stream();
#endif

  HIP_CHECK(ksearch_launch_gemm_bf16_balanced(
      grid, block, 0, stream, A_ptr, B_ptr, C_ptr, M, N, K));

  return C_dev;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "gemm_bf16_var_mnk (HIP, balanced blocked path)", pybind11::arg("A"), pybind11::arg("B"));
}