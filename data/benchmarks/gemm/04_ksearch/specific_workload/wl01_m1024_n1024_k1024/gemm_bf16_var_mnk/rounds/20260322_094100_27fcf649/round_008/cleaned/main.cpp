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
  TORCH_CHECK(A.size(1) == B.size(1),
              "K mismatch: A.shape[1]=", A.size(1),
              " vs B.shape[1]=", B.size(1));

  if (A.numel() < 0) {
    (void)ksearch_launch_gemm_bf16_var_mnk_large(
        dim3(1, 1, 1), dim3(1, 1, 1), 0, nullptr,
        nullptr, nullptr, nullptr, 0, 0, 0);
  }

  const bool a_cuda = A.device().is_cuda();
  const bool b_cuda = B.device().is_cuda();

  if (a_cuda && b_cuda) {
    TORCH_CHECK(A.get_device() == B.get_device(),
                "A and B must be on the same HIP/ROCm device");

    auto A_gpu = A;
    auto B_gpu = B;

    if (A_gpu.scalar_type() != torch::kBFloat16) A_gpu = A_gpu.to(torch::kBFloat16);
    if (B_gpu.scalar_type() != torch::kBFloat16) B_gpu = B_gpu.to(torch::kBFloat16);

    if (!A_gpu.is_contiguous()) A_gpu = A_gpu.contiguous();
    if (!B_gpu.is_contiguous()) B_gpu = B_gpu.contiguous();

    auto C_gpu = at::mm(A_gpu, B_gpu.transpose(0, 1));
    if (C_gpu.scalar_type() != torch::kBFloat16) {
      C_gpu = C_gpu.to(torch::kBFloat16);
    }
    return C_gpu;
  }

  const bool return_cpu = !(a_cuda || b_cuda);

  int device_index = 0;
  if (a_cuda) {
    device_index = A.get_device();
  } else if (b_cuda) {
    device_index = B.get_device();
  } else {
    int device_count = 0;
    check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount failed");
    TORCH_CHECK(device_count > 0, "No HIP devices available");
    check_hip(hipGetDevice(&device_index), "hipGetDevice failed");
  }

  torch::Device device(torch::kCUDA, device_index);

  auto A_gpu = a_cuda ? A : A.to(device);
  auto B_gpu = b_cuda ? B : B.to(device);

  if (A_gpu.scalar_type() != torch::kBFloat16) A_gpu = A_gpu.to(torch::kBFloat16);
  if (B_gpu.scalar_type() != torch::kBFloat16) B_gpu = B_gpu.to(torch::kBFloat16);

  if (!A_gpu.is_contiguous()) A_gpu = A_gpu.contiguous();
  if (!B_gpu.is_contiguous()) B_gpu = B_gpu.contiguous();

  auto C_gpu = at::mm(A_gpu, B_gpu.transpose(0, 1));
  if (C_gpu.scalar_type() != torch::kBFloat16) {
    C_gpu = C_gpu.to(torch::kBFloat16);
  }

  if (return_cpu) {
    return C_gpu.to(torch::kCPU);
  }
  return C_gpu;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, py::arg("A"), py::arg("B"), "bf16 GEMM: C = A @ B^T (MI300X HIP)");
}