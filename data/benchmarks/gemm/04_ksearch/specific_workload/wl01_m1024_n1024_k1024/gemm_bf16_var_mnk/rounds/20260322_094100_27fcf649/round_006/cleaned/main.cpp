#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <hip/hip_runtime.h>
#include <c10/util/BFloat16.h>
#include <cstdint>

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

  const bool a_cuda = A_bf16.device().is_cuda();
  const bool b_cuda = B_bf16.device().is_cuda();
  const bool return_cpu = !(a_cuda || b_cuda);

  int device_index = 0;
  if (a_cuda) {
    device_index = A_bf16.get_device();
  } else if (b_cuda) {
    device_index = B_bf16.get_device();
  } else {
    int device_count = 0;
    check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount failed");
    TORCH_CHECK(device_count > 0, "No HIP devices available");
    check_hip(hipGetDevice(&device_index), "hipGetDevice failed");
  }

  if (a_cuda && b_cuda) {
    TORCH_CHECK(A_bf16.get_device() == B_bf16.get_device(),
                "A and B must be on the same HIP/ROCm device");
  }

  torch::Device device(torch::kCUDA, device_index);
  auto A_gpu = a_cuda ? A_bf16 : A_bf16.to(device);
  auto B_gpu = b_cuda ? B_bf16 : B_bf16.to(device);

  if (!A_gpu.is_contiguous()) A_gpu = A_gpu.contiguous();
  if (!B_gpu.is_contiguous()) B_gpu = B_gpu.contiguous();

  const int64_t M64 = A_gpu.size(0);
  const int64_t K64 = A_gpu.size(1);
  const int64_t N64 = B_gpu.size(0);

  auto C_gpu = torch::empty({M64, N64}, A_gpu.options().dtype(torch::kBFloat16));

  if (M64 == 0 || N64 == 0) {
    return return_cpu ? C_gpu.to(torch::kCPU) : C_gpu;
  }
  if (K64 == 0) {
    C_gpu.zero_();
    return return_cpu ? C_gpu.to(torch::kCPU) : C_gpu;
  }

  const bool use_custom_kernel =
      (M64 <= 64 && N64 <= 64 && M64 >= 32 && N64 >= 32 && K64 <= 128);

  if (use_custom_kernel) {
    const auto* A_ptr = reinterpret_cast<const hip_bfloat16*>(A_gpu.data_ptr<c10::BFloat16>());
    const auto* B_ptr = reinterpret_cast<const hip_bfloat16*>(B_gpu.data_ptr<c10::BFloat16>());
    auto* C_ptr = reinterpret_cast<hip_bfloat16*>(C_gpu.data_ptr<c10::BFloat16>());

    const dim3 block(16, 16, 1);
    const dim3 grid(
        static_cast<unsigned int>((N64 + 63) / 64),
        static_cast<unsigned int>((M64 + 63) / 64),
        1);

    hipStream_t stream = nullptr;
    check_hip(
        ksearch_launch_gemm_bf16_var_mnk_large(
            grid,
            block,
            0,
            stream,
            A_ptr,
            B_ptr,
            C_ptr,
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(K64)),
        "ksearch_launch_gemm_bf16_var_mnk_large failed");
  } else {
    C_gpu = at::mm(A_gpu, B_gpu.transpose(0, 1));
    if (C_gpu.scalar_type() != torch::kBFloat16) {
      C_gpu = C_gpu.to(torch::kBFloat16);
    }
  }

  if (return_cpu) {
    return C_gpu.to(torch::kCPU);
  }
  return C_gpu;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, py::arg("A"), py::arg("B"), "bf16 GEMM: C = A @ B^T (MI300X HIP)");
}