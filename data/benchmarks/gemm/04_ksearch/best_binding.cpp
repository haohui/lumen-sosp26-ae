#include <ATen/hip/HIPContext.h>
#include <c10/util/BFloat16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstdint>

#include "kernel.h"

torch::Tensor run(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a HIP tensor");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B.scalar_type() == at::kBFloat16, "B must be bfloat16");
    TORCH_CHECK(A.get_device() == B.get_device(), "A and B must be on the same device");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(0);
    TORCH_CHECK(
        B.size(1) == K,
        "Shape mismatch: expected A=[M,K], B=[N,K], got A=[",
        M,
        ",",
        K,
        "] B=[",
        N,
        ",",
        B.size(1),
        "]");

    auto A_c = A.contiguous();
    auto B_c = B.contiguous();
    auto C = torch::empty({M, N}, A_c.options());

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    launch_gemm_bf16_var_mnk(
        reinterpret_cast<const void*>(A_c.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const void*>(B_c.data_ptr<c10::BFloat16>()),
        reinterpret_cast<void*>(C.data_ptr<c10::BFloat16>()),
        M,
        N,
        K,
        stream.stream());

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "launch_gemm_bf16_var_mnk failed: ", hipGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "ksearch gemm bf16 var mnk (B as [N,K], computes A @ B^T)");
}
