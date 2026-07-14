#include "kernel.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

torch::Tensor run(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B.scalar_type() == torch::kBFloat16, "B must be bfloat16");
    TORCH_CHECK(A.dim() == 2, "A must be rank-2");
    TORCH_CHECK(B.dim() == 2, "B must be rank-2");

    auto Acontig = A.contiguous();
    auto Bcontig = B.contiguous();

    const int64_t M64 = Acontig.size(0);
    const int64_t K64 = Acontig.size(1);
    const int64_t N64 = Bcontig.size(0);

    TORCH_CHECK(Bcontig.size(1) == K64, "B.shape[1] must equal A.shape[1]");
    TORCH_CHECK(M64 >= 0 && N64 >= 0 && K64 >= 0, "Invalid shapes");
    TORCH_CHECK(M64 <= INT32_MAX && N64 <= INT32_MAX && K64 <= INT32_MAX, "Dimensions too large");

    auto C = torch::empty({M64, N64}, Acontig.options().dtype(torch::kBFloat16));

    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int K = static_cast<int>(K64);

    dim3 block(16, 16, 1);
    dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
    size_t shared_mem = 0;
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const hip_bfloat16* Aptr = reinterpret_cast<const hip_bfloat16*>(Acontig.data_ptr<at::BFloat16>());
    const hip_bfloat16* Bptr = reinterpret_cast<const hip_bfloat16*>(Bcontig.data_ptr<at::BFloat16>());
    hip_bfloat16* Cptr = reinterpret_cast<hip_bfloat16*>(C.data_ptr<at::BFloat16>());

    hipError_t err = ksearch_launch_gemm_bf16_var_mnk(
        grid, block, shared_mem, stream, Aptr, Bptr, Cptr, M, N, K);
    TORCH_CHECK(err == hipSuccess, "ksearch_launch_gemm_bf16_var_mnk failed: ", hipGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "gemm_bf16_var_mnk (HIP, B.T)");
}
