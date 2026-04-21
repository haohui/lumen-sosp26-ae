import functools

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


_EXT_NAME = "kb_p06_rocblas_bf16_v2"


@functools.lru_cache(maxsize=1)
def _load_extension():
    src = r"""
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <ATen/hip/HIPContext.h>
#include <rocblas/rocblas.h>
#include <torch/extension.h>

namespace {

rocblas_handle get_handle() {
    static rocblas_handle handle = nullptr;
    static bool initialized = false;
    if (!initialized) {
        auto st = rocblas_create_handle(&handle);
        if (st != rocblas_status_success) {
            throw std::runtime_error("rocblas_create_handle failed");
        }
        initialized = true;
    }
    return handle;
}

}  // namespace

torch::Tensor gemm_bf16(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "expected CUDA/HIP tensors");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "B must be bf16");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "expected 2D tensors");
    TORCH_CHECK(a.size(1) == b.size(0), "incompatible shapes");

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();

    const int64_t m64 = a_c.size(0);
    const int64_t k64 = a_c.size(1);
    const int64_t n64 = b_c.size(1);
    TORCH_CHECK(
        m64 <= std::numeric_limits<int>::max() &&
        n64 <= std::numeric_limits<int>::max() &&
        k64 <= std::numeric_limits<int>::max(),
        "shape too large"
    );

    const int m = static_cast<int>(m64);
    const int k = static_cast<int>(k64);
    const int n = static_cast<int>(n64);

    auto c = torch::empty({m, n}, a_c.options());

    auto handle = get_handle();
    auto stream = at::hip::getCurrentHIPStream().stream();
    auto st = rocblas_set_stream(handle, stream);
    TORCH_CHECK(st == rocblas_status_success, "rocblas_set_stream failed");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // rocBLAS is column-major. Reading the row-major tensors as transposed
    // column-major matrices produces the requested row-major GEMM result.
    st = rocblas_gemm_ex(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        n,
        m,
        k,
        &alpha,
        b_c.data_ptr(),
        rocblas_datatype_bf16_r,
        n,
        a_c.data_ptr(),
        rocblas_datatype_bf16_r,
        k,
        &beta,
        c.data_ptr(),
        rocblas_datatype_bf16_r,
        n,
        c.data_ptr(),
        rocblas_datatype_bf16_r,
        n,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard,
        0,
        0
    );
    TORCH_CHECK(st == rocblas_status_success, "rocblas_gemm_ex failed");
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_bf16", &gemm_bf16, "BF16 GEMM via rocBLAS");
}
"""
    return load_inline(
        _EXT_NAME,
        cpp_sources=[src],
        functions=None,
        extra_cflags=["-O3"],
        extra_include_paths=["/opt/rocm/include"],
        extra_ldflags=["-L/opt/rocm/lib", "-lrocblas"],
        with_cuda=False,
        verbose=False,
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return _load_extension().gemm_bf16(A, B)
