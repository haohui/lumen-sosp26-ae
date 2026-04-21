import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


os.environ["CXX"] = "hipcc"
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx942")


_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <stdint.h>

namespace {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 32;
constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;
constexpr int TM = 4;
constexpr int TN = 4;

using uint32x4 = uint32_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float to_float(hip_bfloat16 x) {
    return static_cast<float>(x);
}

__device__ __forceinline__ hip_bfloat16 to_bf16(float x) {
    return static_cast<hip_bfloat16>(x);
}

__device__ __forceinline__ uint32x4 make_buffer_resource(const hip_bfloat16* base, uint32_t range_bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<hip_bfloat16*>(base),
        range_bytes,
        0,
        0);
}

__device__ __forceinline__ uint32x4 load_global_b128(
    uint32x4 rsrc,
    uint32_t byte_offset
) {
    return __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset, 0, 0);
}

__device__ __forceinline__ void store_global_b16(
    uint16_t value,
    uint32x4 rsrc,
    uint32_t byte_offset
) {
    __builtin_amdgcn_raw_buffer_store_b16(value, rsrc, byte_offset, 0, 0);
}

__device__ __forceinline__ uint32_t clamp_non_negative_ll(long long x) {
    return static_cast<uint32_t>(x > 0 ? x : 0);
}

__device__ __forceinline__ void zero_a_tile(hip_bfloat16* dst) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst[i] = to_bf16(0.0f);
    }
}

__device__ __forceinline__ void zero_b_tile(hip_bfloat16* dst) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst[i] = to_bf16(0.0f);
    }
}

__device__ __forceinline__ void load_a_vec(
    uint32x4 a_rsrc,
    hip_bfloat16* __restrict__ dst,
    int n,
    int global_row,
    int global_col
) {
    if (global_col + 7 < global_row) {
        zero_a_tile(dst);
        return;
    }

    uint32x4 vec = load_global_b128(
        a_rsrc,
        static_cast<uint32_t>((static_cast<long long>(global_row) * n + global_col) * static_cast<long long>(sizeof(hip_bfloat16))));
    const hip_bfloat16* vals = reinterpret_cast<const hip_bfloat16*>(&vec);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int gc = global_col + i;
        dst[i] = (gc >= global_row) ? vals[i] : to_bf16(0.0f);
    }
}

__device__ __forceinline__ void load_b_vec(
    uint32x4 b_rsrc,
    hip_bfloat16* __restrict__ dst,
    int n,
    int global_row,
    int global_col
) {
    if (global_col + 7 < global_row) {
        zero_b_tile(dst);
        return;
    }

    uint32x4 vec = load_global_b128(
        b_rsrc,
        static_cast<uint32_t>((static_cast<long long>(global_row) * n + global_col) * static_cast<long long>(sizeof(hip_bfloat16))));
    const hip_bfloat16* vals = reinterpret_cast<const hip_bfloat16*>(&vec);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int gc = global_col + i;
        dst[i] = (gc >= global_row) ? vals[i] : to_bf16(0.0f);
    }
}

__device__ __forceinline__ void load_stage(
    uint32x4 a_rsrc,
    uint32x4 b_rsrc,
    hip_bfloat16 As[2][BLOCK_M][BLOCK_K],
    hip_bfloat16 Bs[2][BLOCK_K][BLOCK_N],
    int stage,
    int bm,
    int bn,
    int global_k,
    int n
) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int a_idx = tid * 8;
    int a_row = a_idx / BLOCK_K;
    int a_col = a_idx % BLOCK_K;
    load_a_vec(a_rsrc, &As[stage][a_row][a_col], n, bm + a_row, global_k + a_col);

    int b_idx = tid * 8;
    int b_row = b_idx / BLOCK_N;
    int b_col = b_idx % BLOCK_N;
    load_b_vec(b_rsrc, &Bs[stage][b_row][b_col], n, global_k + b_row, bn + b_col);
}

__device__ __forceinline__ void compute_stage(
    hip_bfloat16 As[2][BLOCK_M][BLOCK_K],
    hip_bfloat16 Bs[2][BLOCK_K][BLOCK_N],
    int stage,
    float acc[TM][TN],
    int row_base,
    int col_base
) {
    #pragma unroll
    for (int kk = 0; kk < BLOCK_K; ++kk) {
        float a_frag[TM];
        float b_frag[TN];
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            a_frag[i] = to_float(As[stage][row_base + i][kk]);
        }
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            b_frag[j] = to_float(Bs[stage][kk][col_base + j]);
        }
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                acc[i][j] += a_frag[i] * b_frag[j];
            }
        }
    }
}

__global__ void upper_triangular_gemm_kernel(
    const hip_bfloat16* __restrict__ A,
    const hip_bfloat16* __restrict__ B,
    hip_bfloat16* __restrict__ C,
    int n
) {
    int bm = blockIdx.y * BLOCK_M;
    int bn = blockIdx.x * BLOCK_N;
    if (bm > bn || bm >= n || bn >= n) {
        return;
    }

    __shared__ hip_bfloat16 As[2][BLOCK_M][BLOCK_K];
    __shared__ hip_bfloat16 Bs[2][BLOCK_K][BLOCK_N];

    int row_base = threadIdx.y * TM;
    int col_base = threadIdx.x * TN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    int k_begin = bm;
    int k_end = std::min(n, bn + BLOCK_N);
    int tiles = (k_end - k_begin + BLOCK_K - 1) / BLOCK_K;
    if (tiles <= 0) {
        return;
    }

    uint32_t matrix_bytes =
        clamp_non_negative_ll(static_cast<long long>(n) * n * sizeof(hip_bfloat16));
    uint32x4 a_rsrc = make_buffer_resource(A, matrix_bytes);
    uint32x4 b_rsrc = make_buffer_resource(B, matrix_bytes);

    load_stage(a_rsrc, b_rsrc, As, Bs, 0, bm, bn, k_begin, n);
    __syncthreads();

    if (tiles > 1) {
        load_stage(a_rsrc, b_rsrc, As, Bs, 1, bm, bn, k_begin + BLOCK_K, n);
        __syncthreads();
    }

    int tile = 0;
    for (; tile + 1 < tiles; tile += 2) {
        compute_stage(As, Bs, tile & 1, acc, row_base, col_base);
        if (tile + 2 < tiles) {
            load_stage(a_rsrc, b_rsrc, As, Bs, tile & 1, bm, bn, k_begin + (tile + 2) * BLOCK_K, n);
        }
        __syncthreads();

        compute_stage(As, Bs, (tile + 1) & 1, acc, row_base, col_base);
        if (tile + 3 < tiles) {
            load_stage(a_rsrc, b_rsrc, As, Bs, (tile + 1) & 1, bm, bn, k_begin + (tile + 3) * BLOCK_K, n);
        }
        __syncthreads();
    }

    if (tile < tiles) {
        compute_stage(As, Bs, tile & 1, acc, row_base, col_base);
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = bm + row_base + i;
        int safe_row = row < n ? row : (n - 1);
        const hip_bfloat16* row_ptr = C + static_cast<long long>(safe_row) * n + safe_row;
        uint32_t row_range_bytes =
            clamp_non_negative_ll(static_cast<long long>(n - row) * sizeof(hip_bfloat16));
        uint32x4 c_row_rsrc = make_buffer_resource(row_ptr, row_range_bytes);

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int col = bn + col_base + j;
            if (col >= row) {
                hip_bfloat16 out = to_bf16(acc[i][j]);
                uint16_t out_bits = *reinterpret_cast<uint16_t*>(&out);
                uint32_t byte_offset = static_cast<uint32_t>((col - row) * static_cast<int>(sizeof(hip_bfloat16)));
                store_global_b16(out_bits, c_row_rsrc, byte_offset);
            }
        }
    }
}

}  // namespace

torch::Tensor upper_triangular_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be on CUDA/HIP device");
    TORCH_CHECK(B.is_cuda(), "B must be on CUDA/HIP device");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bfloat16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "expected 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.sizes() == B.sizes(), "A and B must match");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int n = static_cast<int>(A.size(0));
    auto C = torch::zeros({n, n}, A.options());

    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (n + BLOCK_M - 1) / BLOCK_M);

    const hip_bfloat16* a_ptr = reinterpret_cast<const hip_bfloat16*>(A.data_ptr<at::BFloat16>());
    const hip_bfloat16* b_ptr = reinterpret_cast<const hip_bfloat16*>(B.data_ptr<at::BFloat16>());
    hip_bfloat16* c_ptr = reinterpret_cast<hip_bfloat16*>(C.data_ptr<at::BFloat16>());

    hipLaunchKernelGGL(
        upper_triangular_gemm_kernel,
        grid,
        block,
        0,
        0,
        a_ptr,
        b_ptr,
        c_ptr,
        n
    );

    C = torch::triu(C);
    return C;
}
"""


_EXT = load_inline(
    name="kb_p14_round3_upper_triangular_gemm_v3",
    cpp_sources="torch::Tensor upper_triangular_gemm(torch::Tensor A, torch::Tensor B);",
    cuda_sources=_SRC,
    functions=["upper_triangular_gemm"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "-std=c++17"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._kernel = _EXT

    def forward(self, A, B):
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError("expected 2D tensors")
        if A.shape[1] != B.shape[0]:
            raise ValueError("incompatible shapes")
        if A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1] or A.shape != B.shape:
            raise ValueError("expected matching square matrices")

        A = A.contiguous()
        B = B.contiguous()
        return self._kernel.upper_triangular_gemm(A, B)
