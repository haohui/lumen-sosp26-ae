import os
import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Use HIP compiler on AMD GPUs
os.environ["CXX"] = "hipcc"

hip_flash_attn_cpp = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>
#include <vector>
#include <cmath>
#include <cstdint>

static __device__ __forceinline__ float bf16_to_float(uint16_t x) {
    uint32_t u = ((uint32_t)x) << 16;
    return __uint_as_float(u);
}

static __device__ __forceinline__ uint16_t float_to_bf16_rn(float x) {
    uint32_t u = __float_as_uint(x);
    uint32_t lsb = (u >> 16) & 1;
    uint32_t bias = 0x7fff + lsb;
    u += bias;
    return (uint16_t)(u >> 16);
}

constexpr int WARP_SIZE_ = 64;   // AMD wavefront
constexpr int BLOCK_M_   = 4;    // queries per block
constexpr int BLOCK_N_   = 32;   // keys per tile

__global__ void flash_attn_fwd_bf16_kernel(
    const uint16_t* __restrict__ q,   // [B,S,Hq,D]
    const uint16_t* __restrict__ k,   // [B,S,Hk,D]
    const uint16_t* __restrict__ v,   // [B,S,Hk,D]
    uint16_t* __restrict__ out,       // [B,S,Hq,D]
    int B, int S, int Hq, int Hk, int D, float scale
) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE_;
    int lane = tid % WARP_SIZE_;

    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_start = blockIdx.x * BLOCK_M_;
    int q_idx = q_start + warp_id;
    bool q_valid = (q_idx < S);

    int max_q = q_start + BLOCK_M_ - 1;
    if (max_q >= S) max_q = S - 1;

    int kv_h = h % Hk;

    extern __shared__ float smem[];
    float* k_tile = smem;                      // [BLOCK_N_, D]
    float* v_tile = smem + BLOCK_N_ * D;       // [BLOCK_N_, D]

    int d0 = lane;
    int d1 = lane + WARP_SIZE_;

    float q0 = 0.0f, q1 = 0.0f;
    if (q_valid) {
        int64_t q_base = (((int64_t)b * S + q_idx) * Hq + h) * D;
        if (d0 < D) q0 = bf16_to_float(q[q_base + d0]) * scale;
        if (d1 < D) q1 = bf16_to_float(q[q_base + d1]) * scale;
    }

    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;

    for (int n_start = 0; n_start <= max_q; n_start += BLOCK_N_) {
        int elems = BLOCK_N_ * D;
        for (int idx = tid; idx < elems; idx += blockDim.x) {
            int n = idx / D;
            int d = idx - n * D;
            int k_idx = n_start + n;

            float kval = 0.0f, vval = 0.0f;
            if (k_idx < S) {
                int64_t kv_base = (((int64_t)b * S + k_idx) * Hk + kv_h) * D;
                kval = bf16_to_float(k[kv_base + d]);
                vval = bf16_to_float(v[kv_base + d]);
            }
            k_tile[idx] = kval;
            v_tile[idx] = vval;
        }
        __syncthreads();

        int n_lim = BLOCK_N_;
        if (n_start + n_lim > S) n_lim = S - n_start;

        for (int n = 0; n < n_lim; ++n) {
            int k_idx = n_start + n;
            bool valid = q_valid && (k_idx <= q_idx);

            float score = -INFINITY;
            if (valid) {
                float part = 0.0f;
                if (d0 < D) part += q0 * k_tile[n * D + d0];
                if (d1 < D) part += q1 * k_tile[n * D + d1];

                for (int offset = WARP_SIZE_ / 2; offset > 0; offset >>= 1) {
                    part += __shfl_down(part, offset, WARP_SIZE_);
                }
                score = __shfl(part, 0, WARP_SIZE_);
            }

            float m_new = valid ? fmaxf(m, score) : m;
            float alpha = 0.0f;
            if (m_new > -INFINITY) {
                alpha = (m > -INFINITY) ? expf(m - m_new) : 0.0f;
            }
            float p = valid ? expf(score - m_new) : 0.0f;
            float l_new = l * alpha + p;

            float vv0 = (valid && d0 < D) ? v_tile[n * D + d0] : 0.0f;
            float vv1 = (valid && d1 < D) ? v_tile[n * D + d1] : 0.0f;

            acc0 = acc0 * alpha + p * vv0;
            acc1 = acc1 * alpha + p * vv1;

            m = m_new;
            l = l_new;
        }
        __syncthreads();
    }

    if (q_valid) {
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        int64_t out_base = (((int64_t)b * S + q_idx) * Hq + h) * D;
        if (d0 < D) out[out_base + d0] = float_to_bf16_rn(acc0 * inv_l);
        if (d1 < D) out[out_base + d1] = float_to_bf16_rn(acc1 * inv_l);
    }
}

torch::Tensor flash_attn_bf16_hip(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "flash_attn_bf16_hip: tensors must be on HIP device");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "Q must be bf16");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16, "K must be bf16");
    TORCH_CHECK(v.scalar_type() == at::kBFloat16, "V must be bf16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "Q/K/V must be 4D [B,S,H,D]");

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    int64_t B  = q_c.size(0);
    int64_t S  = q_c.size(1);
    int64_t Hq = q_c.size(2);
    int64_t D  = q_c.size(3);

    TORCH_CHECK(k_c.size(0) == B && v_c.size(0) == B, "B mismatch");
    TORCH_CHECK(k_c.size(1) == S && v_c.size(1) == S, "S mismatch");
    TORCH_CHECK(k_c.size(3) == D && v_c.size(3) == D, "D mismatch");

    int64_t Hk = k_c.size(2);
    TORCH_CHECK(v_c.size(2) == Hk, "K/V head mismatch");
    TORCH_CHECK(Hq % Hk == 0, "Hq must be divisible by Hk");
    TORCH_CHECK(D <= 128, "Current kernel supports D <= 128");
    TORCH_CHECK(
        S == 1024 || S == 2048 || S == 4096 || S == 8192 || S == 16384,
        "Supported S: {1024, 2048, 4096, 8192, 16384}"
    );

    auto out = torch::empty_like(q_c);

    const uint16_t* q_ptr = reinterpret_cast<const uint16_t*>(q_c.data_ptr<at::BFloat16>());
    const uint16_t* k_ptr = reinterpret_cast<const uint16_t*>(k_c.data_ptr<at::BFloat16>());
    const uint16_t* v_ptr = reinterpret_cast<const uint16_t*>(v_c.data_ptr<at::BFloat16>());
    uint16_t* out_ptr = reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>());

    float scale = 1.0f / std::sqrt((float)D);

    dim3 block(BLOCK_M_ * WARP_SIZE_, 1, 1);
    dim3 grid((S + BLOCK_M_ - 1) / BLOCK_M_, Hq, B);
    size_t shmem_bytes = (size_t)(2 * BLOCK_N_ * D * sizeof(float));
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    flash_attn_fwd_bf16_kernel<<<grid, block, shmem_bytes, stream>>>(
        q_ptr, k_ptr, v_ptr, out_ptr,
        (int)B, (int)S, (int)Hq, (int)Hk, (int)D, scale
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "flash_attn_fwd_bf16_kernel launch failed: ", hipGetErrorString(err));

    return out;
}
"""

flash_attn_ext = load_inline(
    name="flash_attn_bf16_hip_ext",
    cpp_sources=hip_flash_attn_cpp,
    functions=["flash_attn_bf16_hip"],
    extra_cflags=["-O3"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.flash_attn_ext = flash_attn_ext

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Input/output layout strictly preserved as [B, S, H, D]
        return self.flash_attn_ext.flash_attn_bf16_hip(Q, K, V)


batch_size = 16
num_q_heads = 8
num_kv_heads = 1
sequence_length = int(os.getenv("ATTN_SEQ_LEN", "1024"))
head_dim = 128
supported_sequence_lengths = (1024, 2048, 4096, 8192, 16384)

def get_inputs():
    Q = torch.randn(batch_size, sequence_length, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(batch_size, sequence_length, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(batch_size, sequence_length, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    return [Q, K, V]

def get_init_inputs():
    return []

ANTI_HACK_MANIFEST = {
    "forbidden_api_used": [],
    "main_compute_kernels": ["flash_attn_fwd_bf16_kernel"],
    "fallback_path": False,
}

PERF_MANIFEST = {
    "launch_config": {
        "grid": "((S + BLOCK_M - 1)//BLOCK_M, Hq, B)",
        "block": "(BLOCK_M * 64, 1, 1)",
        "num_warps": 4,
    },
    "tile_sizes": {
        "BLOCK_M": 4,
        "BLOCK_N": 32,
        "BLOCK_K": 128,
    },
    "expected_parallelism": "Each block uses 4 wavefronts (256 threads). Wavefronts process multiple query rows cooperatively; K/V tiles are loaded cooperatively into shared memory and reused across wavefronts."
}
