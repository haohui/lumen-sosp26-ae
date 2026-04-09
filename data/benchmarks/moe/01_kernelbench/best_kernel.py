#!/usr/bin/env python3
from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

BLOCK_N = 128
BLOCK_K = 128
ROUTE_GROUP_SIZE = 32

# ANTI_HACK_MANIFEST: {"forbidden_api_used": [], "main_compute_kernels": ["fused_moe_dispatch_compute_combine_kernel"], "single_runtime_op": true}
ANTI_HACK_MANIFEST = {
    "forbidden_api_used": [],
    "main_compute_kernels": ["fused_moe_dispatch_compute_combine_kernel"],
    "single_runtime_op": True,
}


@dataclass
class MoeConfig:
    dim: int = 7168
    inter_dim: int = 2048
    experts: int = 32
    topk: int = 4


_MOE_EXT = None


def _get_moe_ext():
    global _MOE_EXT
    if _MOE_EXT is not None:
        return _MOE_EXT

    os.environ.setdefault("CXX", "hipcc")

    cpp_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/HIPStream.h>
#include <vector>
#include <cstdint>
#include <cmath>

constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 128;

__device__ __forceinline__ float silu_device(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void fused_moe_dispatch_compute_combine_kernel(
    const float* __restrict__ input_f,        // [T, D]
    const float* __restrict__ w1_f,           // [E, 2I, D]
    const float* __restrict__ w2_f,           // [E, D, I]
    const float* __restrict__ topk_w,         // [T, K]
    const int32_t* __restrict__ topk_ids,     // [T, K]
    const float* __restrict__ input_scale,    // [T, D/128]
    const float* __restrict__ fc1_scale,      // [E, (2I/128)*(D/128)]
    const float* __restrict__ fc2_scale,      // [E, (D/128)*(I/128)]
    float* __restrict__ out,                  // [T, D], fp32 accumulate
    int T, int D, int E, int K, int I,
    int in_scale_cols,
    int fc1_cols,
    int fc2_cols
) {
    int route = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    int t = route / K;
    int k = route - t * K;
    if (t >= T) return;

    int e = topk_ids[t * K + k];
    if (e < 0 || e >= E) return;

    float rw = topk_w[t * K + k];
    if (rw == 0.0f) return;

    extern __shared__ float smem[];
    float* x_scaled = smem;         // D
    float* hidden = smem + D;       // I

    // 1) Dequant input token blocks into shared memory
    for (int d = tid; d < D; d += blockDim.x) {
        int cb = d / BLOCK_K;
        float s = input_scale[t * in_scale_cols + cb];
        x_scaled[d] = input_f[t * D + d] * s;
    }
    __syncthreads();

    // 2) Expert FC1 + SiLU*Up -> hidden[I]
    int rows1 = I * 2;
    int col_blocks = D / BLOCK_K;
    int fc1_base_e = e * fc1_cols;

    for (int i = tid; i < I; i += blockDim.x) {
        float gate = 0.0f;
        float up = 0.0f;

        int rb_gate = i / BLOCK_N;
        int rb_up = (i + I) / BLOCK_N;

        const float* w1_gate_row = w1_f + ((e * rows1 + i) * D);
        const float* w1_up_row   = w1_f + ((e * rows1 + (i + I)) * D);

        int gate_scale_base = fc1_base_e + rb_gate * col_blocks;
        int up_scale_base   = fc1_base_e + rb_up * col_blocks;

        for (int cb = 0; cb < col_blocks; ++cb) {
            float sg = fc1_scale[gate_scale_base + cb];
            float su = fc1_scale[up_scale_base + cb];
            int d0 = cb * BLOCK_K;

            #pragma unroll 4
            for (int dk = 0; dk < BLOCK_K; ++dk) {
                int d = d0 + dk;
                float x = x_scaled[d];
                gate += x * w1_gate_row[d] * sg;
                up   += x * w1_up_row[d]   * su;
            }
        }

        hidden[i] = silu_device(gate) * up;
    }
    __syncthreads();

    // 3) Expert FC2 and weighted combine directly into out[t, :]
    int i_blocks = I / BLOCK_K;
    int fc2_base_e = e * fc2_cols;

    for (int d_out = tid; d_out < D; d_out += blockDim.x) {
        float acc = 0.0f;
        const float* w2_row = w2_f + ((e * D + d_out) * I);

        int rb = d_out / BLOCK_N;
        int scale_base = fc2_base_e + rb * i_blocks;

        for (int cb = 0; cb < i_blocks; ++cb) {
            float s2 = fc2_scale[scale_base + cb];
            int i0 = cb * BLOCK_K;

            #pragma unroll 4
            for (int ik = 0; ik < BLOCK_K; ++ik) {
                int i = i0 + ik;
                acc += hidden[i] * w2_row[i] * s2;
            }
        }

        atomicAdd(out + t * D + d_out, acc * rw);
    }
}

torch::Tensor moe_fused_hip(
    torch::Tensor input_q,
    torch::Tensor w1_q,
    torch::Tensor w2_q,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor input_scale,
    torch::Tensor fc1_scale,
    torch::Tensor fc2_scale
) {
    TORCH_CHECK(input_q.is_cuda(), "input_q must be CUDA/HIP tensor");
    TORCH_CHECK(w1_q.is_cuda() && w2_q.is_cuda(), "weights must be CUDA/HIP tensors");
    TORCH_CHECK(topk_weights.is_cuda() && topk_ids.is_cuda(), "routing tensors must be CUDA/HIP tensors");
    TORCH_CHECK(input_scale.is_cuda() && fc1_scale.is_cuda() && fc2_scale.is_cuda(), "scale tensors must be CUDA/HIP tensors");

    auto dev = input_q.device();
    TORCH_CHECK(w1_q.device() == dev && w2_q.device() == dev, "all tensors must be on same device");
    TORCH_CHECK(topk_weights.device() == dev && topk_ids.device() == dev, "all tensors must be on same device");
    TORCH_CHECK(input_scale.device() == dev && fc1_scale.device() == dev && fc2_scale.device() == dev, "all tensors must be on same device");

    auto input_f = input_q.contiguous().to(torch::kFloat);
    auto w1_f = w1_q.contiguous().to(torch::kFloat);
    auto w2_f = w2_q.contiguous().to(torch::kFloat);
    auto topk_w = topk_weights.contiguous().to(torch::kFloat);
    auto in_scale_f = input_scale.contiguous().to(torch::kFloat);
    auto fc1_scale_f = fc1_scale.contiguous().to(torch::kFloat);
    auto fc2_scale_f = fc2_scale.contiguous().to(torch::kFloat);

    torch::Tensor topk_i32;
    if (topk_ids.scalar_type() == torch::kInt) {
        topk_i32 = topk_ids.contiguous();
    } else {
        TORCH_CHECK(topk_ids.scalar_type() == torch::kLong, "topk_ids must be int32 or int64");
        topk_i32 = topk_ids.contiguous().to(torch::kInt);
    }

    TORCH_CHECK(input_f.dim() == 2, "input must be [tokens, dim]");
    TORCH_CHECK(w1_f.dim() == 3 && w2_f.dim() == 3, "weights must be rank-3");
    TORCH_CHECK(topk_w.dim() == 2 && topk_i32.dim() == 2, "topk tensors must be [tokens, topk]");

    const int64_t T64 = input_f.size(0);
    const int64_t D64 = input_f.size(1);
    const int64_t E64 = w1_f.size(0);
    const int64_t R1_64 = w1_f.size(1);
    const int64_t D_w1_64 = w1_f.size(2);
    const int64_t D_w2_64 = w2_f.size(1);
    const int64_t I64 = w2_f.size(2);
    const int64_t K64 = topk_w.size(1);

    TORCH_CHECK(D_w1_64 == D64, "w1 last dim mismatch");
    TORCH_CHECK(D_w2_64 == D64, "w2 middle dim mismatch");
    TORCH_CHECK(R1_64 == 2 * I64, "w1 rows must be 2*inter_dim");
    TORCH_CHECK(topk_w.size(0) == T64 && topk_i32.size(0) == T64 && topk_i32.size(1) == K64, "topk shape mismatch");
    TORCH_CHECK(w2_f.size(0) == E64, "expert count mismatch");
    TORCH_CHECK((D64 % BLOCK_K) == 0 && (D64 % BLOCK_N) == 0, "dim must be divisible by 128");
    TORCH_CHECK((I64 % BLOCK_K) == 0 && ((2 * I64) % BLOCK_N) == 0, "inter_dim must match 128-block format");

    const int in_scale_cols = (int)(D64 / BLOCK_K);
    const int fc1_cols_expected = (int)(((2 * I64) / BLOCK_N) * (D64 / BLOCK_K));
    const int fc2_cols_expected = (int)(((D64) / BLOCK_N) * (I64 / BLOCK_K));

    TORCH_CHECK(input_scale.dim() == 2 && input_scale.size(0) == T64 && input_scale.size(1) == in_scale_cols, "input_scale shape mismatch");
    TORCH_CHECK(fc1_scale.dim() == 2 && fc1_scale.size(0) == E64 && fc1_scale.size(1) == fc1_cols_expected, "fc1_scale shape mismatch");
    TORCH_CHECK(fc2_scale.dim() == 2 && fc2_scale.size(0) == E64 && fc2_scale.size(1) == fc2_cols_expected, "fc2_scale shape mismatch");

    auto out_f = torch::zeros({T64, D64}, input_f.options().dtype(torch::kFloat));
    if (T64 == 0 || K64 == 0) {
        return out_f.to(torch::kBFloat16);
    }

    const int T = (int)T64;
    const int D = (int)D64;
    const int E = (int)E64;
    const int K = (int)K64;
    const int I = (int)I64;
    const int fc1_cols = (int)fc1_cols_expected;
    const int fc2_cols = (int)fc2_cols_expected;

    const int threads = 256;
    const uint32_t routes = (uint32_t)(T * K);
    const size_t shmem = (size_t)(D + I) * sizeof(float);
    TORCH_CHECK(shmem <= 64 * 1024, "shared memory requirement exceeds 64KB");
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    hipLaunchKernelGGL(
        fused_moe_dispatch_compute_combine_kernel,
        dim3(routes),
        dim3(threads),
        shmem,
        stream,
        input_f.data_ptr<float>(),
        w1_f.data_ptr<float>(),
        w2_f.data_ptr<float>(),
        topk_w.data_ptr<float>(),
        topk_i32.data_ptr<int32_t>(),
        in_scale_f.data_ptr<float>(),
        fc1_scale_f.data_ptr<float>(),
        fc2_scale_f.data_ptr<float>(),
        out_f.data_ptr<float>(),
        T, D, E, K, I,
        in_scale_cols,
        fc1_cols,
        fc2_cols
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "fused_moe_dispatch_compute_combine_kernel failed: ", hipGetErrorString(err));

    return out_f.to(torch::kBFloat16);
}
"""
    name = "moe_fused_hip_" + hashlib.md5(cpp_source.encode("utf-8")).hexdigest()[:16]
    _MOE_EXT = load_inline(
        name=name,
        cpp_sources=cpp_source,
        functions=["moe_fused_hip"],
        extra_cflags=["-O3"],
        verbose=False,
    )
    return _MOE_EXT


def make_inputs(
    *,
    tokens: int,
    cfg: MoeConfig,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    if cfg.dim % BLOCK_K != 0 or cfg.dim % BLOCK_N != 0:
        raise ValueError(f"dim must be divisible by {BLOCK_N}/{BLOCK_K}, got {cfg.dim}")
    if cfg.inter_dim % BLOCK_K != 0 or (cfg.inter_dim * 2) % BLOCK_N != 0:
        raise ValueError(f"inter_dim must match blockshape, got {cfg.inter_dim}")
    if cfg.topk > cfg.experts:
        raise ValueError(f"topk ({cfg.topk}) must be <= experts ({cfg.experts})")

    g = torch.Generator(device=str(device))
    g.manual_seed(seed + tokens)

    input_q = (
        torch.randn((tokens, cfg.dim), dtype=torch.float32, device=device, generator=g) * 1.0 + 0.1
    ).to(torch.float8_e4m3fnuz)
    w1_q = torch.randn(
        (cfg.experts, cfg.inter_dim * 2, cfg.dim),
        dtype=torch.float32,
        device=device,
        generator=g,
    ).mul_(8.0).to(torch.float8_e4m3fnuz)
    w2_q = torch.randn(
        (cfg.experts, cfg.dim, cfg.inter_dim),
        dtype=torch.float32,
        device=device,
        generator=g,
    ).mul_(8.0).to(torch.float8_e4m3fnuz)

    def _rand_pos(shape: tuple[int, ...], mean: float, std: float) -> torch.Tensor:
        x = torch.randn(shape, dtype=torch.float32, device=device, generator=g) * std + mean
        return x.clamp_min(1e-8)

    input_scale = _rand_pos((tokens, cfg.dim // BLOCK_K), mean=1e-1, std=2e-2)
    fc1_scale = _rand_pos(
        (cfg.experts, ((cfg.inter_dim * 2) // BLOCK_N) * (cfg.dim // BLOCK_K)),
        mean=1e-2,
        std=2e-3,
    )
    fc2_scale = _rand_pos(
        (cfg.experts, (cfg.dim // BLOCK_N) * (cfg.inter_dim // BLOCK_K)),
        mean=1e-2,
        std=2e-3,
    )

    scores = torch.randn((tokens, cfg.experts), dtype=torch.float32, device=device, generator=g)
    topk_val, topk_idx = torch.topk(scores, k=cfg.topk, dim=-1, largest=True, sorted=True)
    topk_ids = topk_idx.to(torch.int32)
    topk_weights = torch.softmax(topk_val, dim=-1).to(torch.float32)

    return {
        "input_q": input_q,
        "w1_q": w1_q,
        "w2_q": w2_q,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "input_scale": input_scale,
        "fc1_scale": fc1_scale,
        "fc2_scale": fc2_scale,
    }


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        raw_cfg = globals().get("EVAL_CONFIG", {})
        cfg = raw_cfg if isinstance(raw_cfg, dict) else {}

        def as_int(key: str, default: int, minimum: int = 1) -> int:
            raw = cfg.get(key, default)
            try:
                val = int(raw)
            except Exception:
                val = int(default)
            return max(minimum, val)

        self.cfg = MoeConfig(
            dim=as_int("dim", 7168, minimum=128),
            inter_dim=as_int("inter_dim", 2048, minimum=128),
            experts=as_int("experts", 32, minimum=1),
            topk=as_int("topk", 4, minimum=1),
        )
        if self.cfg.topk > self.cfg.experts:
            self.cfg.topk = self.cfg.experts
        self.moe_ext = _get_moe_ext()

    def forward(
        self,
        input_q: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        input_scale: torch.Tensor,
        fc1_scale: torch.Tensor,
        fc2_scale: torch.Tensor,
    ) -> torch.Tensor:
        return self.moe_ext.moe_fused_hip(
            input_q,
            w1_q,
            w2_q,
            topk_weights,
            topk_ids,
            input_scale,
            fc1_scale,
            fc2_scale,
        )


Model = ModelNew


def get_init_inputs():
    return []


def get_inputs():
    raw_cfg = globals().get("EVAL_CONFIG", {})
    cfg = raw_cfg if isinstance(raw_cfg, dict) else {}

    def as_int(key: str, default: int, minimum: int = 1) -> int:
        raw = cfg.get(key, default)
        try:
            val = int(raw)
        except Exception:
            val = int(default)
        return max(minimum, val)

    task_cfg = MoeConfig(
        dim=as_int("dim", 7168, minimum=128),
        inter_dim=as_int("inter_dim", 2048, minimum=128),
        experts=as_int("experts", 32, minimum=1),
        topk=as_int("topk", 4, minimum=1),
    )
    if task_cfg.topk > task_cfg.experts:
        task_cfg.topk = task_cfg.experts

    tokens = as_int("tokens", 1024, minimum=1)
    seed = as_int("seed", 20260317, minimum=0)
    d = str(cfg.get("device", "cuda:0")).strip()
    if not d:
        d = "cuda:0"
    if d.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"EVAL_CONFIG.device={d!r} requires CUDA/HIP runtime")
    device = torch.device(d)

    data = make_inputs(tokens=tokens, cfg=task_cfg, device=device, seed=seed)
    return [
        data["input_q"],
        data["w1_q"],
        data["w2_q"],
        data["topk_weights"],
        data["topk_ids"],
        data["input_scale"],
        data["fc1_scale"],
        data["fc2_scale"],
    ]


def run(
    input_q,
    w1_q,
    w2_q,
    topk_weights,
    topk_ids,
    input_scale,
    fc1_scale,
    fc2_scale,
):
    model = ModelNew()
    with torch.inference_mode():
        return model(
            input_q,
            w1_q,
            w2_q,
            topk_weights,
            topk_ids,
            input_scale,
            fc1_scale,
            fc2_scale,
        )
