#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

BLOCK_N = 128
BLOCK_K = 128

ANTI_HACK_MANIFEST = {
    "forbidden_api_used": [],
    "main_compute_kernels": ["fused_moe_1stage_kernel"],
    "single_runtime_op": True,
}

_HIP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdint>

constexpr int BLOCK_N_C = 128;
constexpr int BLOCK_K_C = 128;
constexpr int THREADS_C = 256;
constexpr int MAX_LOCAL_D = 32;

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void fused_moe_1stage_kernel(
    const c10::Float8_e4m3fnuz* __restrict__ input_q,   // [T, D]
    const c10::Float8_e4m3fnuz* __restrict__ w1_q,      // [E, 2I, D]
    const c10::Float8_e4m3fnuz* __restrict__ w2_q,      // [E, D, I]
    const float* __restrict__ topk_weights,             // [T, K]
    const int32_t* __restrict__ topk_ids,               // [T, K]
    const float* __restrict__ input_scale,              // [T, D/128]
    const float* __restrict__ fc1_scale,                // [E, (2I/128)*(D/128)]
    const float* __restrict__ fc2_scale,                // [E, (D/128)*(I/128)]
    c10::BFloat16* __restrict__ out,                    // [T, D]
    int tokens,
    int dim,
    int inter_dim,
    int experts,
    int topk
) {
    int t = static_cast<int>(blockIdx.x);
    if (t >= tokens) return;

    int tid = static_cast<int>(threadIdx.x);

    extern __shared__ float smem[];
    float* s_gate = smem;
    float* s_up = smem + blockDim.x;

    const int input_scale_cols = dim / BLOCK_K_C;
    const int fc1_row_blocks = (inter_dim * 2) / BLOCK_N_C;
    const int fc1_col_blocks = dim / BLOCK_K_C;
    const int fc1_stride = fc1_row_blocks * fc1_col_blocks;

    const int fc2_row_blocks = dim / BLOCK_N_C;
    const int fc2_col_blocks = inter_dim / BLOCK_K_C;
    const int fc2_stride = fc2_row_blocks * fc2_col_blocks;

    const int input_base = t * dim;
    const int input_scale_base = t * input_scale_cols;
    const int topk_base = t * topk;

    c10::BFloat16* out_row = out + static_cast<size_t>(t) * static_cast<size_t>(dim);

    for (int pass_base = tid; pass_base < dim; pass_base += blockDim.x * MAX_LOCAL_D) {
        int d_local[MAX_LOCAL_D];
        float acc_local[MAX_LOCAL_D];
        int n_local = 0;

        for (int d = pass_base; d < dim && n_local < MAX_LOCAL_D; d += blockDim.x) {
            d_local[n_local] = d;
            acc_local[n_local] = 0.0f;
            ++n_local;
        }

        for (int k = 0; k < topk; ++k) {
            int e = topk_ids[topk_base + k];
            if (e < 0 || e >= experts) continue;
            float route_w = topk_weights[topk_base + k];

            for (int i = 0; i < inter_dim; ++i) {
                float gate_partial = 0.0f;
                float up_partial = 0.0f;

                int gate_row = i;
                int up_row = i + inter_dim;

                int gate_rb = gate_row / BLOCK_N_C;
                int up_rb = up_row / BLOCK_N_C;

                for (int m = tid; m < dim; m += blockDim.x) {
                    int cb = m / BLOCK_K_C;

                    float x = static_cast<float>(input_q[input_base + m]) * input_scale[input_scale_base + cb];

                    float s_gate_w = fc1_scale[e * fc1_stride + gate_rb * fc1_col_blocks + cb];
                    float s_up_w = fc1_scale[e * fc1_stride + up_rb * fc1_col_blocks + cb];

                    size_t gate_idx = (static_cast<size_t>(e) * static_cast<size_t>(inter_dim * 2) +
                                       static_cast<size_t>(gate_row)) * static_cast<size_t>(dim) +
                                      static_cast<size_t>(m);
                    size_t up_idx = (static_cast<size_t>(e) * static_cast<size_t>(inter_dim * 2) +
                                     static_cast<size_t>(up_row)) * static_cast<size_t>(dim) +
                                    static_cast<size_t>(m);

                    float w_gate = static_cast<float>(w1_q[gate_idx]) * s_gate_w;
                    float w_up = static_cast<float>(w1_q[up_idx]) * s_up_w;

                    gate_partial += x * w_gate;
                    up_partial += x * w_up;
                }

                s_gate[tid] = gate_partial;
                s_up[tid] = up_partial;
                __syncthreads();

                for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        s_gate[tid] += s_gate[tid + stride];
                        s_up[tid] += s_up[tid + stride];
                    }
                    __syncthreads();
                }

                float activated = silu_f32(s_gate[0]) * s_up[0];
                __syncthreads();

                int ib = i / BLOCK_K_C;

                #pragma unroll 1
                for (int j = 0; j < n_local; ++j) {
                    int d = d_local[j];
                    int db = d / BLOCK_N_C;

                    float s2 = fc2_scale[e * fc2_stride + db * fc2_col_blocks + ib];

                    size_t w2_idx = (static_cast<size_t>(e) * static_cast<size_t>(dim) +
                                     static_cast<size_t>(d)) * static_cast<size_t>(inter_dim) +
                                    static_cast<size_t>(i);

                    float w2v = static_cast<float>(w2_q[w2_idx]) * s2;
                    acc_local[j] += route_w * activated * w2v;
                }

                __syncthreads();
            }
        }

        for (int j = 0; j < n_local; ++j) {
            out_row[d_local[j]] = c10::BFloat16(acc_local[j]);
        }
    }
}

torch::Tensor fused_moe_1stage_hip(
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
    TORCH_CHECK(w1_q.is_cuda() && w2_q.is_cuda(), "w1_q/w2_q must be CUDA/HIP tensors");
    TORCH_CHECK(topk_weights.is_cuda() && topk_ids.is_cuda(), "topk tensors must be CUDA/HIP tensors");
    TORCH_CHECK(input_scale.is_cuda() && fc1_scale.is_cuda() && fc2_scale.is_cuda(), "scale tensors must be CUDA/HIP tensors");

    TORCH_CHECK(input_q.scalar_type() == at::kFloat8_e4m3fnuz, "input_q must be float8_e4m3fnuz");
    TORCH_CHECK(w1_q.scalar_type() == at::kFloat8_e4m3fnuz, "w1_q must be float8_e4m3fnuz");
    TORCH_CHECK(w2_q.scalar_type() == at::kFloat8_e4m3fnuz, "w2_q must be float8_e4m3fnuz");
    TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "topk_weights must be float32");
    TORCH_CHECK(topk_ids.scalar_type() == at::kInt, "topk_ids must be int32");
    TORCH_CHECK(input_scale.scalar_type() == at::kFloat, "input_scale must be float32");
    TORCH_CHECK(fc1_scale.scalar_type() == at::kFloat, "fc1_scale must be float32");
    TORCH_CHECK(fc2_scale.scalar_type() == at::kFloat, "fc2_scale must be float32");

    TORCH_CHECK(input_q.dim() == 2, "input_q shape must be [T, D]");
    TORCH_CHECK(w1_q.dim() == 3, "w1_q shape must be [E, 2I, D]");
    TORCH_CHECK(w2_q.dim() == 3, "w2_q shape must be [E, D, I]");
    TORCH_CHECK(topk_weights.dim() == 2 && topk_ids.dim() == 2, "topk tensors shape must be [T, K]");
    TORCH_CHECK(input_scale.dim() == 2, "input_scale shape must be [T, D/128]");
    TORCH_CHECK(fc1_scale.dim() == 2, "fc1_scale shape must be [E, (2I/128)*(D/128)]");
    TORCH_CHECK(fc2_scale.dim() == 2, "fc2_scale shape must be [E, (D/128)*(I/128)]");

    TORCH_CHECK(input_q.is_contiguous(), "input_q must be contiguous");
    TORCH_CHECK(w1_q.is_contiguous(), "w1_q must be contiguous");
    TORCH_CHECK(w2_q.is_contiguous(), "w2_q must be contiguous");
    TORCH_CHECK(topk_weights.is_contiguous(), "topk_weights must be contiguous");
    TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
    TORCH_CHECK(input_scale.is_contiguous(), "input_scale must be contiguous");
    TORCH_CHECK(fc1_scale.is_contiguous(), "fc1_scale must be contiguous");
    TORCH_CHECK(fc2_scale.is_contiguous(), "fc2_scale must be contiguous");

    const int64_t tokens64 = input_q.size(0);
    const int64_t dim64 = input_q.size(1);

    const int64_t experts64 = w1_q.size(0);
    const int64_t two_inter64 = w1_q.size(1);
    const int64_t w1_dim64 = w1_q.size(2);

    TORCH_CHECK(w1_dim64 == dim64, "w1_q dim mismatch with input_q");
    TORCH_CHECK(two_inter64 % 2 == 0, "w1_q second dim must be 2*inter_dim");
    const int64_t inter64 = two_inter64 / 2;

    TORCH_CHECK(w2_q.size(0) == experts64, "w2_q expert dim mismatch");
    TORCH_CHECK(w2_q.size(1) == dim64, "w2_q dim mismatch");
    TORCH_CHECK(w2_q.size(2) == inter64, "w2_q inter_dim mismatch");

    TORCH_CHECK(topk_weights.size(0) == tokens64 && topk_ids.size(0) == tokens64, "topk token dim mismatch");
    TORCH_CHECK(topk_weights.size(1) == topk_ids.size(1), "topk K mismatch");

    const int64_t topk64 = topk_ids.size(1);

    TORCH_CHECK(dim64 % BLOCK_K_C == 0 && dim64 % BLOCK_N_C == 0, "dim must be divisible by 128");
    TORCH_CHECK(inter64 % BLOCK_K_C == 0, "inter_dim must be divisible by 128");
    TORCH_CHECK((inter64 * 2) % BLOCK_N_C == 0, "2*inter_dim must be divisible by 128");
    TORCH_CHECK(topk64 <= experts64, "topk must be <= experts");

    TORCH_CHECK(input_scale.size(0) == tokens64, "input_scale token dim mismatch");
    TORCH_CHECK(input_scale.size(1) == dim64 / BLOCK_K_C, "input_scale cols mismatch");

    TORCH_CHECK(fc1_scale.size(0) == experts64, "fc1_scale expert dim mismatch");
    TORCH_CHECK(fc1_scale.size(1) == ((inter64 * 2) / BLOCK_N_C) * (dim64 / BLOCK_K_C), "fc1_scale shape mismatch");

    TORCH_CHECK(fc2_scale.size(0) == experts64, "fc2_scale expert dim mismatch");
    TORCH_CHECK(fc2_scale.size(1) == (dim64 / BLOCK_N_C) * (inter64 / BLOCK_K_C), "fc2_scale shape mismatch");

    int tokens = static_cast<int>(tokens64);
    int dim = static_cast<int>(dim64);
    int inter_dim = static_cast<int>(inter64);
    int experts = static_cast<int>(experts64);
    int topk = static_cast<int>(topk64);

    auto out = torch::empty({tokens64, dim64}, input_q.options().dtype(torch::kBFloat16));

    const int threads = THREADS_C;
    dim3 block(threads);
    dim3 grid(static_cast<unsigned int>(tokens));
    size_t shmem = static_cast<size_t>(threads) * 2 * sizeof(float);

    auto stream = at::cuda::getDefaultCUDAStream();

    hipLaunchKernelGGL(
        fused_moe_1stage_kernel,
        grid,
        block,
        shmem,
        stream.stream(),
        input_q.data_ptr<c10::Float8_e4m3fnuz>(),
        w1_q.data_ptr<c10::Float8_e4m3fnuz>(),
        w2_q.data_ptr<c10::Float8_e4m3fnuz>(),
        topk_weights.data_ptr<float>(),
        topk_ids.data_ptr<int32_t>(),
        input_scale.data_ptr<float>(),
        fc1_scale.data_ptr<float>(),
        fc2_scale.data_ptr<float>(),
        out.data_ptr<c10::BFloat16>(),
        tokens,
        dim,
        inter_dim,
        experts,
        topk
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_moe_1stage_hip", &fused_moe_1stage_hip, "Single-kernel fused MoE (HIP)");
}
"""

_fused_moe_mod = None


def _get_fused_moe_mod():
    global _fused_moe_mod
    if _fused_moe_mod is not None:
        return _fused_moe_mod
    name = "fused_moe_1stage_" + hashlib.sha1(_HIP_SRC.encode("utf-8")).hexdigest()[:16]
    _fused_moe_mod = load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=_HIP_SRC,
        functions=None,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        with_cuda=True,
        verbose=False,
    )
    return _fused_moe_mod


@dataclass
class MoeConfig:
    dim: int = 7168
    inter_dim: int = 2048
    experts: int = 32
    topk: int = 4


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


def _eval_cfg() -> dict[str, Any]:
    cfg = globals().get("EVAL_CONFIG", {})
    if not isinstance(cfg, dict):
        return {}
    return cfg


def _cfg_int(cfg: dict[str, Any], key: str, default: int, minimum: int = 1) -> int:
    raw = cfg.get(key, default)
    try:
        val = int(raw)
    except Exception:
        val = int(default)
    return max(minimum, val)


def _cfg_device(cfg: dict[str, Any]) -> torch.device:
    d = str(cfg.get("device", "cuda:0")).strip()
    if not d:
        d = "cuda:0"
    if d.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"EVAL_CONFIG.device={d!r} requires CUDA/HIP runtime")
    return torch.device(d)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = _eval_cfg()
        self.cfg = MoeConfig(
            dim=_cfg_int(cfg, "dim", 7168, minimum=128),
            inter_dim=_cfg_int(cfg, "inter_dim", 2048, minimum=128),
            experts=_cfg_int(cfg, "experts", 32, minimum=1),
            topk=_cfg_int(cfg, "topk", 4, minimum=1),
        )
        if self.cfg.topk > self.cfg.experts:
            self.cfg.topk = self.cfg.experts
        self._ext = _get_fused_moe_mod()

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
        return self._ext.fused_moe_1stage_hip(
            input_q,
            w1_q,
            w2_q,
            topk_weights,
            topk_ids,
            input_scale,
            fc1_scale,
            fc2_scale,
        )


class Model(ModelNew):
    pass


def get_init_inputs():
    return []


def get_inputs():
    cfg = _eval_cfg()
    task_cfg = MoeConfig(
        dim=_cfg_int(cfg, "dim", 7168, minimum=128),
        inter_dim=_cfg_int(cfg, "inter_dim", 2048, minimum=128),
        experts=_cfg_int(cfg, "experts", 32, minimum=1),
        topk=_cfg_int(cfg, "topk", 4, minimum=1),
    )
    if task_cfg.topk > task_cfg.experts:
        task_cfg.topk = task_cfg.experts

    tokens = _cfg_int(cfg, "tokens", 256, minimum=1)
    seed = _cfg_int(cfg, "seed", 20260317, minimum=0)
    device = _cfg_device(cfg)

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