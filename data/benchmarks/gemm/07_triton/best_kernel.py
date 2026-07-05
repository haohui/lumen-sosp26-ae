#!/usr/bin/env python3
"""
Source origin (HipKittens):
  repo path: analysis/baselines/gemm/triton_gemm_v01.py
  commit: 4d15d8e92dfc65b6b33c36ad8b6a7e883c5f7245

This module keeps the kernel logic and autotune configs, and exposes a benchmark-friendly
`matmul_bf16(a, b, out=...)` entry without running the original script's unit test/plot code.

Compared with the baseline extraction, this variant applies AITER-style `remap_xcd` to
the unified pid before grouped tile mapping on AMD backends, and adds optional stagger-K
start offset per output tile.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

_LAST_LAUNCH_META: dict | None = None
_ENTRY_CACHE: dict[tuple[int, int, int, str, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _target_is_cuda() -> bool:
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def _get_cuda_autotune_config():
    base = [
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    ]
    out = []
    for cfg in base:
        kwargs = dict(cfg.kwargs)
        kwargs["REMAP_XCD"] = 0
        kwargs["NUM_XCDS"] = 8
        kwargs["STAGGER_K"] = 1
        out.append(triton.Config(kwargs, num_stages=cfg.num_stages, num_warps=cfg.num_warps))
    return out


def _get_hip_autotune_config():
    base = [
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 16, "GROUP_SIZE_M": 1, "waves_per_eu": 2},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 16, "GROUP_SIZE_M": 4, "waves_per_eu": 2},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1, "waves_per_eu": 2},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "waves_per_eu": 3},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1, "waves_per_eu": 8},
            num_warps=4,
            num_stages=2,
        ),
    ]
    out = []
    for cfg in base:
        for remap_xcd in (0, 1):
            for stagger_k in (1, 2, 4, 8):
                kwargs = dict(cfg.kwargs)
                kwargs["REMAP_XCD"] = remap_xcd
                kwargs["NUM_XCDS"] = 8
                kwargs["STAGGER_K"] = stagger_k
                out.append(triton.Config(kwargs, num_warps=cfg.num_warps, num_stages=cfg.num_stages))
    return out


def _get_autotune_config():
    if _target_is_cuda():
        return _get_cuda_autotune_config()
    return _get_hip_autotune_config()


@triton.jit
def remap_xcd(pid, grid_mn, NUM_XCDS: tl.constexpr = 8):
    pids_per_xcd = (grid_mn + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = grid_mn % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    return pid


@triton.autotune(
    configs=_get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    REMAP_XCD: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    STAGGER_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    grid_mn = num_pid_m * num_pid_n
    if REMAP_XCD:
        pid = remap_xcd(pid, grid_mn, NUM_XCDS=NUM_XCDS)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    if STAGGER_K <= 1:
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        for k in range(0, num_k_tiles):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    else:
        # Start K-loop from a per-(pid_m,pid_n) offset, then wrap around.
        # This de-phases CTA K-start points to reduce synchronized pressure.
        tile_id_mn = pid_m * num_pid_n + pid_n
        k_start_tile = tile_id_mn % STAGGER_K
        for k in range(0, num_k_tiles):
            k_tile = (k + k_start_tile) % num_k_tiles
            k_base = k_tile * BLOCK_SIZE_K
            offs_k_curr = k_base + offs_k
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_curr[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k_curr[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            a = tl.load(a_ptrs, mask=offs_k_curr[None, :] < K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_curr[:, None] < K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.bfloat16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def _capture_best_config_meta() -> dict | None:
    cfg = getattr(matmul_kernel, "best_config", None)
    if cfg is None:
        return None
    kwargs = dict(getattr(cfg, "kwargs", {}) or {})
    meta = {
        "num_warps": int(getattr(cfg, "num_warps", 0) or 0),
        "num_stages": int(getattr(cfg, "num_stages", 0) or 0),
        "num_ctas": int(getattr(cfg, "num_ctas", 0) or 0),
        "maxnreg": int(getattr(cfg, "maxnreg", 0) or 0),
        "kwargs": kwargs,
    }
    if "REMAP_XCD" in kwargs:
        try:
            meta["remap_xcd"] = bool(int(kwargs["REMAP_XCD"]))
        except Exception:
            meta["remap_xcd"] = kwargs["REMAP_XCD"]
    if "NUM_XCDS" in kwargs:
        try:
            meta["num_xcds"] = int(kwargs["NUM_XCDS"])
        except Exception:
            meta["num_xcds"] = kwargs["NUM_XCDS"]
    if "STAGGER_K" in kwargs:
        try:
            meta["stagger_k"] = int(kwargs["STAGGER_K"])
        except Exception:
            meta["stagger_k"] = kwargs["STAGGER_K"]
    if "waves_per_eu" in kwargs:
        try:
            meta["waves_per_eu"] = int(kwargs["waves_per_eu"])
        except Exception:
            pass
    return meta


def get_last_launch_meta() -> dict | None:
    return _LAST_LAUNCH_META


def matmul_bf16(a, b, out=None, activation=""):
    global _LAST_LAUNCH_META
    # A: [M, K], B: [N, K] -> C: [M, N] (compute A @ B^T)
    assert a.shape[1] == b.shape[1], "Incompatible dimensions for A @ B^T"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise RuntimeError("matmul_bf16 expects BF16 inputs")
    M, K = a.shape
    N, _K = b.shape
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    matmul_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        out.stride(0),
        out.stride(1),
        ACTIVATION=activation,
    )
    _LAST_LAUNCH_META = _capture_best_config_meta()
    return out


def kernel_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("kernel_function expects 2D tensors")
    m, k = a.shape
    n, kb = b.shape
    if kb != k:
        raise ValueError(f"incompatible shapes: a={tuple(a.shape)}, b={tuple(b.shape)}")
    b_nk = b.contiguous()
    key = (a.data_ptr(), b.data_ptr(), m, n, str(a.dtype), str(a.device))
    cached = _ENTRY_CACHE.get(key)
    if cached is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
        _ENTRY_CACHE[key] = (out,)
    else:
        (out,) = cached
    return matmul_bf16(a, b_nk, out=out)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return kernel_function(a, b)
