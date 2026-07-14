#!/usr/bin/env python3
"""Compile the Figure 1 FlashAttention kernel with invariant validation."""

import importlib.util
import os
from pathlib import Path

os.environ["AVELANG_VALIDATE_INVARIANTS"] = "1"
os.environ["HACK_SINGLE_WAVE_PER_EU"] = "1"
try:
    from avelang import knobs as avelang_knobs
    avelang_knobs.amdgpu.hack_single_wave_per_eu = True
except Exception:
    pass

import torch  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
KERNEL_PATH = REPO_ROOT / "datasets/inference/attention/lumen/attn_07_invariants.py"


def load_kernel():
    spec = importlib.util.spec_from_file_location("attn_07_invariants", KERNEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {KERNEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Invariant validation requires a ROCm-capable GPU")

    torch.manual_seed(0)
    seq_len, q_heads, kv_heads, head_dim = 64, 8, 1, 128
    q = torch.randn((seq_len, q_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((seq_len, kv_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((seq_len, kv_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    seq_ptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    load_kernel().flash_attn(q, k, v, seq_ptr, seq_len)
    print("FlashAttention invariant validation passed.")


if __name__ == "__main__":
    main()
