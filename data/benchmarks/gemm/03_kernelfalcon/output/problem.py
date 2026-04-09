import torch
import torch.nn as nn


_DTYPE_MAP = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _get_eval_config() -> dict:
    cfg = globals().get("EVAL_CONFIG", {})
    if not isinstance(cfg, dict):
        cfg = {}
    return cfg


def _get_dtype_from_config(cfg: dict) -> torch.dtype:
    dtype_name = str(cfg.get("dtype", "fp32")).lower()
    return _DTYPE_MAP.get(dtype_name, torch.float32)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        # Transposed-input formulation: A [M, K], B [N, K] => C [M, N] = A @ B^T
        return torch.matmul(a, b.t())


def get_init_inputs():
    return []


def get_inputs():
    cfg = _get_eval_config()
    m = int(cfg.get("M", 1024))
    n = int(cfg.get("N", 1024))
    k = int(cfg.get("K", 1024))
    dtype = _get_dtype_from_config(cfg)

    a = torch.randn(m, k, dtype=dtype)
    # Transposed-input formulation: generate B as [N, K]
    b = torch.randn(n, k, dtype=dtype)
    return [a, b]
