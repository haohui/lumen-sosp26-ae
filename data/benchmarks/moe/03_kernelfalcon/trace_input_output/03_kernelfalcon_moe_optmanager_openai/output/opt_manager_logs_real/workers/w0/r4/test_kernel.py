#!/usr/bin/env python3
import sys
import traceback
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Summary:
# Test a fused MoE kernel against the provided PyTorch reference semantics:
# dequantize input/weights -> route build -> per-expert FC1 + SiLU*up + FC2 -> weighted combine -> BF16 output.

BLOCK_N = 128
BLOCK_K = 128
ROUTE_GROUP_SIZE = 32


@dataclass
class MoeConfig:
    dim: int = 256
    inter_dim: int = 128
    experts: int = 8
    topk: int = 4


class MoeSingleOpTorchRef:
    def __init__(self, out_dtype: torch.dtype = torch.bfloat16):
        self.block_n = BLOCK_N
        self.block_k = BLOCK_K
        self.out_dtype = out_dtype

    def _dequantize_input(self, input_q: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        tokens, model_dim = input_q.shape
        blocks = input_q.to(torch.float32).view(tokens, model_dim // self.block_k, self.block_k)
        return (blocks * input_scale.to(torch.float32).unsqueeze(-1)).reshape(tokens, model_dim)

    def _dequantize_weight_expert(
        self,
        weight_q_expert: torch.Tensor,
        scale_expert: torch.Tensor,
    ) -> torch.Tensor:
        rows, cols = weight_q_expert.shape
        row_blocks = rows // self.block_n
        col_blocks = cols // self.block_k
        blocks = (
            weight_q_expert.to(torch.float32)
            .view(row_blocks, self.block_n, col_blocks, self.block_k)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        scaled_blocks = blocks * scale_expert.to(torch.float32).view(row_blocks, col_blocks, 1, 1)
        return scaled_blocks.permute(0, 2, 1, 3).reshape(rows, cols).contiguous()

    def _build_sorted_routes(
        self,
        topk_ids_i64: torch.Tensor,
        topk_weights_f: torch.Tensor,
        num_experts: int,
    ):
        tokens, topk = topk_ids_i64.shape
        max_num_tokens_padded = tokens * topk + num_experts * ROUTE_GROUP_SIZE - topk
        max_num_m_blocks = (max_num_tokens_padded + ROUTE_GROUP_SIZE - 1) // ROUTE_GROUP_SIZE

        init_val = (topk << 24) | tokens
        sorted_token_ids = torch.full(
            (max_num_tokens_padded,),
            init_val,
            dtype=torch.int64,
            device=topk_ids_i64.device,
        )
        sorted_weights = torch.zeros((max_num_tokens_padded,), dtype=torch.float32, device=topk_weights_f.device)
        sorted_expert_ids = torch.full(
            (max_num_m_blocks,),
            -1,
            dtype=torch.int64,
            device=topk_ids_i64.device,
        )

        sorted_ids_begin = 0
        sorted_expert_ids_begin = 0
        for expert in range(num_experts):
            mask = topk_ids_i64.eq(expert)
            if not bool(mask.any()):
                continue

            token_ids, slot_ids = torch.nonzero(mask, as_tuple=True)
            tokens_num = int(token_ids.numel())

            route_ids = (slot_ids.to(torch.int64) << 24) | token_ids.to(torch.int64)
            sorted_token_ids[sorted_ids_begin: sorted_ids_begin + tokens_num] = route_ids
            sorted_weights[sorted_ids_begin: sorted_ids_begin + tokens_num] = topk_weights_f[token_ids, slot_ids]

            sorted_expert_ids_num = (tokens_num + ROUTE_GROUP_SIZE - 1) // ROUTE_GROUP_SIZE
            tokens_num_pad = sorted_expert_ids_num * ROUTE_GROUP_SIZE
            sorted_expert_ids[sorted_expert_ids_begin: sorted_expert_ids_begin + sorted_expert_ids_num] = expert

            sorted_ids_begin += tokens_num_pad
            sorted_expert_ids_begin += sorted_expert_ids_num

        num_valid_ids = torch.empty((2,), dtype=torch.int64, device=topk_ids_i64.device)
        num_valid_ids[0] = sorted_ids_begin
        num_valid_ids[1] = tokens
        return sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids

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
        tokens, model_dim = input_q.shape
        experts, _, inter_dim = w2_q.shape

        input_f = self._dequantize_input(input_q, input_scale)
        topk_ids_i64 = topk_ids.to(torch.int64).contiguous()
        topk_weights_f = topk_weights.to(torch.float32).contiguous()

        out = torch.zeros((tokens, model_dim), dtype=torch.float32, device=input_q.device)
        sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = self._build_sorted_routes(
            topk_ids_i64=topk_ids_i64,
            topk_weights_f=topk_weights_f,
            num_experts=experts,
        )
        num_valid_routes = int(num_valid_ids[0].item())
        if num_valid_routes == 0:
            return out.to(self.out_dtype)

        valid_route_idx = torch.arange(num_valid_routes, dtype=torch.int64, device=input_q.device)
        route_group_idx = torch.div(valid_route_idx, ROUTE_GROUP_SIZE, rounding_mode="floor")
        route_expert_ids = sorted_expert_ids.index_select(0, route_group_idx)
        route_token_ids = sorted_token_ids[:num_valid_routes] & 0x00FFFFFF
        route_weights = sorted_weights[:num_valid_routes]
        route_token_valid = route_token_ids.lt(tokens)

        for expert in range(experts):
            route_mask = route_expert_ids.eq(expert) & route_token_valid
            if not bool(route_mask.any()):
                continue

            token_ids = route_token_ids[route_mask]
            weights_e = route_weights[route_mask].unsqueeze(-1)
            x_e = input_f.index_select(0, token_ids)

            w1_e = self._dequantize_weight_expert(w1_q[expert], fc1_scale[expert])
            stage1 = x_e @ w1_e.transpose(0, 1)
            gate, up = stage1.split(inter_dim, dim=-1)
            activated = F.silu(gate) * up

            w2_e = self._dequantize_weight_expert(w2_q[expert], fc2_scale[expert])
            stage2 = activated @ w2_e.transpose(0, 1)
            out.index_add_(0, token_ids, stage2 * weights_e)

        return out.to(self.out_dtype)


class Model(nn.Module):
    def __init__(self, cfg: MoeConfig):
        super().__init__()
        self.cfg = cfg
        self.ref_impl = MoeSingleOpTorchRef(out_dtype=torch.bfloat16)

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
        return self.ref_impl.forward(
            input_q=input_q,
            w1_q=w1_q,
            w2_q=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            input_scale=input_scale,
            fc1_scale=fc1_scale,
            fc2_scale=fc2_scale,
        )


def get_init_inputs():
    return []


def _gpu_available() -> bool:
    if hasattr(torch, "gpu") and hasattr(torch.gpu, "is_available"):
        try:
            if torch.gpu.is_available():
                return True
        except Exception:
            pass
    return torch.cuda.is_available()


def _get_device() -> torch.device:
    if not _gpu_available():
        raise RuntimeError("HIP/ROCm not available")
    # On ROCm PyTorch typically uses device type 'cuda'
    return torch.device("cuda:0")


def make_inputs(tokens: int, cfg: MoeConfig, device: torch.device, seed: int):
    if not hasattr(torch, "float8_e4m3fnuz"):
        raise RuntimeError("torch.float8_e4m3fnuz is required by this problem but is not available.")

    if cfg.dim % BLOCK_K != 0 or cfg.dim % BLOCK_N != 0:
        raise ValueError(f"dim must be divisible by {BLOCK_N}/{BLOCK_K}, got {cfg.dim}")
    if cfg.inter_dim % BLOCK_K != 0 or (cfg.inter_dim * 2) % BLOCK_N != 0:
        raise ValueError(f"inter_dim must match blockshape, got {cfg.inter_dim}")
    if cfg.topk > cfg.experts:
        raise ValueError(f"topk ({cfg.topk}) must be <= experts ({cfg.experts})")

    g = torch.Generator(device=str(device))
    g.manual_seed(seed + tokens)
    f8 = torch.float8_e4m3fnuz

    input_q = (torch.randn((tokens, cfg.dim), dtype=torch.float32, device=device, generator=g) * 1.0 + 0.1).to(f8)
    w1_q = torch.randn(
        (cfg.experts, cfg.inter_dim * 2, cfg.dim),
        dtype=torch.float32,
        device=device,
        generator=g,
    ).mul_(8.0).to(f8)
    w2_q = torch.randn(
        (cfg.experts, cfg.dim, cfg.inter_dim),
        dtype=torch.float32,
        device=device,
        generator=g,
    ).mul_(8.0).to(f8)

    def _rand_pos(shape, mean, std):
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

    return [
        input_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        input_scale,
        fc1_scale,
        fc2_scale,
    ]


def _print_debug(expected: torch.Tensor, actual: torch.Tensor, inputs):
    print("---- DEBUG INFO ----")
    print(f"Expected shape/dtype/device: {expected.shape} / {expected.dtype} / {expected.device}")
    print(f"Actual   shape/dtype/device: {actual.shape} / {actual.dtype} / {actual.device}")

    e = expected.to(torch.float32)
    a = actual.to(torch.float32)
    diff = (a - e).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    denom = e.abs().clamp_min(1e-8)
    rel = (diff / denom).max().item()

    print(f"max_abs_diff={max_abs:.6e}, mean_abs_diff={mean_abs:.6e}, max_rel_diff={rel:.6e}")
    print(f"Expected sample: {e.flatten()[:10]}")
    print(f"Actual   sample: {a.flatten()[:10]}")
    print(f"Abs diff sample: {diff.flatten()[:10]}")

    names = [
        "input_q",
        "w1_q",
        "w2_q",
        "topk_weights",
        "topk_ids",
        "input_scale",
        "fc1_scale",
        "fc2_scale",
    ]
    for n, t in zip(names, inputs):
        sample = t.flatten()[:6]
        print(f"{n}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, sample={sample}")


def test_kernel():
    """Test the kernel implementation."""
    try:
        from kernel import kernel_function

        if not callable(kernel_function):
            print("kernel_function is not callable")
            return False

        device = _get_device()

        # Problem is config-driven; use one valid config that follows exact dtype/layout semantics.
        cfg = MoeConfig(dim=256, inter_dim=128, experts=8, topk=4)
        tokens = 96
        seed = 20260317

        _ = get_init_inputs()
        inputs = make_inputs(tokens=tokens, cfg=cfg, device=device, seed=seed)

        model = Model(cfg).to(device)
        model.eval()

        with torch.inference_mode():
            y_ref = model(*inputs)

        # Call kernel_function as a regular Python function (no Triton launch syntax).
        y = kernel_function(*inputs)

        if not isinstance(y, torch.Tensor):
            print(f"kernel_function must return a torch.Tensor, got {type(y)}")
            return False

        if y.device != inputs[0].device:
            print(f"Device mismatch: result.device={y.device}, input.device={inputs[0].device}")
            return False

        if y.shape != y_ref.shape:
            print(f"Shape mismatch: got {y.shape}, expected {y_ref.shape}")
            return False

        if y.dtype != y_ref.dtype:
            print(f"Dtype mismatch: got {y.dtype}, expected {y_ref.dtype}")
            return False

        if torch.isnan(y).any() or torch.isinf(y).any():
            print("Result contains NaN or Inf")
            _print_debug(y_ref, y, inputs)
            return False

        try:
            # Default tolerance first.
            close_default = torch.allclose(y.to(torch.float32), y_ref.to(torch.float32), rtol=1e-3, atol=1e-3)
            if not close_default:
                # Relaxed tolerance justified by BF16 output + float8 quantized inputs/weights + large reductions.
                close_relaxed = torch.allclose(y.to(torch.float32), y_ref.to(torch.float32), rtol=1e-2, atol=2e-2)
                if not close_relaxed:
                    print("NUMERICAL MISMATCH: failed default and relaxed tolerances")
                    _print_debug(y_ref, y, inputs)
                    return False
                print("Warning: passed only with relaxed tolerance (rtol=1e-2, atol=2e-2).")
        except Exception as cmp_err:
            print(f"Comparison error: {cmp_err}")
            traceback.print_exc()
            _print_debug(y_ref, y, inputs)
            return False

        print("Test passed.")
        return True

    except Exception as e:
        if isinstance(e, NameError):
            print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        else:
            print(f"Test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)