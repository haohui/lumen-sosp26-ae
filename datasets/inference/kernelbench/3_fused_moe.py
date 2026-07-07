from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK_N = 128
BLOCK_K = 128
DIM = 7168
INTER_DIM = 2048
EXPERTS = 32
TOPK = 4
TOKENS = 256
SEED = 20260317


def make_inputs(
    *,
    tokens: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    if DIM % BLOCK_K != 0 or DIM % BLOCK_N != 0:
        raise ValueError(f"dim must be divisible by {BLOCK_N}/{BLOCK_K}, got {DIM}")
    if INTER_DIM % BLOCK_K != 0 or (INTER_DIM * 2) % BLOCK_N != 0:
        raise ValueError(f"inter_dim must match blockshape, got {INTER_DIM}")
    if TOPK > EXPERTS:
        raise ValueError(f"topk ({TOPK}) must be <= experts ({EXPERTS})")

    g = torch.Generator()
    g.manual_seed(seed + tokens)

    input_q = (
        torch.randn((tokens, DIM), dtype=torch.float32, generator=g) * 1.0 + 0.1
    ).to(torch.float8_e4m3fnuz)
    w1_q = torch.randn(
        (EXPERTS, INTER_DIM * 2, DIM),
        dtype=torch.float32,
        generator=g,
    ).mul_(8.0).to(torch.float8_e4m3fnuz)
    w2_q = torch.randn(
        (EXPERTS, DIM, INTER_DIM),
        dtype=torch.float32,
        generator=g,
    ).mul_(8.0).to(torch.float8_e4m3fnuz)

    def _rand_pos(shape: tuple[int, ...], mean: float, std: float) -> torch.Tensor:
        x = torch.randn(shape, dtype=torch.float32, generator=g) * std + mean
        return x.clamp_min(1e-8)

    input_scale = _rand_pos((tokens, DIM // BLOCK_K), mean=1e-1, std=2e-2)
    fc1_scale = _rand_pos(
        (EXPERTS, ((INTER_DIM * 2) // BLOCK_N) * (DIM // BLOCK_K)),
        mean=1e-2,
        std=2e-3,
    )
    fc2_scale = _rand_pos(
        (EXPERTS, (DIM // BLOCK_N) * (INTER_DIM // BLOCK_K)),
        mean=1e-2,
        std=2e-3,
    )

    # Match MoE routing semantics: one token selects top-k unique experts.
    scores = torch.randn((tokens, EXPERTS), dtype=torch.float32, generator=g)
    topk_val, topk_idx = torch.topk(scores, k=TOPK, dim=-1, largest=True, sorted=True)
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


class Model(nn.Module):
    """AE generation reference wrapper."""

    def __init__(self):
        super().__init__()

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
        def _dequantize_weight(weight_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            experts, rows, cols = weight_q.shape
            row_blocks = rows // BLOCK_N
            col_blocks = cols // BLOCK_K

            dense_q = weight_q.to(torch.float32)
            blocks = (
                dense_q.view(experts, row_blocks, BLOCK_N, col_blocks, BLOCK_K)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )
            scaled_blocks = blocks * scale.to(torch.float32).view(
                experts, row_blocks, col_blocks, 1, 1
            )
            return (
                scaled_blocks.permute(0, 1, 3, 2, 4)
                .reshape(experts, rows, cols)
                .contiguous()
            )

        tokens, model_dim = input_q.shape
        _, _, inter_dim = w2_q.shape

        input_blocks = input_q.to(torch.float32).view(tokens, model_dim // BLOCK_K, BLOCK_K)
        input_f = (input_blocks * input_scale.to(torch.float32).unsqueeze(-1)).reshape(
            tokens, model_dim
        )
        topk_weights_f = topk_weights.to(torch.float32)

        # Capture-friendly reference: avoid per-expert dynamic torch.where branches.
        w1_f = _dequantize_weight(w1_q, fc1_scale)  # [E, 2I, D]
        w2_f = _dequantize_weight(w2_q, fc2_scale)  # [E, D, I]

        stage1_all = torch.einsum("td,eid->tei", input_f, w1_f)  # [T, E, 2I]
        gate, up = stage1_all.split([inter_dim, inter_dim], dim=-1)
        activated = F.silu(gate) * up  # [T, E, I]
        route_all = torch.einsum("tei,edi->ted", activated, w2_f)  # [T, E, D]

        gather_index = topk_ids.to(torch.int64).unsqueeze(-1).expand(tokens, TOPK, model_dim)
        selected = torch.gather(route_all, dim=1, index=gather_index)  # [T, K, D]
        out = (selected * topk_weights_f.unsqueeze(-1)).sum(dim=1)  # [T, D]
        return out.to(torch.bfloat16)


def get_init_inputs():
    return []


def get_inputs():
    data = make_inputs(tokens=TOKENS, seed=SEED)
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
