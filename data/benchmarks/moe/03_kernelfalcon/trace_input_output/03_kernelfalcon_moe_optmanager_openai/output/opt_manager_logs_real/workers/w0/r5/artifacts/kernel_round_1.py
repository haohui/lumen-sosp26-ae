import torch
import triton
import triton.language as tl

# Quantization block shapes from problem statement / reference
_Q_BLOCK_N = 128
_Q_BLOCK_K = 128


@triton.jit
def _fused_moe_kernel(
    input_q_ptr,      # [T, D] fp8
    w1_q_ptr,         # [E, 2I, D] fp8
    w2_q_ptr,         # [E, D, I] fp8
    topk_w_ptr,       # [T, K] f32
    topk_ids_ptr,     # [T, K] i32/i64
    input_scale_ptr,  # [T, D/128] f32
    fc1_scale_ptr,    # [E, ((2I)/128)*(D/128)] f32
    fc2_scale_ptr,    # [E, (D/128)*(I/128)] f32
    out_ptr,          # [T, D] bf16
    stride_in_t,
    stride_in_d,
    stride_w1_e,
    stride_w1_r,
    stride_w1_c,
    stride_w2_e,
    stride_w2_r,
    stride_w2_c,
    stride_tw_t,
    stride_tw_k,
    stride_tid_t,
    stride_tid_k,
    stride_is_t,
    stride_is_b,
    stride_fc1_e,
    stride_fc1_s,
    stride_fc2_e,
    stride_fc2_s,
    stride_out_t,
    stride_out_d,
    num_experts,
    MODEL_DIM: tl.constexpr,
    INTER_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    MODEL_K_BLOCKS: tl.constexpr,
    INTER_K_BLOCKS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,  # output tile over model dim
    BLOCK_J: tl.constexpr,    # hidden tile over inter dim
    BLOCK_K: tl.constexpr,    # quant K block (=128)
    BLOCK_N: tl.constexpr,    # quant N block (=128)
):
    # Program ids: one program per (token, output-block)
    token_id = tl.program_id(axis=0)
    out_blk = tl.program_id(axis=1)

    out_start = out_blk * BLOCK_OUT
    offs_out = out_start + tl.arange(0, BLOCK_OUT)
    mask_out = offs_out < MODEL_DIM

    # fc2 row block index (for block-wise dequant scale)
    fc2_row_blk = out_start // BLOCK_N
    has_any_out = out_start < MODEL_DIM

    # Base pointers for this token
    in_t_ptr = input_q_ptr + token_id * stride_in_t
    tw_t_ptr = topk_w_ptr + token_id * stride_tw_t
    tid_t_ptr = topk_ids_ptr + token_id * stride_tid_t
    is_t_ptr = input_scale_ptr + token_id * stride_is_t
    out_t_ptr = out_ptr + token_id * stride_out_t

    # Accumulate output in fp32, cast on store
    acc_out = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

    # Fused pipeline per route:
    # dequant(input) + dequant(w1 blocks) + FC1 + SiLU*up + dequant(w2 blocks) + FC2 + route weighting
    for slot in tl.static_range(0, TOPK):
        expert = tl.load(tid_t_ptr + slot * stride_tid_k).to(tl.int32)
        route_w = tl.load(tw_t_ptr + slot * stride_tw_k).to(tl.float32)

        valid = (expert >= 0) & (expert < num_experts)
        route_w = tl.where(valid, route_w, 0.0)

        # Clamp expert index to keep pointer math valid even for invalid ids
        expert = tl.where(expert < 0, 0, expert)
        expert = tl.where(expert >= num_experts, num_experts - 1, expert)

        w1_e_ptr = w1_q_ptr + expert * stride_w1_e
        w2_e_ptr = w2_q_ptr + expert * stride_w2_e
        fc1_e_ptr = fc1_scale_ptr + expert * stride_fc1_e
        fc2_e_ptr = fc2_scale_ptr + expert * stride_fc2_e

        for j0 in tl.static_range(0, INTER_DIM, BLOCK_J):
            offs_j = j0 + tl.arange(0, BLOCK_J)
            mask_j = offs_j < INTER_DIM

            gate = tl.zeros((BLOCK_J,), dtype=tl.float32)
            up = tl.zeros((BLOCK_J,), dtype=tl.float32)

            gate_row_blk = j0 // BLOCK_N
            up_row_blk = (j0 + INTER_DIM) // BLOCK_N
            fc2_col_blk = j0 // BLOCK_K

            # FC1 accumulation over model dim with dequantized input + dequantized weight blocks
            for k0 in tl.static_range(0, MODEL_DIM, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < MODEL_DIM
                kb = k0 // BLOCK_K

                x_q = tl.load(in_t_ptr + offs_k * stride_in_d, mask=mask_k, other=0.0).to(tl.float32)
                x_s = tl.load(is_t_ptr + kb * stride_is_b).to(tl.float32)
                x = x_q * x_s

                s_g = tl.load(fc1_e_ptr + (gate_row_blk * MODEL_K_BLOCKS + kb) * stride_fc1_s).to(tl.float32)
                w1g_ptrs = w1_e_ptr + offs_j[:, None] * stride_w1_r + offs_k[None, :] * stride_w1_c
                w1g_q = tl.load(w1g_ptrs, mask=mask_j[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
                gate += s_g * tl.sum(w1g_q * x[None, :], axis=1)

                s_u = tl.load(fc1_e_ptr + (up_row_blk * MODEL_K_BLOCKS + kb) * stride_fc1_s).to(tl.float32)
                up_rows = offs_j + INTER_DIM
                w1u_ptrs = w1_e_ptr + up_rows[:, None] * stride_w1_r + offs_k[None, :] * stride_w1_c
                w1u_q = tl.load(w1u_ptrs, mask=mask_j[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
                up += s_u * tl.sum(w1u_q * x[None, :], axis=1)

            # SiLU(gate) * up
            sig = 1.0 / (1.0 + tl.exp(-gate))
            act = gate * sig * up

            # FC2 block scale for this [out_block, inter_block]
            s2 = tl.load(
                fc2_e_ptr + (fc2_row_blk * INTER_K_BLOCKS + fc2_col_blk) * stride_fc2_s,
                mask=has_any_out,
                other=0.0,
            ).to(tl.float32)

            w2_ptrs = w2_e_ptr + offs_out[:, None] * stride_w2_r + offs_j[None, :] * stride_w2_c
            w2_q = tl.load(w2_ptrs, mask=mask_out[:, None] & mask_j[None, :], other=0.0).to(tl.float32)

            acc_out += route_w * s2 * tl.sum(w2_q * act[None, :], axis=1)

    tl.store(out_t_ptr + offs_out * stride_out_d, acc_out.to(out_ptr.dtype.element_ty), mask=mask_out)


def kernel_function(
    input_q: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    input_scale: torch.Tensor,
    fc1_scale: torch.Tensor,
    fc2_scale: torch.Tensor,
) -> torch.Tensor:
    # Validation only (no compute in wrapper)
    assert input_q.ndim == 2
    assert w1_q.ndim == 3 and w2_q.ndim == 3
    assert topk_weights.ndim == 2 and topk_ids.ndim == 2
    assert input_scale.ndim == 2 and fc1_scale.ndim == 2 and fc2_scale.ndim == 2
    assert topk_ids.dtype in (torch.int32, torch.int64)

    tokens, model_dim = input_q.shape
    experts = w1_q.shape[0]
    inter2 = w1_q.shape[1]
    assert inter2 % 2 == 0
    inter_dim = inter2 // 2

    assert w1_q.shape[2] == model_dim
    assert w2_q.shape == (experts, model_dim, inter_dim)

    assert topk_weights.shape[0] == tokens and topk_ids.shape[0] == tokens
    topk = topk_ids.shape[1]
    assert topk_weights.shape[1] == topk

    assert model_dim % _Q_BLOCK_K == 0
    assert inter_dim % _Q_BLOCK_K == 0
    assert (2 * inter_dim) % _Q_BLOCK_N == 0

    assert input_scale.shape == (tokens, model_dim // _Q_BLOCK_K)
    expected_fc1 = ((2 * inter_dim) // _Q_BLOCK_N) * (model_dim // _Q_BLOCK_K)
    expected_fc2 = (model_dim // _Q_BLOCK_N) * (inter_dim // _Q_BLOCK_K)
    assert fc1_scale.shape == (experts, expected_fc1)
    assert fc2_scale.shape == (experts, expected_fc2)

    out = torch.empty((tokens, model_dim), device=input_q.device, dtype=torch.bfloat16)
    if tokens == 0:
        return out

    # Keep BLOCK_OUT <= BLOCK_N so each output program maps to one fc2 row-scale block.
    BLOCK_OUT = 128
    BLOCK_J = 128

    grid = (tokens, triton.cdiv(model_dim, BLOCK_OUT))

    _fused_moe_kernel[grid](
        input_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        input_scale,
        fc1_scale,
        fc2_scale,
        out,
        input_q.stride(0),
        input_q.stride(1),
        w1_q.stride(0),
        w1_q.stride(1),
        w1_q.stride(2),
        w2_q.stride(0),
        w2_q.stride(1),
        w2_q.stride(2),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        input_scale.stride(0),
        input_scale.stride(1),
        fc1_scale.stride(0),
        fc1_scale.stride(1),
        fc2_scale.stride(0),
        fc2_scale.stride(1),
        out.stride(0),
        out.stride(1),
        experts,
        MODEL_DIM=model_dim,
        INTER_DIM=inter_dim,
        TOPK=topk,
        MODEL_K_BLOCKS=model_dim // _Q_BLOCK_K,
        INTER_K_BLOCKS=inter_dim // _Q_BLOCK_K,
        BLOCK_OUT=BLOCK_OUT,
        BLOCK_J=BLOCK_J,
        BLOCK_K=_Q_BLOCK_K,
        BLOCK_N=_Q_BLOCK_N,
        num_warps=8,
        num_stages=2,
    )
    return out