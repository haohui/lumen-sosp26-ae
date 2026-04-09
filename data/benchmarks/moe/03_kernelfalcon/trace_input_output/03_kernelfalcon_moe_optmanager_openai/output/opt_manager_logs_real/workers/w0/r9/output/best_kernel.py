import torch
import triton
import triton.language as tl

# Quantization/layout constants from the reference semantics
_Q_BLOCK_N = 128
_Q_BLOCK_K = 128

# Compute tiling (kept aligned with quant blocks for scale indexing)
_BLOCK_OUT = 128
_BLOCK_J = 128


@triton.jit
def _fused_moe_kernel(
    input_q_ptr,      # [T, D]      fp8
    w1_q_ptr,         # [E, 2I, D]  fp8
    w2_q_ptr,         # [E, D, I]   fp8
    topk_w_ptr,       # [T, K]      f32
    topk_ids_ptr,     # [T, K]      i32/i64
    input_scale_ptr,  # [T, D/128]  f32
    fc1_scale_ptr,    # [E, (2I/128)*(D/128)] f32
    fc2_scale_ptr,    # [E, (D/128)*(I/128)]  f32
    tmp_ptr,          # [T, D]      f32
    out_ptr,          # [T, D]      bf16
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
    stride_tmp_t,
    stride_tmp_d,
    stride_out_t,
    stride_out_d,
    MODEL_DIM: tl.constexpr,
    INTER_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    MODEL_K_BLOCKS: tl.constexpr,
    INTER_K_BLOCKS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOG2E: tl.constexpr,
):
    # Single fused kernel:
    # input dequant + FC1 + SiLU*up + FC2 + route-weighted accumulation + bf16 store
    tl.static_assert(BLOCK_OUT == BLOCK_N)
    tl.static_assert(BLOCK_J == BLOCK_K)

    token_id = tl.program_id(axis=0)

    offs_out = tl.arange(0, BLOCK_OUT)
    tmp_t_ptr = tmp_ptr + token_id * stride_tmp_t
    out_t_ptr = out_ptr + token_id * stride_out_t

    # Zero temporary accumulator for this token.
    for n0 in range(0, MODEL_DIM, BLOCK_OUT):
        offs_n = n0 + offs_out
        mask_n = offs_n < MODEL_DIM
        tl.store(tmp_t_ptr + offs_n * stride_tmp_d, 0.0, mask=mask_n)

    in_t_ptr = input_q_ptr + token_id * stride_in_t
    tw_t_ptr = topk_w_ptr + token_id * stride_tw_t
    tid_t_ptr = topk_ids_ptr + token_id * stride_tid_t
    is_t_ptr = input_scale_ptr + token_id * stride_is_t

    inter_row_blocks = INTER_DIM // BLOCK_N

    # Iterate token's routed experts directly; equivalent to reference weighted sum semantics.
    for slot in range(TOPK):
        expert = tl.load(tid_t_ptr + slot * stride_tid_k).to(tl.int32)
        route_w = tl.load(tw_t_ptr + slot * stride_tw_k).to(tl.float32)

        w1_e_ptr = w1_q_ptr + expert * stride_w1_e
        w2_e_ptr = w2_q_ptr + expert * stride_w2_e
        fc1_e_ptr = fc1_scale_ptr + expert * stride_fc1_e
        fc2_e_ptr = fc2_scale_ptr + expert * stride_fc2_e

        # FC1 produces [gate, up] chunks of size INTER_DIM each.
        for j0 in range(0, INTER_DIM, BLOCK_J):
            offs_j = j0 + tl.arange(0, BLOCK_J)

            gate = tl.zeros((BLOCK_J,), dtype=tl.float32)
            up = tl.zeros((BLOCK_J,), dtype=tl.float32)

            gate_rb = j0 // BLOCK_N
            up_rb = gate_rb + inter_row_blocks
            cb2 = j0 // BLOCK_K  # col-block in FC2 scales

            # Accumulate over model dim blocks
            for k0 in range(0, MODEL_DIM, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                kb = k0 // BLOCK_K

                # dequantized input block
                x_q = tl.load(in_t_ptr + offs_k * stride_in_d).to(tl.float32)
                x_s = tl.load(is_t_ptr + kb * stride_is_b).to(tl.float32)
                x = x_q * x_s

                # gate branch
                s_g = tl.load(fc1_e_ptr + (gate_rb * MODEL_K_BLOCKS + kb) * stride_fc1_s).to(tl.float32)
                w1g_ptrs = w1_e_ptr + offs_j[:, None] * stride_w1_r + offs_k[None, :] * stride_w1_c
                w1g_q = tl.load(w1g_ptrs).to(tl.float32)
                gate += s_g * tl.sum(w1g_q * x[None, :], axis=1)

                # up branch
                s_u = tl.load(fc1_e_ptr + (up_rb * MODEL_K_BLOCKS + kb) * stride_fc1_s).to(tl.float32)
                up_rows = offs_j + INTER_DIM
                w1u_ptrs = w1_e_ptr + up_rows[:, None] * stride_w1_r + offs_k[None, :] * stride_w1_c
                w1u_q = tl.load(w1u_ptrs).to(tl.float32)
                up += s_u * tl.sum(w1u_q * x[None, :], axis=1)

            # SiLU(gate) * up, then route weight (all fp32)
            sig = 1.0 / (1.0 + tl.exp2(-gate * LOG2E))
            act = gate * sig * up * route_w

            # FC2 and accumulate into token output
            rb = 0
            for n0 in range(0, MODEL_DIM, BLOCK_OUT):
                offs_n = n0 + offs_out
                mask_n = offs_n < MODEL_DIM

                s2 = tl.load(fc2_e_ptr + (rb * INTER_K_BLOCKS + cb2) * stride_fc2_s).to(tl.float32)

                w2_ptrs = w2_e_ptr + offs_n[:, None] * stride_w2_r + offs_j[None, :] * stride_w2_c
                w2_q = tl.load(w2_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

                contrib = s2 * tl.sum(w2_q * act[None, :], axis=1)

                tmp_ptrs = tmp_t_ptr + offs_n * stride_tmp_d
                cur = tl.load(tmp_ptrs, mask=mask_n, other=0.0)
                tl.store(tmp_ptrs, cur + contrib, mask=mask_n)

                rb += 1

    # Final cast to output dtype (bf16 expected by test)
    for n0 in range(0, MODEL_DIM, BLOCK_OUT):
        offs_n = n0 + offs_out
        mask_n = offs_n < MODEL_DIM
        v = tl.load(tmp_t_ptr + offs_n * stride_tmp_d, mask=mask_n, other=0.0)
        tl.store(out_t_ptr + offs_n * stride_out_d, v.to(out_ptr.dtype.element_ty), mask=mask_n)


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
    # Wrapper: validation + allocation + launch only (no math).
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
    assert inter_dim % _BLOCK_J == 0
    assert model_dim % _BLOCK_OUT == 0

    assert input_scale.shape == (tokens, model_dim // _Q_BLOCK_K)
    expected_fc1 = ((2 * inter_dim) // _Q_BLOCK_N) * (model_dim // _Q_BLOCK_K)
    expected_fc2 = (model_dim // _Q_BLOCK_N) * (inter_dim // _Q_BLOCK_K)
    assert fc1_scale.shape == (experts, expected_fc1)
    assert fc2_scale.shape == (experts, expected_fc2)

    out = torch.empty((tokens, model_dim), device=input_q.device, dtype=torch.bfloat16)
    if tokens == 0:
        return out

    tmp = torch.empty((tokens, model_dim), device=input_q.device, dtype=torch.float32)

    grid = (tokens,)

    _fused_moe_kernel[grid](
        input_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        input_scale,
        fc1_scale,
        fc2_scale,
        tmp,
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
        tmp.stride(0),
        tmp.stride(1),
        out.stride(0),
        out.stride(1),
        MODEL_DIM=model_dim,
        INTER_DIM=inter_dim,
        TOPK=topk,
        MODEL_K_BLOCKS=model_dim // _Q_BLOCK_K,
        INTER_K_BLOCKS=inter_dim // _Q_BLOCK_K,
        BLOCK_OUT=_BLOCK_OUT,
        BLOCK_J=_BLOCK_J,
        BLOCK_K=_Q_BLOCK_K,
        BLOCK_N=_Q_BLOCK_N,
        LOG2E=1.4426950408889634,
        num_warps=8,
        num_stages=2,
    )
    return out