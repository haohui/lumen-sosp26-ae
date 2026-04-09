import torch
import triton
import triton.language as tl

# Quant block sizes from the problem statement / reference implementation.
_Q_BLOCK_N = 128
_Q_BLOCK_K = 128


@triton.jit
def _fused_moe_kernel(
    # Inputs
    input_q_ptr,          # [tokens, model_dim]          float8
    w1_q_ptr,             # [experts, 2*inter_dim, dim]  float8
    w2_q_ptr,             # [experts, dim, inter_dim]    float8
    topk_w_ptr,           # [tokens, topk]               float32
    topk_ids_ptr,         # [tokens, topk]               int32/int64
    input_scale_ptr,      # [tokens, dim/128]            float32
    fc1_scale_ptr,        # [experts, (2*inter/128)*(dim/128)] float32
    fc2_scale_ptr,        # [experts, (dim/128)*(inter/128)]   float32
    # Output
    out_ptr,              # [tokens, model_dim]          bf16
    # Strides
    stride_in_t, stride_in_d,
    stride_w1_e, stride_w1_r, stride_w1_c,
    stride_w2_e, stride_w2_r, stride_w2_c,
    stride_tw_t, stride_tw_k,
    stride_tid_t, stride_tid_k,
    stride_is_t, stride_is_b,
    stride_fc1_e, stride_fc1_s,
    stride_fc2_e, stride_fc2_s,
    stride_out_t, stride_out_d,
    # Runtime sizes
    num_experts,
    # Compile-time sizes
    MODEL_DIM: tl.constexpr,
    INTER_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    MODEL_K_BLOCKS: tl.constexpr,   # MODEL_DIM // 128
    INTER_K_BLOCKS: tl.constexpr,   # INTER_DIM // 128
    BLOCK_OUT: tl.constexpr,        # output tile in model_dim
    BLOCK_J: tl.constexpr,          # hidden tile in inter_dim
    BLOCK_K: tl.constexpr,          # quant K block (128)
    BLOCK_N: tl.constexpr,          # quant N block (128)
):
    """
    Fused per-token MoE kernel.

    Fused stages:
      1) input dequant (block-wise)
      2) FC1 (gate/up) using block-wise dequantized W1
      3) SiLU(gate) * up
      4) FC2 using block-wise dequantized W2
      5) weighted top-k combine
      6) bf16 output store
    """
    token_id = tl.program_id(0)
    n_block = tl.program_id(1)

    offs_n = n_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    mask_n = offs_n < MODEL_DIM

    acc_out = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

    # Sum top-k routed experts for this token.
    for slot in range(TOPK):
        expert = tl.load(
            topk_ids_ptr + token_id * stride_tid_t + slot * stride_tid_k
        ).to(tl.int32)
        route_w = tl.load(
            topk_w_ptr + token_id * stride_tw_t + slot * stride_tw_k
        ).to(tl.float32)

        # Safety clamp for expert index; zero out invalid routes.
        valid = (expert >= 0) & (expert < num_experts)
        route_w = tl.where(valid, route_w, 0.0)
        expert = tl.where(expert < 0, 0, expert)
        expert = tl.where(expert >= num_experts, num_experts - 1, expert)

        w1_e_ptr = w1_q_ptr + expert * stride_w1_e
        w2_e_ptr = w2_q_ptr + expert * stride_w2_e
        fc1_e_ptr = fc1_scale_ptr + expert * stride_fc1_e
        fc2_e_ptr = fc2_scale_ptr + expert * stride_fc2_e

        # Stream over hidden dimension in BLOCK_J chunks.
        for j0 in range(0, INTER_DIM, BLOCK_J):
            offs_j = j0 + tl.arange(0, BLOCK_J)
            mask_j = offs_j < INTER_DIM

            gate = tl.zeros((BLOCK_J,), dtype=tl.float32)
            up = tl.zeros((BLOCK_J,), dtype=tl.float32)

            # FC1 K-reduction over model_dim in quantized 128-blocks.
            for k0 in range(0, MODEL_DIM, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < MODEL_DIM
                kb = k0 // BLOCK_K

                # Dequantized input block x[token, k0:k0+128]
                x_q = tl.load(
                    input_q_ptr + token_id * stride_in_t + offs_k * stride_in_d,
                    mask=mask_k,
                    other=0.0,
                ).to(tl.float32)
                x_scale = tl.load(
                    input_scale_ptr + token_id * stride_is_t + kb * stride_is_b
                ).to(tl.float32)
                x = x_q * x_scale

                # Gate rows: [j, k]
                w1g_ptrs = w1_e_ptr + offs_j[:, None] * stride_w1_r + offs_k[None, :] * stride_w1_c
                w1g_q = tl.load(
                    w1g_ptrs,
                    mask=mask_j[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                gate_rb = offs_j // BLOCK_N
                sidx_g = gate_rb * MODEL_K_BLOCKS + kb
                s_g = tl.load(
                    fc1_e_ptr + sidx_g * stride_fc1_s,
                    mask=mask_j,
                    other=0.0,
                ).to(tl.float32)
                gate += tl.sum((w1g_q * s_g[:, None]) * x[None, :], axis=1)

                # Up rows: [j + INTER_DIM, k]
                up_rows = offs_j + INTER_DIM
                w1u_ptrs = w1_e_ptr + up_rows[:, None] * stride_w1_r + offs_k[None, :] * stride_w1_c
                w1u_q = tl.load(
                    w1u_ptrs,
                    mask=mask_j[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                up_rb = up_rows // BLOCK_N
                sidx_u = up_rb * MODEL_K_BLOCKS + kb
                s_u = tl.load(
                    fc1_e_ptr + sidx_u * stride_fc1_s,
                    mask=mask_j,
                    other=0.0,
                ).to(tl.float32)
                up += tl.sum((w1u_q * s_u[:, None]) * x[None, :], axis=1)

            # SiLU(gate) * up
            sig = 1.0 / (1.0 + tl.exp(-gate))
            act = gate * sig * up

            # FC2 contribution for current hidden tile.
            w2_ptrs = w2_e_ptr + offs_n[:, None] * stride_w2_r + offs_j[None, :] * stride_w2_c
            w2_q = tl.load(
                w2_ptrs,
                mask=mask_n[:, None] & mask_j[None, :],
                other=0.0,
            ).to(tl.float32)

            rb2 = offs_n // BLOCK_N
            cb2 = offs_j // BLOCK_K
            sidx2 = rb2[:, None] * INTER_K_BLOCKS + cb2[None, :]
            s2 = tl.load(
                fc2_e_ptr + sidx2 * stride_fc2_s,
                mask=mask_n[:, None] & mask_j[None, :],
                other=0.0,
            ).to(tl.float32)

            acc_out += route_w * tl.sum((w2_q * s2) * act[None, :], axis=1)

    out_ptrs = out_ptr + token_id * stride_out_t + offs_n * stride_out_d
    tl.store(out_ptrs, acc_out.to(out_ptr.dtype.element_ty), mask=mask_n)


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
    """
    Triton MoE implementation.

    Fusion note:
      This launches one fused kernel that performs the whole operator pipeline
      (dequant input/weights + FC1 + SiLU*up + FC2 + top-k weighted combine),
      directly producing bf16 output. No PyTorch math is used in the wrapper.
    """
    # Basic validation (wrapper-only: checks + allocation + launch).
    assert input_q.ndim == 2, "input_q must be [tokens, dim]"
    assert w1_q.ndim == 3 and w2_q.ndim == 3, "w1_q/w2_q must be 3D"
    assert topk_weights.ndim == 2 and topk_ids.ndim == 2, "topk tensors must be [tokens, topk]"
    assert input_scale.ndim == 2 and fc1_scale.ndim == 2 and fc2_scale.ndim == 2, "scale tensors must be 2D"

    tokens, model_dim = input_q.shape
    experts = w1_q.shape[0]
    inter2 = w1_q.shape[1]
    assert inter2 % 2 == 0, "w1_q second dim must be 2 * inter_dim"
    inter_dim = inter2 // 2

    assert w1_q.shape[2] == model_dim, "w1_q dim mismatch"
    assert w2_q.shape[0] == experts and w2_q.shape[1] == model_dim and w2_q.shape[2] == inter_dim, "w2_q shape mismatch"

    assert topk_weights.shape[0] == tokens and topk_ids.shape[0] == tokens, "topk token dimension mismatch"
    topk = topk_ids.shape[1]
    assert topk_weights.shape[1] == topk, "topk size mismatch"

    assert model_dim % _Q_BLOCK_K == 0, "model_dim must be divisible by 128"
    assert inter_dim % _Q_BLOCK_K == 0, "inter_dim must be divisible by 128"
    assert (2 * inter_dim) % _Q_BLOCK_N == 0, "2*inter_dim must be divisible by 128"

    assert input_scale.shape == (tokens, model_dim // _Q_BLOCK_K), "input_scale shape mismatch"
    expected_fc1 = ((2 * inter_dim) // _Q_BLOCK_N) * (model_dim // _Q_BLOCK_K)
    expected_fc2 = (model_dim // _Q_BLOCK_N) * (inter_dim // _Q_BLOCK_K)
    assert fc1_scale.shape == (experts, expected_fc1), "fc1_scale shape mismatch"
    assert fc2_scale.shape == (experts, expected_fc2), "fc2_scale shape mismatch"

    # Allocate output (required by test: bf16).
    out = torch.empty((tokens, model_dim), device=input_q.device, dtype=torch.bfloat16)
    if tokens == 0:
        return out

    BLOCK_OUT = 64
    BLOCK_J = 32

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
        input_q.stride(0), input_q.stride(1),
        w1_q.stride(0), w1_q.stride(1), w1_q.stride(2),
        w2_q.stride(0), w2_q.stride(1), w2_q.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        input_scale.stride(0), input_scale.stride(1),
        fc1_scale.stride(0), fc1_scale.stride(1),
        fc2_scale.stride(0), fc2_scale.stride(1),
        out.stride(0), out.stride(1),
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
        num_warps=4,
        num_stages=2,
    )

    return out