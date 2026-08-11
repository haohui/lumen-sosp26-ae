import math

import substrate
import substrate.language as S
import torch
import torch.nn as nn

SEQ_LEN: S.constexpr = 512
HEAD_DIM: S.constexpr = 1024
BLOCK_ROWS: S.constexpr = 8

NEG_INF = -1.0e30
SCALE_LOG2 = math.log2(math.e) / math.sqrt(HEAD_DIM)


@substrate.jit
def naive_attention_kernel(
    out_ptr: S.Pointer(S.bf16),
    q_ptr: S.Pointer(S.bf16),
    k_ptr: S.Pointer(S.bf16),
    v_ptr: S.Pointer(S.bf16),
    bh: S.i32,
):
    zero = S.convert(0.0, S.f32)
    one = S.convert(1.0, S.f32)
    neg_inf = S.convert(NEG_INF, S.f32)
    scale_log2 = S.convert(SCALE_LOG2, S.f32)

    bh_idx = S.convert(S.block_id(1), S.i32)
    block_row = S.convert(S.block_id(0), S.i32)
    thread_row = S.convert(S.thread_id(0), S.i32)
    row_in_seq = block_row * S.convert(BLOCK_ROWS, S.i32) + thread_row

    if bh_idx >= bh or row_in_seq >= S.convert(SEQ_LEN, S.i32):
        return

    q_memref = S.make_tensor(q_ptr, S.bf16, S.make_layout((bh, SEQ_LEN, HEAD_DIM), (SEQ_LEN * HEAD_DIM, HEAD_DIM, 1)))
    k_memref = S.make_tensor(k_ptr, S.bf16, S.make_layout((bh, SEQ_LEN, HEAD_DIM), (SEQ_LEN * HEAD_DIM, HEAD_DIM, 1)))
    v_memref = S.make_tensor(v_ptr, S.bf16, S.make_layout((bh, SEQ_LEN, HEAD_DIM), (SEQ_LEN * HEAD_DIM, HEAD_DIM, 1)))
    out_memref = S.make_tensor(out_ptr, S.bf16, S.make_layout((bh, SEQ_LEN, HEAD_DIM), (SEQ_LEN * HEAD_DIM, HEAD_DIM, 1)))

    row_max = neg_inf
    row_sum = zero
    acc = S.make_local((HEAD_DIM,), S.f32)
    for d in S.range(HEAD_DIM):
        acc[d] = zero

    for key_idx in S.range(SEQ_LEN):
        score = zero
        for d in S.range(HEAD_DIM):
            q_val = S.convert(q_memref[bh_idx, row_in_seq, d], S.f32)
            k_val = S.convert(k_memref[bh_idx, key_idx, d], S.f32)
            score = score + q_val * k_val
        scaled_score = score * scale_log2

        next_max = scaled_score if scaled_score > row_max else row_max
        prev_scale = S.exp2(row_max - next_max)
        prob = S.exp2(scaled_score - next_max)

        for d in S.range(HEAD_DIM):
            v_val = S.convert(v_memref[bh_idx, key_idx, d], S.f32)
            acc[d] = acc[d] * prev_scale + prob * v_val
        row_sum = row_sum * prev_scale + prob
        row_max = next_max

    inv_row_sum = one / row_sum
    for d in S.range(HEAD_DIM):
        out_memref[bh_idx, row_in_seq, d] = S.convert(acc[d] * inv_row_sum, S.bf16)


def substrate_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Tensors must be on CUDA/HIP device."
    assert Q.shape == K.shape == V.shape, "Q, K, and V must have the same shape."
    batch_size, num_heads, seq_len, head_dim = Q.shape
    assert seq_len == SEQ_LEN and head_dim == HEAD_DIM, "This kernel is specialized for (seq_len=512, head_dim=1024)."

    orig_dtype = Q.dtype
    q = Q.contiguous().to(dtype=torch.bfloat16)
    k = K.contiguous().to(dtype=torch.bfloat16)
    v = V.contiguous().to(dtype=torch.bfloat16)

    bh = batch_size * num_heads
    q3 = q.view(bh, seq_len, head_dim)
    k3 = k.view(bh, seq_len, head_dim)
    v3 = v.view(bh, seq_len, head_dim)
    out = torch.empty_like(q3)

    grid_rows = (seq_len + BLOCK_ROWS - 1) // BLOCK_ROWS
    naive_attention_kernel[lambda: ((grid_rows, bh, 1), (BLOCK_ROWS, 1, 1))](
        out,
        q3,
        k3,
        v3,
        bh,
        num_warps=1,
    )

    return out.view(batch_size, num_heads, seq_len, head_dim).to(dtype=orig_dtype)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return substrate_scaled_dot_product_attention(Q, K, V)
