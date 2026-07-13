import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

_BLOCK_M = 128
_BLOCK_N = 64
_BLOCK_D = 64
_NUM_THREADS = 256

_LANES = _NUM_THREADS // _BLOCK_M  # 2
_COLS_PER_LANE = _BLOCK_N // _LANES  # 32
_D_PER_LANE = _BLOCK_D // _LANES  # 32


@avelang.jit
def flash_attention_kernel(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    v_ptr: al.Pointer(al.bf16),
    o_fp32_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch: al.i32,
    num_heads: al.i32,
    seq_len: al.i32,
    head_dim: al.i32,
    BLOCK_M: al.constexpr,
    BLOCK_N: al.constexpr,
    BLOCK_D: al.constexpr,
    LANES: al.constexpr,
    COLS_PER_LANE: al.constexpr,
    D_PER_LANE: al.constexpr,
):
    bh_id = al.block_id(0)
    m_block = al.block_id(1)
    tid = al.thread_id(0)
    bd0 = al.block_dim(0)

    batch_idx = bh_id // num_heads
    head_idx = bh_id - batch_idx * num_heads
    q_seq_start = m_block * BLOCK_M

    row = tid % BLOCK_M
    lane = tid // BLOCK_M

    d_f32 = al.convert(head_dim, al.f32)
    scale = al.convert(1.0, al.f32) / al.sqrt(d_f32)

    stride_b = num_heads * seq_len * head_dim
    stride_h = seq_len * head_dim
    stride_s = head_dim
    one = al.convert(1, al.i32)

    layout_4d = al.make_layout(
        (batch, num_heads, seq_len, head_dim),
        (stride_b, stride_h, stride_s, one),
    )
    q = al.make_tensor(q_ptr, al.bf16, layout_4d)
    k = al.make_tensor(k_ptr, al.bf16, layout_4d)
    v = al.make_tensor(v_ptr, al.bf16, layout_4d)
    o_fp32 = al.make_tensor(o_fp32_ptr, al.f32, layout_4d)
    out = al.make_tensor(out_ptr, al.bf16, layout_4d)

    q_tile = al.make_shared((BLOCK_M, BLOCK_D), al.bf16)
    kv_tile = al.make_shared((BLOCK_N, BLOCK_D), al.bf16)
    s_tile = al.make_shared((BLOCK_M, BLOCK_N), al.f32)
    scratch = al.make_shared((BLOCK_M, 2), al.f32)
    m_shared = al.make_shared((BLOCK_M,), al.f32)
    l_shared = al.make_shared((BLOCK_M,), al.f32)

    neg_inf = al.convert(-10000000000.0, al.f32)
    zero_f32 = al.convert(0.0, al.f32)

    if lane == al.convert(0, al.i32):
        m_shared[row] = neg_inf
        l_shared[row] = zero_f32
    al.syncthreads()

    num_kv_blocks = seq_len // BLOCK_N
    num_d_blocks = head_dim // BLOCK_D

    tile_elems_q = BLOCK_M * BLOCK_D
    tile_elems_kv = BLOCK_N * BLOCK_D
    q_elems_per_thread = tile_elems_q // bd0
    kv_elems_per_thread = tile_elems_kv // bd0

    for kv_block in al.range(num_kv_blocks):
        kv_seq_start = kv_block * BLOCK_N

        if lane == al.convert(0, al.i32):
            for c in al.range(BLOCK_N):
                s_tile[row, c] = zero_f32
        al.syncthreads()

        for d_block in al.range(num_d_blocks):
            d_start = d_block * BLOCK_D

            for idx in al.range(q_elems_per_thread):
                flat = tid + idx * bd0
                r = flat // BLOCK_D
                c = flat - r * BLOCK_D
                q_tile[r, c] = q[batch_idx, head_idx, q_seq_start + r, d_start + c]

            for idx in al.range(kv_elems_per_thread):
                flat = tid + idx * bd0
                r = flat // BLOCK_D
                c = flat - r * BLOCK_D
                kv_tile[r, c] = k[batch_idx, head_idx, kv_seq_start + r, d_start + c]

            al.syncthreads()

            col_start = lane * COLS_PER_LANE
            for j_offset in al.range(COLS_PER_LANE):
                j = col_start + j_offset
                dot = zero_f32
                for d_i in al.range(BLOCK_D):
                    q_val = al.convert(q_tile[row, d_i], al.f32)
                    k_val = al.convert(kv_tile[j, d_i], al.f32)
                    dot = dot + q_val * k_val
                s_tile[row, j] = s_tile[row, j] + dot

            al.syncthreads()

        col_start = lane * COLS_PER_LANE

        partial_max = neg_inf
        for j_offset in al.range(COLS_PER_LANE):
            j = col_start + j_offset
            s_val = s_tile[row, j] * scale
            s_tile[row, j] = s_val
            if s_val > partial_max:
                partial_max = s_val

        old_m = m_shared[row]
        scratch[row, lane] = partial_max
        al.syncthreads()

        if lane == al.convert(0, al.i32):
            combined = partial_max
            other = scratch[row, al.convert(1, al.i32)]
            if other > combined:
                combined = other
            if combined > old_m:
                m_shared[row] = combined
            new_m = m_shared[row]
            scratch[row, 0] = al.exp(old_m - new_m)
            scratch[row, 1] = new_m
        al.syncthreads()

        alpha = scratch[row, 0]
        new_m = scratch[row, 1]

        partial_sum = zero_f32
        for j_offset in al.range(COLS_PER_LANE):
            j = col_start + j_offset
            p_val = al.exp(s_tile[row, j] - new_m)
            s_tile[row, j] = p_val
            partial_sum = partial_sum + p_val

        scratch[row, lane] = partial_sum
        al.syncthreads()

        if lane == al.convert(0, al.i32):
            total_sum = scratch[row, 0] + scratch[row, al.convert(1, al.i32)]
            l_shared[row] = alpha * l_shared[row] + total_sum
        al.syncthreads()

        for d_block in al.range(num_d_blocks):
            d_start = d_block * BLOCK_D

            for idx in al.range(kv_elems_per_thread):
                flat = tid + idx * bd0
                r = flat // BLOCK_D
                c = flat - r * BLOCK_D
                kv_tile[r, c] = v[batch_idx, head_idx, kv_seq_start + r, d_start + c]

            al.syncthreads()

            d_col_start = lane * D_PER_LANE
            for d_offset in al.range(D_PER_LANE):
                d_col = d_col_start + d_offset
                contrib = zero_f32
                for j in al.range(BLOCK_N):
                    p_val = s_tile[row, j]
                    v_val = al.convert(kv_tile[j, d_col], al.f32)
                    contrib = contrib + p_val * v_val

                old_o = o_fp32[batch_idx, head_idx, q_seq_start + row, d_start + d_col]
                new_o = alpha * old_o + contrib
                o_fp32[batch_idx, head_idx, q_seq_start + row, d_start + d_col] = new_o

            al.syncthreads()

    for d_block in al.range(num_d_blocks):
        d_start = d_block * BLOCK_D
        d_col_start = lane * D_PER_LANE
        for d_offset in al.range(D_PER_LANE):
            d_col = d_col_start + d_offset
            o_val = o_fp32[batch_idx, head_idx, q_seq_start + row, d_start + d_col]
            normalized = o_val / l_shared[row]
            out[batch_idx, head_idx, q_seq_start + row, d_start + d_col] = al.convert(normalized, al.bf16)


def _sdpa_avelang(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D) and V.shape == (B, H, S, D)

    Q_bf16 = Q.to(torch.bfloat16).contiguous()
    K_bf16 = K.to(torch.bfloat16).contiguous()
    V_bf16 = V.to(torch.bfloat16).contiguous()

    o_fp32 = torch.zeros(B, H, S, D, dtype=torch.float32, device=Q.device)
    out_bf16 = torch.empty(B, H, S, D, dtype=torch.bfloat16, device=Q.device)

    grid_m = (S + _BLOCK_M - 1) // _BLOCK_M
    grid = (B * H, grid_m, 1)
    block = (_NUM_THREADS, 1, 1)

    flash_attention_kernel[lambda: (grid, block)](
        Q_bf16, K_bf16, V_bf16, o_fp32, out_bf16,
        B, H, S, D,
        _BLOCK_M, _BLOCK_N, _BLOCK_D, _LANES, _COLS_PER_LANE, _D_PER_LANE,
    )

    return out_bf16.to(Q.dtype)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return _sdpa_avelang(Q, K, V)
