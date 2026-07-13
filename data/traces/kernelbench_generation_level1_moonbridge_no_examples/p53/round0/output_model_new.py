import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_N = 8
BLOCK_SIZE = 256


# ---------------------------------------------------------------------------
# Kernel: reduce over dim 0  (B, D, N) -> (D, N)
# ---------------------------------------------------------------------------
@avelang.jit
def min_reduce_dim0_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    D: al.i32,
    N: al.i32,
    TILE: al.constexpr,
    BLK: al.constexpr,
):
    T = TILE
    BS = BLK
    x_layout = al.make_layout((B, D, N), (D * N, N, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    out_layout = al.make_layout((D, N), (N, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    tid = al.thread_id(0)
    d_idx = al.block_id(0)
    n_start = al.block_id(1) * T
    INF = al.convert(3.402823e+38, al.f32)

    min0 = INF; min1 = INF; min2 = INF; min3 = INF
    min4 = INF; min5 = INF; min6 = INF; min7 = INF

    for b in al.range(tid, B, BS):
        v = al.convert(x[b, d_idx, n_start + 0], al.f32)
        if v < min0: min0 = v
        v = al.convert(x[b, d_idx, n_start + 1], al.f32)
        if v < min1: min1 = v
        v = al.convert(x[b, d_idx, n_start + 2], al.f32)
        if v < min2: min2 = v
        v = al.convert(x[b, d_idx, n_start + 3], al.f32)
        if v < min3: min3 = v
        v = al.convert(x[b, d_idx, n_start + 4], al.f32)
        if v < min4: min4 = v
        v = al.convert(x[b, d_idx, n_start + 5], al.f32)
        if v < min5: min5 = v
        v = al.convert(x[b, d_idx, n_start + 6], al.f32)
        if v < min6: min6 = v
        v = al.convert(x[b, d_idx, n_start + 7], al.f32)
        if v < min7: min7 = v

    shared = al.make_shared((BS,), al.f32)

    # --- k=0 ---
    val = min0
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 0 < N:
            out[d_idx, n_start + 0] = al.convert(best, al.bf16)

    # --- k=1 ---
    val = min1
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 1 < N:
            out[d_idx, n_start + 1] = al.convert(best, al.bf16)

    # --- k=2 ---
    val = min2
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 2 < N:
            out[d_idx, n_start + 2] = al.convert(best, al.bf16)

    # --- k=3 ---
    val = min3
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 3 < N:
            out[d_idx, n_start + 3] = al.convert(best, al.bf16)

    # --- k=4 ---
    val = min4
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 4 < N:
            out[d_idx, n_start + 4] = al.convert(best, al.bf16)

    # --- k=5 ---
    val = min5
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 5 < N:
            out[d_idx, n_start + 5] = al.convert(best, al.bf16)

    # --- k=6 ---
    val = min6
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 6 < N:
            out[d_idx, n_start + 6] = al.convert(best, al.bf16)

    # --- k=7 ---
    val = min7
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 7 < N:
            out[d_idx, n_start + 7] = al.convert(best, al.bf16)


# ---------------------------------------------------------------------------
# Kernel: reduce over dim 1  (B, D, N) -> (B, N)
# ---------------------------------------------------------------------------
@avelang.jit
def min_reduce_dim1_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    D: al.i32,
    N: al.i32,
    TILE: al.constexpr,
    BLK: al.constexpr,
):
    T = TILE
    BS = BLK
    x_layout = al.make_layout((B, D, N), (D * N, N, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    out_layout = al.make_layout((B, N), (N, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    tid = al.thread_id(0)
    b_idx = al.block_id(0)
    n_start = al.block_id(1) * T
    INF = al.convert(3.402823e+38, al.f32)

    min0 = INF; min1 = INF; min2 = INF; min3 = INF
    min4 = INF; min5 = INF; min6 = INF; min7 = INF

    for d in al.range(tid, D, BS):
        v = al.convert(x[b_idx, d, n_start + 0], al.f32)
        if v < min0: min0 = v
        v = al.convert(x[b_idx, d, n_start + 1], al.f32)
        if v < min1: min1 = v
        v = al.convert(x[b_idx, d, n_start + 2], al.f32)
        if v < min2: min2 = v
        v = al.convert(x[b_idx, d, n_start + 3], al.f32)
        if v < min3: min3 = v
        v = al.convert(x[b_idx, d, n_start + 4], al.f32)
        if v < min4: min4 = v
        v = al.convert(x[b_idx, d, n_start + 5], al.f32)
        if v < min5: min5 = v
        v = al.convert(x[b_idx, d, n_start + 6], al.f32)
        if v < min6: min6 = v
        v = al.convert(x[b_idx, d, n_start + 7], al.f32)
        if v < min7: min7 = v

    shared = al.make_shared((BS,), al.f32)

    # --- k=0 ---
    val = min0
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 0 < N:
            out[b_idx, n_start + 0] = al.convert(best, al.bf16)

    # --- k=1 ---
    val = min1
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 1 < N:
            out[b_idx, n_start + 1] = al.convert(best, al.bf16)

    # --- k=2 ---
    val = min2
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 2 < N:
            out[b_idx, n_start + 2] = al.convert(best, al.bf16)

    # --- k=3 ---
    val = min3
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 3 < N:
            out[b_idx, n_start + 3] = al.convert(best, al.bf16)

    # --- k=4 ---
    val = min4
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 4 < N:
            out[b_idx, n_start + 4] = al.convert(best, al.bf16)

    # --- k=5 ---
    val = min5
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 5 < N:
            out[b_idx, n_start + 5] = al.convert(best, al.bf16)

    # --- k=6 ---
    val = min6
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 6 < N:
            out[b_idx, n_start + 6] = al.convert(best, al.bf16)

    # --- k=7 ---
    val = min7
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if n_start + 7 < N:
            out[b_idx, n_start + 7] = al.convert(best, al.bf16)


# ---------------------------------------------------------------------------
# Kernel: reduce over dim 2  (B, D, N) -> (B, D)
# ---------------------------------------------------------------------------
@avelang.jit
def min_reduce_dim2_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    D: al.i32,
    N: al.i32,
    TILE: al.constexpr,
    BLK: al.constexpr,
):
    T = TILE
    BS = BLK
    x_layout = al.make_layout((B, D, N), (D * N, N, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    out_layout = al.make_layout((B, D), (D, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    tid = al.thread_id(0)
    b_idx = al.block_id(0)
    d_start = al.block_id(1) * T
    INF = al.convert(3.402823e+38, al.f32)

    min0 = INF; min1 = INF; min2 = INF; min3 = INF
    min4 = INF; min5 = INF; min6 = INF; min7 = INF

    for n in al.range(tid, N, BS):
        v = al.convert(x[b_idx, d_start + 0, n], al.f32)
        if v < min0: min0 = v
        v = al.convert(x[b_idx, d_start + 1, n], al.f32)
        if v < min1: min1 = v
        v = al.convert(x[b_idx, d_start + 2, n], al.f32)
        if v < min2: min2 = v
        v = al.convert(x[b_idx, d_start + 3, n], al.f32)
        if v < min3: min3 = v
        v = al.convert(x[b_idx, d_start + 4, n], al.f32)
        if v < min4: min4 = v
        v = al.convert(x[b_idx, d_start + 5, n], al.f32)
        if v < min5: min5 = v
        v = al.convert(x[b_idx, d_start + 6, n], al.f32)
        if v < min6: min6 = v
        v = al.convert(x[b_idx, d_start + 7, n], al.f32)
        if v < min7: min7 = v

    shared = al.make_shared((BS,), al.f32)

    # --- k=0 ---
    val = min0
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if d_start + 0 < D:
            out[b_idx, d_start + 0] = al.convert(best, al.bf16)

    # --- k=1 ---
    val = min1
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if d_start + 1 < D:
            out[b_idx, d_start + 1] = al.convert(best, al.bf16)

    # --- k=2 ---
    val = min2
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if d_start + 2 < D:
            out[b_idx, d_start + 2] = al.convert(best, al.bf16)

    # --- k=3 ---
    val = min3
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if d_start + 3 < D:
            out[b_idx, d_start + 3] = al.convert(best, al.bf16)

    # --- k=4 ---
    val = min4
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if d_start + 4 < D:
            out[b_idx, d_start + 4] = al.convert(best, al.bf16)

    # --- k=5 ---
    val = min5
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if d_start + 5 < D:
            out[b_idx, d_start + 5] = al.convert(best, al.bf16)

    # --- k=6 ---
    val = min6
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if d_start + 6 < D:
            out[b_idx, d_start + 6] = al.convert(best, al.bf16)

    # --- k=7 ---
    val = min7
    o = al.shuffle_down(val, 32, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 16, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 8, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 4, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 2, 64)
    if o < val: val = o
    o = al.shuffle_down(val, 1, 64)
    if o < val: val = o
    shared[tid] = val
    al.syncthreads()
    if tid == 0:
        best = shared[0]
        if shared[64] < best: best = shared[64]
        if shared[128] < best: best = shared[128]
        if shared[192] < best: best = shared[192]
        if d_start + 7 < D:
            out[b_idx, d_start + 7] = al.convert(best, al.bf16)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------
def avelang_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    assert x.ndim == 3, f"Expected 3D input, got {x.ndim}D"
    assert x.dtype == torch.bfloat16, f"Expected bf16 input, got {x.dtype}"

    x = x.contiguous()
    B, D, N = x.shape
    block = (BLOCK_SIZE, 1, 1)

    if dim == 0:
        out = torch.empty(D, N, dtype=torch.bfloat16, device=x.device)
        grid_N = (N + TILE_N - 1) // TILE_N
        grid = (D, grid_N, 1)
        min_reduce_dim0_kernel[lambda: (grid, block)](
            x.data_ptr(), out.data_ptr(),
            B, D, N, TILE_N, BLOCK_SIZE,
        )
    elif dim == 1:
        out = torch.empty(B, N, dtype=torch.bfloat16, device=x.device)
        grid_N = (N + TILE_N - 1) // TILE_N
        grid = (B, grid_N, 1)
        min_reduce_dim1_kernel[lambda: (grid, block)](
            x.data_ptr(), out.data_ptr(),
            B, D, N, TILE_N, BLOCK_SIZE,
        )
    elif dim == 2:
        out = torch.empty(B, D, dtype=torch.bfloat16, device=x.device)
        grid_D = (D + TILE_N - 1) // TILE_N
        grid = (B, grid_D, 1)
        min_reduce_dim2_kernel[lambda: (grid, block)](
            x.data_ptr(), out.data_ptr(),
            B, D, N, TILE_N, BLOCK_SIZE,
        )
    else:
        raise ValueError(f"Unsupported dim={dim} for 3D tensor")

    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_min(x, self.dim)
