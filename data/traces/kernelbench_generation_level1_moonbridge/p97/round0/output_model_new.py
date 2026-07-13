import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Tile constants for tiled dot-product GEMM ──
GEMM_THREADS = 256
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
THREADS_PER_ROW = 8
THREADS_PER_COL = 32
ROWS_PER_THREAD = BLOCK_M // THREADS_PER_COL
COLS_PER_THREAD = BLOCK_N // THREADS_PER_ROW

# ── Softmax constants ──
SOFTMAX_THREADS = 256


# ═══════════════════════════════════════════════════════════════════════
# Kernel 1: Batched Q @ K^T = S  (tiled dot-product GEMM)
# Grid: (g_n * total_bh, g_m, 1)
# ═══════════════════════════════════════════════════════════════════════

@avelang.jit
def qk_gemm_kernel(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    s_ptr: al.Pointer(al.bf16),
    total_bh: al.u32,
    g_n: al.u32,
    L: al.u32,
    D: al.u32,
):
    tid = al.thread_id(0)
    combined = al.block_id(0)
    block_m_idx = al.block_id(1)
    bh = combined // g_n
    block_n_idx = combined - bh * g_n

    thread_row = tid // THREADS_PER_ROW
    thread_col = tid - thread_row * THREADS_PER_ROW

    q_g = al.make_tensor(q_ptr, al.bf16, al.make_layout((total_bh, L, D), (L * D, D, 1)))
    k_g = al.make_tensor(k_ptr, al.bf16, al.make_layout((total_bh, L, D), (L * D, D, 1)))
    s_g = al.make_tensor(s_ptr, al.bf16, al.make_layout((total_bh, L, L), (L * L, L, 1)))

    shm_q = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    shm_k = al.make_shared((BLOCK_N, BLOCK_K), al.bf16)

    acc = al.make_local((ROWS_PER_THREAD * COLS_PER_THREAD,), al.f32)
    for a in al.range(ROWS_PER_THREAD * COLS_PER_THREAD):
        acc[a] = al.convert(0.0, al.f32)

    num_k_tiles = D // BLOCK_K
    for kt in al.range(num_k_tiles):
        k_start = kt * BLOCK_K

        elems_per_thread_q = (BLOCK_M * BLOCK_K) // GEMM_THREADS
        for e in al.range(elems_per_thread_q):
            idx = tid * elems_per_thread_q + e
            r = idx // BLOCK_K
            c = idx - r * BLOCK_K
            global_r = block_m_idx * BLOCK_M + r
            global_c = k_start + c
            if global_r < L and global_c < D:
                shm_q[r, c] = q_g[bh, global_r, global_c]

        elems_per_thread_k = (BLOCK_N * BLOCK_K) // GEMM_THREADS
        for e in al.range(elems_per_thread_k):
            idx = tid * elems_per_thread_k + e
            r = idx // BLOCK_K
            c = idx - r * BLOCK_K
            global_r = block_n_idx * BLOCK_N + r
            global_c = k_start + c
            if global_r < L and global_c < D:
                shm_k[r, c] = k_g[bh, global_r, global_c]

        al.syncthreads()

        for ri in al.range(ROWS_PER_THREAD):
            local_m = thread_row * ROWS_PER_THREAD + ri
            for ci in al.range(COLS_PER_THREAD):
                local_n = thread_col * COLS_PER_THREAD + ci
                acc_idx = ri * COLS_PER_THREAD + ci
                for kk in al.range(BLOCK_K):
                    a_val = al.convert(shm_q[local_m, kk], al.f32)
                    b_val = al.convert(shm_k[local_n, kk], al.f32)
                    acc[acc_idx] = acc[acc_idx] + a_val * b_val

        al.syncthreads()

    for ri in al.range(ROWS_PER_THREAD):
        local_m = thread_row * ROWS_PER_THREAD + ri
        global_m = block_m_idx * BLOCK_M + local_m
        for ci in al.range(COLS_PER_THREAD):
            local_n = thread_col * COLS_PER_THREAD + ci
            global_n = block_n_idx * BLOCK_N + local_n
            acc_idx = ri * COLS_PER_THREAD + ci
            if global_m < L and global_n < L:
                s_g[bh, global_m, global_n] = al.convert(acc[acc_idx], al.bf16)


# ═══════════════════════════════════════════════════════════════════════
# Kernel 2: Scale + row-wise softmax (shared-memory reduction)
# Grid: (L, total_bh, 1)
# ═══════════════════════════════════════════════════════════════════════

@avelang.jit
def scale_softmax_kernel(
    s_ptr: al.Pointer(al.bf16),
    p_ptr: al.Pointer(al.bf16),
    total_bh: al.u32,
    L: al.u32,
):
    tid = al.thread_id(0)
    row_idx = al.block_id(0)
    bh = al.block_id(1)

    s_g = al.make_tensor(s_ptr, al.bf16, al.make_layout((total_bh, L, L), (L * L, L, 1)))
    p_g = al.make_tensor(p_ptr, al.bf16, al.make_layout((total_bh, L, L), (L * L, L, 1)))

    smem_max = al.make_shared((SOFTMAX_THREADS,), al.f32)
    smem_sum = al.make_shared((SOFTMAX_THREADS,), al.f32)

    if row_idx < L:
        scale = al.convert(0.03125, al.f32)

        local_max = al.convert(-1e30, al.f32)
        for i in al.range(tid, L, SOFTMAX_THREADS):
            val = al.convert(s_g[bh, row_idx, i], al.f32)
            val = val * scale
            if val > local_max:
                local_max = val

        smem_max[tid] = local_max
        al.syncthreads()

        if tid < 128:
            if smem_max[tid + 128] > smem_max[tid]:
                smem_max[tid] = smem_max[tid + 128]
        al.syncthreads()
        if tid < 64:
            if smem_max[tid + 64] > smem_max[tid]:
                smem_max[tid] = smem_max[tid + 64]
        al.syncthreads()
        if tid < 32:
            if smem_max[tid + 32] > smem_max[tid]:
                smem_max[tid] = smem_max[tid + 32]
        al.syncthreads()
        if tid < 16:
            if smem_max[tid + 16] > smem_max[tid]:
                smem_max[tid] = smem_max[tid + 16]
        al.syncthreads()
        if tid < 8:
            if smem_max[tid + 8] > smem_max[tid]:
                smem_max[tid] = smem_max[tid + 8]
        al.syncthreads()
        if tid < 4:
            if smem_max[tid + 4] > smem_max[tid]:
                smem_max[tid] = smem_max[tid + 4]
        al.syncthreads()
        if tid < 2:
            if smem_max[tid + 2] > smem_max[tid]:
                smem_max[tid] = smem_max[tid + 2]
        al.syncthreads()
        if tid < 1:
            if smem_max[tid + 1] > smem_max[tid]:
                smem_max[tid] = smem_max[tid + 1]
        al.syncthreads()

        row_max = smem_max[0]

        local_sum = al.convert(0.0, al.f32)
        log2_e = al.convert(1.44269504089, al.f32)
        for i in al.range(tid, L, SOFTMAX_THREADS):
            val = al.convert(s_g[bh, row_idx, i], al.f32)
            val = val * scale
            diff = val - row_max
            local_sum = local_sum + al.exp2(diff * log2_e)

        smem_sum[tid] = local_sum
        al.syncthreads()

        if tid < 128:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
        al.syncthreads()

        row_sum = smem_sum[0]
        inv_sum = al.convert(1.0, al.f32) / row_sum

        for i in al.range(tid, L, SOFTMAX_THREADS):
            val = al.convert(s_g[bh, row_idx, i], al.f32)
            val = val * scale
            diff = val - row_max
            prob = al.exp2(diff * log2_e) * inv_sum
            p_g[bh, row_idx, i] = al.convert(prob, al.bf16)


# ═══════════════════════════════════════════════════════════════════════
# Kernel 3: Batched P @ V = O  (tiled dot-product GEMM)
# Grid: (g_n * total_bh, g_m, 1)
# ═══════════════════════════════════════════════════════════════════════

@avelang.jit
def sv_gemm_kernel(
    p_ptr: al.Pointer(al.bf16),
    v_ptr: al.Pointer(al.bf16),
    o_ptr: al.Pointer(al.bf16),
    total_bh: al.u32,
    g_n: al.u32,
    L: al.u32,
    D: al.u32,
):
    tid = al.thread_id(0)
    combined = al.block_id(0)
    block_m_idx = al.block_id(1)
    bh = combined // g_n
    block_n_idx = combined - bh * g_n

    thread_row = tid // THREADS_PER_ROW
    thread_col = tid - thread_row * THREADS_PER_ROW

    p_g = al.make_tensor(p_ptr, al.bf16, al.make_layout((total_bh, L, L), (L * L, L, 1)))
    v_g = al.make_tensor(v_ptr, al.bf16, al.make_layout((total_bh, L, D), (L * D, D, 1)))
    o_g = al.make_tensor(o_ptr, al.bf16, al.make_layout((total_bh, L, D), (L * D, D, 1)))

    shm_p = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
    shm_v = al.make_shared((BLOCK_N, BLOCK_K), al.bf16)

    acc = al.make_local((ROWS_PER_THREAD * COLS_PER_THREAD,), al.f32)
    for a in al.range(ROWS_PER_THREAD * COLS_PER_THREAD):
        acc[a] = al.convert(0.0, al.f32)

    num_k_tiles = L // BLOCK_K
    for kt in al.range(num_k_tiles):
        k_start = kt * BLOCK_K

        elems_per_thread_p = (BLOCK_M * BLOCK_K) // GEMM_THREADS
        for e in al.range(elems_per_thread_p):
            idx = tid * elems_per_thread_p + e
            r = idx // BLOCK_K
            c = idx - r * BLOCK_K
            global_r = block_m_idx * BLOCK_M + r
            global_c = k_start + c
            if global_r < L and global_c < L:
                shm_p[r, c] = p_g[bh, global_r, global_c]

        elems_per_thread_v = (BLOCK_N * BLOCK_K) // GEMM_THREADS
        for e in al.range(elems_per_thread_v):
            idx = tid * elems_per_thread_v + e
            r = idx // BLOCK_K
            c = idx - r * BLOCK_K
            global_r = block_n_idx * BLOCK_N + r
            global_c = k_start + c
            if global_r < D and global_c < L:
                shm_v[r, c] = v_g[bh, global_c, global_r]

        al.syncthreads()

        for ri in al.range(ROWS_PER_THREAD):
            local_m = thread_row * ROWS_PER_THREAD + ri
            for ci in al.range(COLS_PER_THREAD):
                local_n = thread_col * COLS_PER_THREAD + ci
                acc_idx = ri * COLS_PER_THREAD + ci
                for kk in al.range(BLOCK_K):
                    a_val = al.convert(shm_p[local_m, kk], al.f32)
                    b_val = al.convert(shm_v[local_n, kk], al.f32)
                    acc[acc_idx] = acc[acc_idx] + a_val * b_val

        al.syncthreads()

    for ri in al.range(ROWS_PER_THREAD):
        local_m = thread_row * ROWS_PER_THREAD + ri
        global_m = block_m_idx * BLOCK_M + local_m
        for ci in al.range(COLS_PER_THREAD):
            local_n = thread_col * COLS_PER_THREAD + ci
            global_n = block_n_idx * BLOCK_N + local_n
            acc_idx = ri * COLS_PER_THREAD + ci
            if global_m < L and global_n < D:
                o_g[bh, global_m, global_n] = al.convert(acc[acc_idx], al.bf16)


# ═══════════════════════════════════════════════════════════════════════
# Host wrapper
# ═══════════════════════════════════════════════════════════════════════

def _prepare_bf16(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    return t.to(dtype=torch.bfloat16).contiguous()


def _sdpa_avelang(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    Q = _prepare_bf16(Q)
    K = _prepare_bf16(K)
    V = _prepare_bf16(V)

    B, H, L, D = Q.shape
    total_bh = B * H

    Q_flat = Q.reshape(total_bh, L, D).contiguous()
    K_flat = K.reshape(total_bh, L, D).contiguous()
    V_flat = V.reshape(total_bh, L, D).contiguous()

    g_m = (L + BLOCK_M - 1) // BLOCK_M
    g_n_qk = (L + BLOCK_N - 1) // BLOCK_N
    g_n_sv = (D + BLOCK_N - 1) // BLOCK_N

    # Step 1: Q @ K^T = S  (2D grid: combine N and batch)
    S = torch.empty((total_bh, L, L), device=Q.device, dtype=torch.bfloat16)
    grid_qk = (g_n_qk * total_bh, g_m, 1)
    qk_gemm_kernel[lambda: (grid_qk, (GEMM_THREADS, 1, 1))](Q_flat, K_flat, S, total_bh, g_n_qk, L, D)

    # Step 2: Scale + Softmax (already 2D grid)
    P = torch.empty((total_bh, L, L), device=Q.device, dtype=torch.bfloat16)
    grid_sm = (L, total_bh, 1)
    scale_softmax_kernel[lambda: (grid_sm, (SOFTMAX_THREADS, 1, 1))](S, P, total_bh, L)

    # Step 3: P @ V = O  (2D grid: combine N and batch)
    O = torch.empty((total_bh, L, D), device=Q.device, dtype=torch.bfloat16)
    grid_sv = (g_n_sv * total_bh, g_m, 1)
    sv_gemm_kernel[lambda: (grid_sv, (GEMM_THREADS, 1, 1))](P, V_flat, O, total_bh, g_n_sv, L, D)

    return O.reshape(B, H, L, D)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return _sdpa_avelang(Q, K, V)
