import torch
import torch.nn as nn
import avelang
import avelang.language as al

GROUP_M = 64
GROUP_N = 64
GROUP_K = 16
MMA_M = 32
MMA_N = 32
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
WARPS_M = 2
WARPS_N = 2
ACC_SIZE = 16
VEC_ELEMS = 8
BF16_BYTES = 2
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW
ROW_U32 = A_VECS_PER_ROW * 4
K_STRIDE = 2 * GROUP_K


@avelang.jit
def _load_global_a(
    shm_a: al.Tensor((SHM_A_VECS, 4), al.u32),
    x_rsrc: al.Tensor((4,), al.u32),
    block_m: al.u32,
    k_base: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    shm_vecs = al.convert(SHM_A_VECS, al.u32)
    vecs_per_row = al.convert(A_VECS_PER_ROW, al.u32)
    if idx < shm_vecs:
        row = idx // vecs_per_row
        col_vec = idx % vecs_per_row
        off = ((block_m * GROUP_M + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_a[idx] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, off, 0)


@avelang.jit
def _load_global_b(
    shm_b: al.Tensor((SHM_B_VECS, 4), al.u32),
    w_rsrc: al.Tensor((4,), al.u32),
    block_n: al.u32,
    k_base: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    shm_vecs = al.convert(SHM_B_VECS, al.u32)
    vecs_per_row = al.convert(B_VECS_PER_ROW, al.u32)
    if idx < shm_vecs:
        row = idx // vecs_per_row
        col_vec = idx % vecs_per_row
        off = ((block_n * GROUP_N + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_b[idx] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, off, 0)


@avelang.jit
def _fetch_operand(
    ret: al.Tensor((2, 4), al.bf16),
    shm: al.Tensor((SHM_A_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
):
    ret_u32 = al.view(ret, al.Tensor((4,), al.u32))
    shm_u32 = al.view(shm, al.Tensor((SHM_A_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32

    ret_u32[0] = shm_u32[row_base + k_group_u32]
    ret_u32[1] = shm_u32[row_base + k_group_u32 + 1]
    ret_u32[2] = shm_u32[row_base + 4 + k_group_u32]
    ret_u32[3] = shm_u32[row_base + 5 + k_group_u32]


@avelang.jit
def fused_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid = al.thread_id(0)
    block_n = al.block_id(0)
    block_m = al.block_id(1)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // WARPS_N
    warp_col = wid % WARPS_N

    x_memref = al.make_tensor(x_ptr, al.bf16, al.make_layout((m * k,), (1,)))
    w_memref = al.make_tensor(w_ptr, al.bf16, al.make_layout((n * k,), (1,)))
    g_bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    x_rsrc = al.amdgpu.make_rsrc(x_memref, m * k * BF16_BYTES)
    w_rsrc = al.amdgpu.make_rsrc(w_memref, n * k * BF16_BYTES)

    shm_a0 = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_a1 = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b0 = al.make_shared((SHM_B_VECS, 4), al.u32)
    shm_b1 = al.make_shared((SHM_B_VECS, 4), al.u32)

    a_reg = al.make_local((2, 4), al.bf16)
    b_reg = al.make_local((2, 4), al.bf16)
    acc = al.make_local((ACC_SIZE,), al.f32)
    for i in al.range(ACC_SIZE):
        acc[i] = 0

    zero_f = al.convert(0.0, al.f32)
    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    tile_i = warp_row
    tile_j = warp_col

    c0 = al.convert(0, al.u32)
    group_k_c = al.convert(GROUP_K, al.u32)

    # ============================================================
    # Prologue: load tile k=0 into buffer 0, compute it, preload tile 1
    # ============================================================
    _load_global_a(shm_a0, x_rsrc, block_m, c0, k, tid)
    _load_global_b(shm_b0, w_rsrc, block_n, c0, k, tid)
    al.syncthreads()

    _load_global_a(shm_a1, x_rsrc, block_m, group_k_c, k, tid)
    _load_global_b(shm_b1, w_rsrc, block_n, group_k_c, k, tid)

    _fetch_operand(a_reg, shm_a0, tile_i, lane)
    _fetch_operand(b_reg, shm_b0, tile_j, lane)
    a_u32_0 = al.view(a_reg[0], al.Tensor((2,), al.u32))
    b_u32_0 = al.view(b_reg[0], al.Tensor((2,), al.u32))
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32_0, b_u32_0, acc)
    a_u32_1 = al.view(a_reg[1], al.Tensor((2,), al.u32))
    b_u32_1 = al.view(b_reg[1], al.Tensor((2,), al.u32))
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32_1, b_u32_1, acc)

    al.syncthreads()

    # ============================================================
    # Main loop: double-buffered, software-pipelined, unrolled by 2
    # Structure per iteration (processes 2*GROUP_K K elements):
    #   1. Issue async loads for tile[k] into sm0
    #   2. Compute tile[k-GROUP_K] from sm1 (while sm0 loads fly)
    #   3. Sync (ensure sm1 reads done before overwriting)
    #   4. Issue async loads for tile[k+GROUP_K] into sm1
    #   5. Sync (ensure sm0 loads done)
    #   6. Compute tile[k] from sm0 (while sm1 loads fly)
    #   7. Sync (ensure sm1 loads done)
    # ============================================================
    for k_block in al.range(K_STRIDE, k, K_STRIDE):
        k_block_u32 = al.convert(k_block, al.u32)
        k_next = k_block_u32 + group_k_c

        # --- Step 1+2: load sm0 (async), compute from sm1 ---
        _load_global_a(shm_a0, x_rsrc, block_m, k_block_u32, k, tid)
        _load_global_b(shm_b0, w_rsrc, block_n, k_block_u32, k, tid)

        _fetch_operand(a_reg, shm_a1, tile_i, lane)
        _fetch_operand(b_reg, shm_b1, tile_j, lane)
        a_u32_0 = al.view(a_reg[0], al.Tensor((2,), al.u32))
        b_u32_0 = al.view(b_reg[0], al.Tensor((2,), al.u32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32_0, b_u32_0, acc)
        a_u32_1 = al.view(a_reg[1], al.Tensor((2,), al.u32))
        b_u32_1 = al.view(b_reg[1], al.Tensor((2,), al.u32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32_1, b_u32_1, acc)

        # --- Step 3: barrier ensures sm1 reads complete before overwriting ---
        al.syncthreads()

        # --- Step 4: load sm1 (async, overwrites) ---
        _load_global_a(shm_a1, x_rsrc, block_m, k_next, k, tid)
        _load_global_b(shm_b1, w_rsrc, block_n, k_next, k, tid)

        # --- Step 5: barrier ensures sm0 loads complete ---
        al.syncthreads()

        # --- Step 6: compute from sm0 (while sm1 loads fly) ---
        _fetch_operand(a_reg, shm_a0, tile_i, lane)
        _fetch_operand(b_reg, shm_b0, tile_j, lane)
        a_u32_0 = al.view(a_reg[0], al.Tensor((2,), al.u32))
        b_u32_0 = al.view(b_reg[0], al.Tensor((2,), al.u32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32_0, b_u32_0, acc)
        a_u32_1 = al.view(a_reg[1], al.Tensor((2,), al.u32))
        b_u32_1 = al.view(b_reg[1], al.Tensor((2,), al.u32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32_1, b_u32_1, acc)

        # --- Step 7: barrier ensures sm1 loads complete ---
        al.syncthreads()

    # ============================================================
    # Epilogue: compute last tile from buffer 1
    # ============================================================
    _fetch_operand(a_reg, shm_a1, tile_i, lane)
    _fetch_operand(b_reg, shm_b1, tile_j, lane)
    a_u32_0 = al.view(a_reg[0], al.Tensor((2,), al.u32))
    b_u32_0 = al.view(b_reg[0], al.Tensor((2,), al.u32))
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32_0, b_u32_0, acc)
    a_u32_1 = al.view(a_reg[1], al.Tensor((2,), al.u32))
    b_u32_1 = al.view(b_reg[1], al.Tensor((2,), al.u32))
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_u32_1, b_u32_1, acc)

    # ============================================================
    # Output writeback: bias + ReLU
    # ============================================================
    col = block_col_base + warp_col * MMA_N + lane_col
    bias_val = al.convert(g_bias[col], al.f32)

    for t in al.range(ACC_SIZE):
        row_base = block_row_base + warp_row * MMA_M
        row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
        result = acc[t] + bias_val
        if result < zero_f:
            result = zero_f
        g_out[row, col] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        Bv = x.shape[0]
        Kv = x.shape[1]
        Nv = self.bias.shape[0]

        w = self.gemm.weight.to(device=x.device, dtype=x.dtype).contiguous()
        eb = self.bias.to(device=x.device, dtype=x.dtype).contiguous()
        y = torch.empty((Bv, Nv), device=x.device, dtype=x.dtype)

        gm = (Bv + GROUP_M - 1) // GROUP_M
        gn = (Nv + GROUP_N - 1) // GROUP_N

        fused_kernel[lambda: ((gn, gm, 1), (THREADS, 1, 1))](
            x.contiguous(), w, eb, y, Bv, Nv, Kv
        )
        return y
