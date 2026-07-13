import torch
import torch.nn as nn
import avelang
import avelang.language as al

SQRT_2 = 1.4142135623730951
NEG_SLOPE = 0.01
BF16_BYTES = 2

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS


@avelang.jit
def gemm_kernel(
    X_ptr: al.Pointer(al.bf16),
    W_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    tid = al.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid - wid * WARP_SIZE
    warp_m = wid // 2
    warp_n = wid - warp_m * 2
    lane_col = wtid - (wtid // 32) * 32
    lane_group = wtid // 32

    block_m_idx = al.block_id(1)
    block_n_idx = al.block_id(0)
    block_m = block_m_idx * BLOCK_M
    block_n = block_n_idx * BLOCK_N

    X_flat = al.make_tensor(X_ptr, al.bf16, al.make_layout((M * K,), (1,)))
    W_flat = al.make_tensor(W_ptr, al.bf16, al.make_layout((N * K,), (1,)))

    a_rows = M - block_m
    b_rows = N - block_n
    if a_rows > BLOCK_M:
        a_rows = BLOCK_M
    if b_rows > BLOCK_N:
        b_rows = BLOCK_N

    X_block = al.subview(X_flat, (block_m * K,), (a_rows * K,), (1,))
    W_block = al.subview(W_flat, (block_n * K,), (b_rows * K,), (1,))
    X_rsrc = al.amdgpu.make_rsrc(X_block, a_rows * K * BF16_BYTES)
    W_rsrc = al.amdgpu.make_rsrc(W_block, b_rows * K * BF16_BYTES)

    a_rows_lds = BLOCK_M * (BLOCK_K // 8)
    lds_cols = BLOCK_K // 4
    A_LDS = al.make_shared((a_rows_lds, lds_cols), al.i32)
    B_LDS = al.make_shared((a_rows_lds, lds_cols), al.i32)
    C_SMEM = al.make_shared((BLOCK_M, BLOCK_N), al.f32)

    acc = al.full((16,), 0.0, al.f32)
    zero = al.convert(0, al.i32)
    num_k_tiles = K // BLOCK_K

    for kt in al.range(num_k_tiles):
        k_base = kt * BLOCK_K

        if tid < 128:
            local_row = tid - (tid // 64) * 64
            kg = tid // 64
            offset_bf16 = local_row * K + k_base + kg * 8
            load_offset = al.convert(offset_bf16 * BF16_BYTES, al.i32)
            A_LDS[tid] = al.amdgpu.raw_buffer_load_x4(X_rsrc, zero, load_offset, 0)

        if tid >= 128:
            local_tid = tid - 128
            local_row = local_tid - (local_tid // 64) * 64
            kg = local_tid // 64
            offset_bf16 = local_row * K + k_base + kg * 8
            load_offset = al.convert(offset_bf16 * BF16_BYTES, al.i32)
            B_LDS[local_tid] = al.amdgpu.raw_buffer_load_x4(W_rsrc, zero, load_offset, 0)

        al.syncthreads()

        a_entry = warp_m * 32 + lane_col + lane_group * BLOCK_M
        b_entry = warp_n * 32 + lane_col + lane_group * BLOCK_N
        a_words = A_LDS[a_entry]
        b_words = B_LDS[b_entry]
        a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))
        b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    for r in al.range(16):
        smem_col = ((r // 4) * 8) + lane_group * 4 + (r - (r // 4) * 4)
        C_SMEM[warp_m * 32 + lane_col, warp_n * 32 + smem_col] = acc[r]

    al.syncthreads()

    C_out = al.make_tensor(C_ptr, al.bf16, al.make_layout((M, N), (N, 1)))
    for r in al.range(16):
        smem_col = ((r // 4) * 8) + lane_group * 4 + (r - (r // 4) * 4)
        smem_row = warp_m * 32 + lane_col
        smem_col_full = warp_n * 32 + smem_col
        gr = block_m + smem_row
        gc = block_n + smem_col_full
        if (gr < M) and (gc < N):
            C_out[gr, gc] = al.convert(C_SMEM[smem_row, smem_col_full], al.bf16)


@avelang.jit
def activation_kernel(
    C_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    Y_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    tid = al.thread_id(0)
    row = al.block_id(0)
    bdim = al.block_dim(0)

    C_tensor = al.make_tensor(C_ptr, al.bf16, al.make_layout((M, N), (N, 1)))
    B_tensor = al.make_tensor(B_ptr, al.bf16, al.make_layout((N,), (1,)))
    Y_tensor = al.make_tensor(Y_ptr, al.bf16, al.make_layout((M, 1), (1, 1)))

    neg_inf = al.convert(-1.0e30, al.f32)
    local_max = neg_inf
    local_sum = al.convert(0.0, al.f32)

    for j in al.range(tid, N, bdim):
        val = al.convert(C_tensor[row, j], al.f32) + al.convert(B_tensor[j], al.f32)
        if val > local_max:
            local_max = val

    al.syncthreads()

    smem = al.make_shared((256,), al.f32)
    smem[tid] = local_max
    al.syncthreads()

    if tid < 128:
        other = smem[tid + 128]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()
    if tid < 64:
        other = smem[tid + 64]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()
    if tid < 32:
        other = smem[tid + 32]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()
    if tid < 16:
        other = smem[tid + 16]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()
    if tid < 8:
        other = smem[tid + 8]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()
    if tid < 4:
        other = smem[tid + 4]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()
    if tid < 2:
        other = smem[tid + 2]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()
    if tid < 1:
        other = smem[tid + 1]
        if other > smem[tid]:
            smem[tid] = other
    al.syncthreads()

    row_max = smem[0]
    al.syncthreads()

    for j in al.range(tid, N, bdim):
        val = al.convert(C_tensor[row, j], al.f32) + al.convert(B_tensor[j], al.f32)
        local_sum = local_sum + al.exp(val - row_max)

    al.syncthreads()

    smem[tid] = local_sum
    al.syncthreads()

    if tid < 128:
        smem[tid] = smem[tid] + smem[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem[tid] = smem[tid] + smem[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]
    al.syncthreads()

    row_sum = smem[0]
    al.syncthreads()

    if tid == 0:
        x = row_max + al.log(row_sum)
        zero_f32 = al.convert(0.0, al.f32)
        ns_f32 = al.convert(NEG_SLOPE, al.f32)
        if x < zero_f32:
            x = x * ns_f32
        if x < zero_f32:
            x = x * ns_f32
        half = al.convert(0.5, al.f32)
        one = al.convert(1.0, al.f32)
        sqrt2 = al.convert(SQRT_2, al.f32)
        x = half * x * (one + al.erf(x / sqrt2))
        x = half * x * (one + al.erf(x / sqrt2))
        Y_tensor[row, 0] = al.convert(x, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._w_cache = None
        self._b_cache = None
        self._w_ptr = None

    def forward(self, x):
        if x.dtype != torch.bfloat16:
            raise RuntimeError("This kernel only supports BF16 input.")

        dev = x.device
        M_int = x.shape[0]
        K_int = x.shape[1]
        N_int = self.linear.out_features

        w = self.linear.weight.data
        bias = self.linear.bias.data

        w_ptr = w.data_ptr()
        if self._w_cache is None or self._w_ptr != w_ptr:
            self._w_cache = w.to(device=dev, dtype=torch.bfloat16).contiguous()
            self._b_cache = bias.to(device=dev, dtype=torch.bfloat16).contiguous()
            self._w_ptr = w_ptr

        w_dev = self._w_cache
        b_dev = self._b_cache
        x = x.contiguous()

        c = torch.empty((M_int, N_int), device=dev, dtype=torch.bfloat16)
        grid_n = (N_int + BLOCK_N - 1) // BLOCK_N
        grid_m = (M_int + BLOCK_M - 1) // BLOCK_M
        gemm_kernel[lambda: ((grid_n, grid_m, 1), (THREADS, 1, 1))](
            x, w_dev, c, M_int, N_int, K_int,
        )

        y = torch.empty((M_int, 1), device=dev, dtype=torch.bfloat16)
        activation_kernel[lambda: ((M_int, 1, 1), (256, 1, 1))](
            c, b_dev, y, M_int, N_int,
        )

        return y
