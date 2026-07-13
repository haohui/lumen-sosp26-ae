import torch
import torch.nn as nn
import avelang
import avelang.language as al

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NUM_GROUPS = 16
HARDTANH_MIN = -2.0
HARDTANH_MAX = 2.0
EPS = 1e-05


WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
GROUP_M = 128
GROUP_N = 128
GROUP_K = 16
MMA_M = 32
MMA_N = 32
VEC_ELEMS = 8
BF16_BYTES = 2
ACC_SIZE = 16
WARPS_M = 2
WARPS_N = 2
M_TILES_PER_WARP = GROUP_M // (WARPS_M * MMA_M)
N_TILES_PER_WARP = GROUP_N // (WARPS_N * MMA_N)
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW
GLOBAL_LOADS_A = SHM_A_VECS // THREADS
GLOBAL_LOADS_B = SHM_B_VECS // THREADS
ROW_U32 = A_VECS_PER_ROW * 4

GN_THREADS = 256


@avelang.jit
def _load_global_a_to_shm(
    shm_a: al.Tensor((SHM_A_VECS, 4), al.u32),
    x_rsrc: al.Tensor((4,), al.u32),
    block_m: al.u32,
    k_base: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_A):
        row = idx // A_VECS_PER_ROW
        col_vec = idx % A_VECS_PER_ROW
        off = ((block_m * GROUP_M + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_a[idx] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero, off, 0)
        idx = idx + THREADS


@avelang.jit
def _load_global_b_to_shm(
    shm_b: al.Tensor((SHM_B_VECS, 4), al.u32),
    w_rsrc: al.Tensor((4,), al.u32),
    block_n: al.u32,
    k_base: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_B):
        row = idx // B_VECS_PER_ROW
        col_vec = idx % B_VECS_PER_ROW
        off = ((block_n * GROUP_N + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_b[idx] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, off, 0)
        idx = idx + THREADS


@avelang.jit
def _fetch_mfma_operand_32x32x16_i32(
    ret_i32: al.Tensor((2, 2), al.i32),
    shm: al.Tensor((SHM_A_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
):
    shm_u32 = al.view(shm, al.Tensor((SHM_A_VECS * 4,), al.u32))
    ret_flat = al.view(ret_i32, al.Tensor((4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32

    ret_flat[0] = shm_u32[row_base + k_group_u32]
    ret_flat[1] = shm_u32[row_base + k_group_u32 + 1]
    ret_flat[2] = shm_u32[row_base + 4 + k_group_u32]
    ret_flat[3] = shm_u32[row_base + 5 + k_group_u32]


@avelang.jit
def gemm_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.bf16),
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
    g_out = al.make_tensor(y_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    x_rsrc = al.amdgpu.make_rsrc(x_memref, m * k * BF16_BYTES)
    w_rsrc = al.amdgpu.make_rsrc(w_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    a_reg = al.make_local((M_TILES_PER_WARP, 2, 2), al.i32)
    b_reg = al.make_local((N_TILES_PER_WARP, 2, 2), al.i32)
    acc = al.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), al.f32)

    for i in al.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for j in al.range(ACC_SIZE):
            acc[i, j] = 0

    k_tiles = k // GROUP_K
    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K
        _load_global_a_to_shm(shm_a, x_rsrc, block_m, k_base, k, tid)
        _load_global_b_to_shm(shm_b, w_rsrc, block_n, k_base, k, tid)
        al.syncthreads()

        for i in al.range(M_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16_i32(a_reg[i], shm_a, warp_row * M_TILES_PER_WARP + i, lane)
        for j in al.range(N_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16_i32(b_reg[j], shm_b, warp_col * N_TILES_PER_WARP + j, lane)

        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_reg[i, 0], b_reg[j, 0], acc[acc_idx])
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_reg[i, 1], b_reg[j, 1], acc[acc_idx])

        al.syncthreads()

    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    for j in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        bias = al.convert(g_bias[col], al.f32)
        for i in al.range(M_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            for t in al.range(ACC_SIZE):
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                result = acc[acc_idx, t] + bias
                g_out[row, col] = al.convert(result, al.bf16)


@avelang.jit
def groupnorm_hardtanh_kernel(
    y_ptr: al.Pointer(al.bf16),
    gn_weight_ptr: al.Pointer(al.bf16),
    gn_bias_ptr: al.Pointer(al.bf16),
    m: al.i32,
    c: al.i32,
    g: al.i32,
):
    tid = al.thread_id(0)
    sample_idx = al.block_id(0)
    group_idx = al.block_id(1)
    block_dim = al.block_dim(0)

    eps = al.convert(EPS, al.f32)
    hardtanh_min = al.convert(HARDTANH_MIN, al.f32)
    hardtanh_max = al.convert(HARDTANH_MAX, al.f32)

    group_size = c // g
    start_c = group_idx * group_size

    y_layout = al.make_layout((m, c), (c, 1))
    y = al.make_tensor(y_ptr, al.bf16, y_layout)
    gw_layout = al.make_layout((c,), (1,))
    gw = al.make_tensor(gn_weight_ptr, al.bf16, gw_layout)
    gb = al.make_tensor(gn_bias_ptr, al.bf16, gw_layout)

    smem = al.make_shared((GN_THREADS,), al.f32)

    partial = al.convert(0.0, al.f32)
    for idx in al.range(tid, group_size, block_dim):
        cc = start_c + idx
        partial = partial + al.convert(y[sample_idx, cc], al.f32)
    smem[tid] = partial
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
        smem[0] = smem[0] + smem[1]
    al.syncthreads()

    mean = smem[0] / al.convert(group_size, al.f32)

    partial_var = al.convert(0.0, al.f32)
    for idx in al.range(tid, group_size, block_dim):
        cc = start_c + idx
        d = al.convert(y[sample_idx, cc], al.f32) - mean
        partial_var = partial_var + d * d
    smem[tid] = partial_var
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
        smem[0] = smem[0] + smem[1]
    al.syncthreads()

    var = smem[0] / al.convert(group_size, al.f32)
    denom = al.sqrt(var + eps)

    smem[0] = mean
    smem[1] = denom
    al.syncthreads()

    l_mean = smem[0]
    l_denom = smem[1]

    for idx in al.range(tid, group_size, block_dim):
        cc = start_c + idx
        v = (al.convert(y[sample_idx, cc], al.f32) - l_mean) / l_denom
        v = v * al.convert(gw[cc], al.f32) + al.convert(gb[cc], al.f32)
        if v < hardtanh_min:
            v = hardtanh_min
        if v > hardtanh_max:
            v = hardtanh_max
        y[sample_idx, cc] = al.convert(v, al.bf16)


class ModelNew(nn.Module):

    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != 'cuda':
            x = x.cuda()
        x_bf16 = x.contiguous().to(dtype=torch.bfloat16)

        m, k = x_bf16.shape

        weight = self.gemm.weight.contiguous().to(dtype=torch.bfloat16, device=x_bf16.device)
        bias = self.gemm.bias.contiguous().to(dtype=torch.bfloat16, device=x_bf16.device)
        gn_w = self.group_norm.weight.contiguous().to(dtype=torch.bfloat16, device=x_bf16.device)
        gn_b = self.group_norm.bias.contiguous().to(dtype=torch.bfloat16, device=x_bf16.device)

        n = weight.shape[0]

        y = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)

        grid_gemm = (n // GROUP_N, m // GROUP_M, 1)
        gemm_kernel[lambda: (grid_gemm, (THREADS, 1, 1))](
            x_bf16, weight, bias, y, m, n, k
        )

        grid_gn = (m, self.group_norm.num_groups, 1)
        groupnorm_hardtanh_kernel[lambda: (grid_gn, (GN_THREADS, 1, 1))](
            y, gn_w, gn_b, m, n, self.group_norm.num_groups,
        )

        return y
