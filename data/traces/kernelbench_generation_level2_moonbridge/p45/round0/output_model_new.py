import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Tile constants ──────────────────────────────────────────────────────────
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS  # 256
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
M_TILES_PER_WARP = GROUP_M // (WARPS_M * MMA_M)  # 128 // 64 = 2
N_TILES_PER_WARP = GROUP_N // (WARPS_N * MMA_N)  # 128 // 64 = 2
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS  # 2
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS  # 2
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW  # 256
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW  # 256
GLOBAL_LOADS_A = SHM_A_VECS // THREADS  # 1
GLOBAL_LOADS_B = SHM_B_VECS // THREADS  # 1
ROW_U32 = A_VECS_PER_ROW * 4  # 8

# ── LogSumExp constants ─────────────────────────────────────────────────────
LSE_BLOCK_SIZE: al.constexpr = 256


# ── Shared helper kernels ───────────────────────────────────────────────────


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
        idx += THREADS


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
        idx += THREADS


@avelang.jit
def _fetch_mfma_operand_32x32x16(
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


# ── Kernel 1: GEMM1 + Sigmoid ──────────────────────────────────────────────


@avelang.jit
def gemm1_sigmoid_kernel(
    x_ptr: al.Pointer(al.bf16),
    w1_ptr: al.Pointer(al.bf16),
    bias1_ptr: al.Pointer(al.bf16),
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
    w_memref = al.make_tensor(w1_ptr, al.bf16, al.make_layout((n * k,), (1,)))
    g_bias = al.make_tensor(bias1_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    x_rsrc = al.amdgpu.make_rsrc(x_memref, m * k * BF16_BYTES)
    w_rsrc = al.amdgpu.make_rsrc(w_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    a_reg = al.make_local((M_TILES_PER_WARP, 2, 4), al.bf16)
    b_reg = al.make_local((N_TILES_PER_WARP, 2, 4), al.bf16)
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
            _fetch_mfma_operand_32x32x16(a_reg[i], shm_a, warp_row * M_TILES_PER_WARP + i, lane)
        for j in al.range(N_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(b_reg[j], shm_b, warp_col * N_TILES_PER_WARP + j, lane)

        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                a0 = al.view(a_reg[i, 0], al.Tensor((2,), al.u32))
                b0 = al.view(b_reg[j, 0], al.Tensor((2,), al.u32))
                a1 = al.view(a_reg[i, 1], al.Tensor((2,), al.u32))
                b1 = al.view(b_reg[j, 1], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a0, b0, acc[acc_idx])
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a1, b1, acc[acc_idx])

        al.syncthreads()

    one = al.convert(1.0, al.f32)
    zero = al.convert(0.0, al.f32)
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
                neg_result = zero - result
                exp_neg = al.exp(neg_result)
                result = one / (one + exp_neg)
                g_out[row, col] = al.convert(result, al.bf16)


# ── Kernel 2: GEMM2 ────────────────────────────────────────────────────────


@avelang.jit
def gemm2_kernel(
    x_ptr: al.Pointer(al.bf16),
    w2_ptr: al.Pointer(al.bf16),
    bias2_ptr: al.Pointer(al.bf16),
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
    w_memref = al.make_tensor(w2_ptr, al.bf16, al.make_layout((n * k,), (1,)))
    g_bias = al.make_tensor(bias2_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    x_rsrc = al.amdgpu.make_rsrc(x_memref, m * k * BF16_BYTES)
    w_rsrc = al.amdgpu.make_rsrc(w_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    a_reg = al.make_local((M_TILES_PER_WARP, 2, 4), al.bf16)
    b_reg = al.make_local((N_TILES_PER_WARP, 2, 4), al.bf16)
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
            _fetch_mfma_operand_32x32x16(a_reg[i], shm_a, warp_row * M_TILES_PER_WARP + i, lane)
        for j in al.range(N_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(b_reg[j], shm_b, warp_col * N_TILES_PER_WARP + j, lane)

        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                a0 = al.view(a_reg[i, 0], al.Tensor((2,), al.u32))
                b0 = al.view(b_reg[j, 0], al.Tensor((2,), al.u32))
                a1 = al.view(a_reg[i, 1], al.Tensor((2,), al.u32))
                b1 = al.view(b_reg[j, 1], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a0, b0, acc[acc_idx])
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a1, b1, acc[acc_idx])

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


# ── Kernel 3: LogSumExp reduction over dim=1 ───────────────────────────────


@avelang.jit
def logsumexp_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    rows: al.u32,
    cols: al.u32,
):
    tid = al.thread_id(0)
    row = al.block_id(0)

    if row >= rows:
        return

    smem = al.make_shared((LSE_BLOCK_SIZE,), al.f32)

    layout_in = al.make_layout((rows, cols), (cols, 1))
    x = al.make_tensor(x_ptr, al.bf16, layout_in)

    neg_inf = al.convert(-3.4028234663852886e+38, al.f32)
    zero = al.convert(0.0, al.f32)

    # Pass 1: find per-row max
    local_max = neg_inf
    for c in al.range(tid, cols, LSE_BLOCK_SIZE):
        val = al.convert(x[row, c], al.f32)
        if val > local_max:
            local_max = val

    smem[tid] = local_max
    al.syncthreads()

    if tid < 128:
        a = smem[tid]
        b = smem[tid + 128]
        smem[tid] = a if a > b else b
    al.syncthreads()
    if tid < 64:
        a = smem[tid]
        b = smem[tid + 64]
        smem[tid] = a if a > b else b
    al.syncthreads()
    if tid < 32:
        a = smem[tid]
        b = smem[tid + 32]
        smem[tid] = a if a > b else b
    al.syncthreads()
    if tid < 16:
        a = smem[tid]
        b = smem[tid + 16]
        smem[tid] = a if a > b else b
    al.syncthreads()
    if tid < 8:
        a = smem[tid]
        b = smem[tid + 8]
        smem[tid] = a if a > b else b
    al.syncthreads()
    if tid < 4:
        a = smem[tid]
        b = smem[tid + 4]
        smem[tid] = a if a > b else b
    al.syncthreads()
    if tid < 2:
        a = smem[tid]
        b = smem[tid + 2]
        smem[tid] = a if a > b else b
    al.syncthreads()
    if tid < 1:
        a = smem[tid]
        b = smem[tid + 1]
        smem[tid] = a if a > b else b
    al.syncthreads()

    global_max = smem[0]

    # Pass 2: sum exp(x - max)
    local_sum = zero
    for c in al.range(tid, cols, LSE_BLOCK_SIZE):
        val = al.convert(x[row, c], al.f32)
        local_sum = local_sum + al.exp(val - global_max)

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

    global_sum = smem[0]
    result = al.log(global_sum) + global_max

    layout_out = al.make_layout((rows,), (1,))
    ot = al.make_tensor(out_ptr, al.bf16, layout_out)
    ot[row] = al.convert(result, al.bf16)


# ── Host wrappers ───────────────────────────────────────────────────────────


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_gemm1_sigmoid(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)
    b_bf16 = _prepare_bf16_cuda_contiguous(bias)

    m, k = x_bf16.shape
    n, wk = w_bf16.shape

    out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (n // GROUP_N, m // GROUP_M, 1)
    gemm1_sigmoid_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, b_bf16, out, m, n, k
    )
    return out


def avelang_gemm2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)
    b_bf16 = _prepare_bf16_cuda_contiguous(bias)

    m, k = x_bf16.shape
    n, wk = w_bf16.shape

    out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (n // GROUP_N, m // GROUP_M, 1)
    gemm2_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, b_bf16, out, m, n, k
    )
    return out


def avelang_logsumexp(
    x: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)

    rows, cols = x_bf16.shape
    out = torch.empty((rows,), device=x_bf16.device, dtype=torch.bfloat16)

    logsumexp_kernel[lambda: ((rows, 1, 1), (LSE_BLOCK_SIZE, 1, 1))](
        x_bf16, out, rows, cols
    )
    return out


# ── ModelNew ────────────────────────────────────────────────────────────────

batch_size = 16384
input_size = 2048
hidden_size = 4096
output_size = 1024


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weight1 = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias1 = nn.Parameter(torch.empty(hidden_size))
        self.weight2 = nn.Parameter(torch.empty(output_size, hidden_size))
        self.bias2 = nn.Parameter(torch.empty(output_size))

        nn.init.kaiming_uniform_(self.weight1, a=5 ** 0.5)
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
        bound1 = 1 / (fan_in1 ** 0.5) if fan_in1 > 0 else 0
        nn.init.uniform_(self.bias1, -bound1, bound1)

        nn.init.kaiming_uniform_(self.weight2, a=5 ** 0.5)
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
        bound2 = 1 / (fan_in2 ** 0.5) if fan_in2 > 0 else 0
        nn.init.uniform_(self.bias2, -bound2, bound2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = avelang_gemm1_sigmoid(x, self.weight1, self.bias1)
        x = avelang_gemm2(x, self.weight2, self.bias2)
        x = avelang_logsumexp(x)
        return x.to(orig_dtype)


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, output_size]
