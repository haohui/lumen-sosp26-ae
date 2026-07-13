import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Problem dimensions ──────────────────────────────────────────────
BATCH_SIZE = 128
IN_CHANNELS = 8
OUT_CHANNELS = 64
HEIGHT = 256
WIDTH = 256
KERNEL_H = 3
KERNEL_W = 3
H_OUT = HEIGHT - KERNEL_H + 1   # 254
W_OUT = WIDTH - KERNEL_W + 1    # 254
K_ORIG = IN_CHANNELS * KERNEL_H * KERNEL_W  # 72
K_PAD = 80  # next multiple of 16 for clean MFMA tiling
SPATIAL_SIZE = H_OUT * W_OUT    # 64516

# ── GEMM tiling constants ───────────────────────────────────────────
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS   # 256
GROUP_M = 128
GROUP_N = 64
GROUP_K = 16
MMA_M = 32
MMA_N = 32
VEC_ELEMS = 8
BF16_BYTES = 2
ACC_SIZE = 16
WARPS_M = 2
WARPS_N = 2
M_TILES_PER_WARP = GROUP_M // (WARPS_M * MMA_M)  # 2
N_TILES_PER_WARP = GROUP_N // (WARPS_N * MMA_N)  # 1
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS             # 2
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS             # 2
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW             # 256
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW             # 128
SHM_VECS = SHM_A_VECS                             # 256
GLOBAL_LOADS_A = (SHM_A_VECS + THREADS - 1) // THREADS  # 1
GLOBAL_LOADS_B = (SHM_B_VECS + THREADS - 1) // THREADS  # 1
ROW_U32 = A_VECS_PER_ROW * 4                      # 8

# ── Im2Col constants ─────────────────────────────────────────────────
POOL_THREADS = 256
IM2COL_THREADS = 256


# ══════════════════════════════════════════════════════════════════════
# Kernel 1: Im2Col
# ══════════════════════════════════════════════════════════════════════

@avelang.jit
def im2col_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.i32,
    KW: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K_total: al.i32,
    K_orig: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    row = bid * IM2COL_THREADS + tid
    M_total = N * H_out * W_out

    if row < M_total:
        input_t = al.make_tensor(
            input_ptr, al.bf16,
            al.make_layout((N, C, H, W), (C * H * W, H * W, W, 1)),
        )
        output_t = al.make_tensor(
            output_ptr, al.bf16,
            al.make_layout((M_total, K_total), (K_total, 1)),
        )

        batch = row // (H_out * W_out)
        rem = row - batch * (H_out * W_out)
        h_out = rem // W_out
        w_out = rem - h_out * W_out

        for c in al.range(C):
            for kh in al.range(KH):
                for kw in al.range(KW):
                    k_idx = c * KH * KW + kh * KW + kw
                    src_h = h_out + kh
                    src_w = w_out + kw
                    output_t[row, k_idx] = input_t[batch, c, src_h, src_w]

        for k in al.range(K_orig, K_total):
            output_t[row, k] = al.convert(0.0, al.bf16)


def _run_im2col(x_bf16: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x_bf16.shape
    H_out = H - KERNEL_H + 1
    W_out = W - KERNEL_W + 1
    M_total = N * H_out * W_out

    out = torch.empty((M_total, K_PAD), device=x_bf16.device, dtype=torch.bfloat16)
    grid = ((M_total + IM2COL_THREADS - 1) // IM2COL_THREADS, 1, 1)
    im2col_kernel[lambda: (grid, (IM2COL_THREADS, 1, 1))](
        x_bf16, out,
        N, C, H, W, KERNEL_H, KERNEL_W,
        H_out, W_out, K_PAD, K_ORIG,
    )
    return out


# ══════════════════════════════════════════════════════════════════════
# Kernel 2: MFMA GEMM + GELU
# ══════════════════════════════════════════════════════════════════════

@avelang.jit
def _load_a_to_shm(
    shm_a: al.Tensor((SHM_VECS, 4), al.u32),
    a_rsrc: al.Tensor((4,), al.u32),
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
        shm_a[idx] = al.amdgpu.raw_buffer_load_x4(a_rsrc, zero, off, 0)
        idx = idx + THREADS


@avelang.jit
def _load_b_to_shm(
    shm_b: al.Tensor((SHM_VECS, 4), al.u32),
    b_rsrc: al.Tensor((4,), al.u32),
    block_n: al.u32,
    k_base: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_B):
        if idx < SHM_B_VECS:
            row = idx // B_VECS_PER_ROW
            col_vec = idx % B_VECS_PER_ROW
            off = ((block_n * GROUP_N + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
            shm_b[idx] = al.amdgpu.raw_buffer_load_x4(b_rsrc, zero, off, 0)
        idx = idx + THREADS


@avelang.jit
def _fetch_operand_k0(
    ret: al.Tensor((2,), al.u32),
    shm: al.Tensor((SHM_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
):
    shm_u32 = al.view(shm, al.Tensor((SHM_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32
    ret[0] = shm_u32[row_base + k_group_u32]
    ret[1] = shm_u32[row_base + k_group_u32 + 1]


@avelang.jit
def _fetch_operand_k1(
    ret: al.Tensor((2,), al.u32),
    shm: al.Tensor((SHM_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
):
    shm_u32 = al.view(shm, al.Tensor((SHM_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32 + 4
    ret[0] = shm_u32[row_base + k_group_u32]
    ret[1] = shm_u32[row_base + k_group_u32 + 1]


@avelang.jit
def gemm_gelu_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
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

    a_memref = al.make_tensor(a_ptr, al.bf16, al.make_layout((m * k,), (1,)))
    b_memref = al.make_tensor(b_ptr, al.bf16, al.make_layout((n * k,), (1,)))
    g_bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    a_rsrc = al.amdgpu.make_rsrc(a_memref, m * k * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_VECS, 4), al.u32)
    a_frag = al.make_local((M_TILES_PER_WARP, 2, 2), al.u32)
    b_frag = al.make_local((N_TILES_PER_WARP, 2, 2), al.u32)
    acc = al.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), al.f32)

    for i in al.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for j in al.range(ACC_SIZE):
            acc[i, j] = 0.0

    k_tiles = k // GROUP_K
    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K
        _load_a_to_shm(shm_a, a_rsrc, block_m, k_base, k, tid)
        _load_b_to_shm(shm_b, b_rsrc, block_n, k_base, k, tid)
        al.syncthreads()

        for i in al.range(M_TILES_PER_WARP):
            _fetch_operand_k0(a_frag[i, 0], shm_a, warp_row * M_TILES_PER_WARP + i, lane)
            _fetch_operand_k1(a_frag[i, 1], shm_a, warp_row * M_TILES_PER_WARP + i, lane)
        for jj in al.range(N_TILES_PER_WARP):
            _fetch_operand_k0(b_frag[jj, 0], shm_b, warp_col * N_TILES_PER_WARP + jj, lane)
            _fetch_operand_k1(b_frag[jj, 1], shm_b, warp_col * N_TILES_PER_WARP + jj, lane)

        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(
                    a_frag[i, 0], b_frag[j, 0], acc[acc_idx])
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(
                    a_frag[i, 1], b_frag[j, 1], acc[acc_idx])

        al.syncthreads()

    sqrt_2_pi = al.convert(0.7978845608028654, al.f32)
    gelu_coeff = al.convert(0.044715, al.f32)
    half = al.convert(0.5, al.f32)
    one = al.convert(1.0, al.f32)

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
                x = result
                x_cube = x * x * x
                inner = sqrt_2_pi * (x + gelu_coeff * x_cube)
                gelu = half * x * (one + al.tanh(inner))
                g_out[row, col] = al.convert(gelu, al.bf16)


def _run_gemm_gelu(
    a_bf16: torch.Tensor,
    weight_bf16: torch.Tensor,
    bias_bf16: torch.Tensor,
) -> torch.Tensor:
    m, k = a_bf16.shape
    n, wk = weight_bf16.shape
    if wk != k:
        raise ValueError(f"K mismatch: A has K={k}, weight has K={wk}")

    out = torch.empty((m, n), device=a_bf16.device, dtype=torch.bfloat16)
    grid = (n // GROUP_N, m // GROUP_M, 1)
    gemm_gelu_kernel[lambda: (grid, (THREADS, 1, 1))](
        a_bf16, weight_bf16, bias_bf16, out,
        m, n, k,
    )
    return out


# ══════════════════════════════════════════════════════════════════════
# Kernel 3: Spatial average pooling
# ══════════════════════════════════════════════════════════════════════

@avelang.jit
def spatial_pool_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    spatial_size: al.i32,
    channels: al.i32,
):
    tid = al.thread_id(0)
    block_c = al.block_id(0)
    block_b = al.block_id(1)

    input_t = al.make_tensor(
        input_ptr, al.bf16,
        al.make_layout(
            (batch_size * spatial_size, channels),
            (channels, 1),
        ),
    )

    sum_val = al.convert(0.0, al.f32)
    row_start = block_b * spatial_size
    for pos in al.range(tid, spatial_size, POOL_THREADS):
        row = row_start + pos
        val = al.convert(input_t[row, block_c], al.f32)
        sum_val = sum_val + val

    shm = al.make_shared((POOL_THREADS,), al.f32)
    shm[tid] = sum_val
    al.syncthreads()

    if tid < 128:
        shm[tid] = shm[tid] + shm[tid + 128]
    al.syncthreads()
    if tid < 64:
        shm[tid] = shm[tid] + shm[tid + 64]
    al.syncthreads()
    if tid < 32:
        shm[tid] = shm[tid] + shm[tid + 32]
    al.syncthreads()
    if tid < 16:
        shm[tid] = shm[tid] + shm[tid + 16]
    al.syncthreads()
    if tid < 8:
        shm[tid] = shm[tid] + shm[tid + 8]
    al.syncthreads()
    if tid < 4:
        shm[tid] = shm[tid] + shm[tid + 4]
    al.syncthreads()
    if tid < 2:
        shm[tid] = shm[tid] + shm[tid + 2]
    al.syncthreads()
    if tid == 0:
        shm[0] = shm[0] + shm[1]
    al.syncthreads()

    if tid == 0:
        avg = shm[0] / al.convert(spatial_size, al.f32)
        output_t = al.make_tensor(
            output_ptr, al.bf16,
            al.make_layout((batch_size, channels), (channels, 1)),
        )
        output_t[block_b, block_c] = al.convert(avg, al.bf16)


def _run_spatial_pool(
    x_bf16: torch.Tensor,
    batch_size: int,
    spatial_size: int,
    channels: int,
) -> torch.Tensor:
    out = torch.empty((batch_size, channels), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (channels, batch_size, 1)
    spatial_pool_kernel[lambda: (grid, (POOL_THREADS, 1, 1))](
        x_bf16, out, batch_size, spatial_size, channels,
    )
    return out


# ══════════════════════════════════════════════════════════════════════
# Host utility
# ══════════════════════════════════════════════════════════════════════

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


# ══════════════════════════════════════════════════════════════════════
# ModelNew
# ══════════════════════════════════════════════════════════════════════

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device required.")

        x_bf16 = _prepare_bf16_cuda_contiguous(x)

        # Step 1: im2col
        im2col_bf16 = _run_im2col(x_bf16)

        # Step 2: prepare weight + bias, GEMM + GELU
        weight_2d = self.weight.data.reshape(self.out_channels, -1)
        weight_padded = torch.nn.functional.pad(weight_2d, (0, K_PAD - K_ORIG))
        weight_bf16 = _prepare_bf16_cuda_contiguous(weight_padded)
        bias_bf16 = _prepare_bf16_cuda_contiguous(self.bias.data)

        gemm_out_bf16 = _run_gemm_gelu(im2col_bf16, weight_bf16, bias_bf16)

        # Step 3: spatial average pooling -> (BATCH_SIZE, OUT_CHANNELS)
        out_bf16 = _run_spatial_pool(
            gemm_out_bf16, BATCH_SIZE, SPATIAL_SIZE, self.out_channels,
        )

        return out_bf16


# ══════════════════════════════════════════════════════════════════════
# Input contract
# ══════════════════════════════════════════════════════════════════════

def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_H]
