import torch
import torch.nn as nn
import math
import avelang
import avelang.language as al

# =============================================================================
# GEMM tile constants (following the verified MFMA GEMM pattern)
# =============================================================================
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

# im2col block size
IM2COL_BLOCK = 256


# =============================================================================
# GEMM helper kernels (from verified MFMA GEMM pattern)
# =============================================================================

@avelang.jit
def _load_global_a_to_shm(
    shm_a: al.Tensor((SHM_A_VECS, 4), al.u32),
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
        idx += THREADS


@avelang.jit
def _load_global_b_to_shm(
    shm_b: al.Tensor((SHM_B_VECS, 4), al.u32),
    b_rsrc: al.Tensor((4,), al.u32),
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
        shm_b[idx] = al.amdgpu.raw_buffer_load_x4(b_rsrc, zero, off, 0)
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


# =============================================================================
# im2col kernel: converts (N, IC, H, W) to (M, K) column matrix
# Processes rows [m_offset, m_offset + M_chunk) of the im2col output.
# =============================================================================

@avelang.jit
def im2col_bf16_kernel(
    input_ptr: al.Pointer(al.bf16),
    col_ptr: al.Pointer(al.bf16),
    m_offset: al.i32,
    N: al.i32,
    IC: al.i32,
    H: al.i32,
    W: al.i32,
    KH: al.i32,
    KW: al.i32,
    OH: al.i32,
    OW: al.i32,
    M_chunk: al.i32,
    K: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    m_global = m_offset + bid * IM2COL_BLOCK + tid
    m_end = m_offset + M_chunk
    if m_global < m_end:
        m_local = m_global - m_offset

        # Decode global output pixel position
        oh_ow = OH * OW
        n = m_global // oh_ow
        resid = m_global - n * oh_ow
        h_out = resid // OW
        w_out = resid - h_out * OW

        # Build 4D input view and 2D column view
        inp = al.make_tensor(
            input_ptr, al.bf16,
            al.make_layout((N, IC, H, W), (IC * H * W, H * W, W, 1)),
        )
        col = al.make_tensor(
            col_ptr, al.bf16,
            al.make_layout((M_chunk, K), (K, 1)),
        )

        khkw = KH * KW

        for k_idx in al.range(K):
            ic = k_idx // khkw
            resid_k = k_idx - ic * khkw
            kh = resid_k // KW
            kw = resid_k - kh * KW

            h_in = h_out + kh
            w_in = w_out + kw

            col[m_local, k_idx] = inp[n, ic, h_in, w_in]


# =============================================================================
# MFMA GEMM kernel: C = A x B^T  where A:(M,K), B:(N,K)
# All layouts use 2D to keep each dimension within 32-bit range.
# The A-buffer byte-range (M * K * 2) must be <= 2^32 - 1.
# =============================================================================

@avelang.jit
def gemm_bf16_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
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

    a_memref = al.make_tensor(a_ptr, al.bf16, al.make_layout((m, k), (k, 1)))
    b_memref = al.make_tensor(b_ptr, al.bf16, al.make_layout((n, k), (k, 1)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    a_rsrc = al.amdgpu.make_rsrc(a_memref, m * k * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    a_reg = al.make_local((M_TILES_PER_WARP, 2, 4), al.bf16)
    b_reg = al.make_local((N_TILES_PER_WARP, 2, 4), al.bf16)
    acc = al.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), al.f32)

    # Zero accumulators
    for i in al.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for j in al.range(ACC_SIZE):
            acc[i, j] = al.convert(0.0, al.f32)

    k_tiles = k // GROUP_K
    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K
        _load_global_a_to_shm(shm_a, a_rsrc, block_m, k_base, k, tid)
        _load_global_b_to_shm(shm_b, b_rsrc, block_n, k_base, k, tid)
        al.syncthreads()

        for i in al.range(M_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(
                a_reg[i], shm_a, warp_row * M_TILES_PER_WARP + i, lane
            )
        for j in al.range(N_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(
                b_reg[j], shm_b, warp_col * N_TILES_PER_WARP + j, lane
            )

        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(
                    al.view(a_reg[i, 0], al.Tensor((2,), al.u32)),
                    al.view(b_reg[j, 0], al.Tensor((2,), al.u32)),
                    acc[acc_idx],
                )
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(
                    al.view(a_reg[i, 1], al.Tensor((2,), al.u32)),
                    al.view(b_reg[j, 1], al.Tensor((2,), al.u32)),
                    acc[acc_idx],
                )

        al.syncthreads()

    # Writeback
    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    for j in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        for i in al.range(M_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            for t in al.range(ACC_SIZE):
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                g_out[row, col] = al.convert(acc[acc_idx, t], al.bf16)


# =============================================================================
# Host wrapper
# =============================================================================

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    BF16 2D convolution via chunked im2col + MFMA GEMM.

    The M dimension (N * OH * OW) is processed in chunks so each
    intermediate im2col buffer stays under the 32-bit raw-buffer
    resource limit (~4 GB).

    x:      (N, IC, H, W)
    weight: (OC, IC, KH, KW)
    Returns: (N, OC, OH, OW)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)

    N, IC, H, W = x_bf16.shape
    OC, w_IC, KH, KW = w_bf16.shape

    OH = H - KH + 1
    OW = W - KW + 1
    M_total = N * OH * OW
    K = IC * KH * KW

    w_reshaped = w_bf16.reshape(OC, K).contiguous()

    # Max M per chunk: col bytes = M_chunk * K * 2 must be < 2^32
    max_chunk_bytes = (1 << 32) - 1
    max_chunk_m = (max_chunk_bytes // (K * BF16_BYTES) // GROUP_M) * GROUP_M
    if max_chunk_m < GROUP_M:
        raise RuntimeError("K too large to fit in any chunk")

    # Allocate output buffer
    out = torch.empty((N, OC, OH, OW), device=x_bf16.device, dtype=torch.bfloat16)
    out_flat = out.permute(0, 2, 3, 1).reshape(M_total, OC)

    chunk_start = 0
    while chunk_start < M_total:
        chunk_end = min(chunk_start + max_chunk_m, M_total)
        M_chunk = chunk_end - chunk_start
        M_chunk_padded = ((M_chunk + GROUP_M - 1) // GROUP_M) * GROUP_M

        # Step 1: im2col for rows [chunk_start, chunk_end)
        col = torch.empty(
            (M_chunk_padded, K), device=x_bf16.device, dtype=torch.bfloat16
        )
        im2col_grid = ((M_chunk + IM2COL_BLOCK - 1) // IM2COL_BLOCK, 1, 1)
        im2col_bf16_kernel[lambda: (im2col_grid, (IM2COL_BLOCK, 1, 1))](
            x_bf16, col, chunk_start, N, IC, H, W, KH, KW, OH, OW, M_chunk, K
        )

        # Step 2: GEMM
        M_tiles = M_chunk_padded // GROUP_M
        N_tiles = OC // GROUP_N
        out_gemm = torch.empty(
            (M_chunk_padded, OC), device=x_bf16.device, dtype=torch.bfloat16
        )
        gemm_grid = (N_tiles, M_tiles, 1)
        gemm_bf16_kernel[lambda: (gemm_grid, (THREADS, 1, 1))](
            col, w_reshaped, out_gemm, M_chunk_padded, OC, K
        )

        # Step 3: Copy valid rows to output
        out_flat[chunk_start:chunk_end, :] = out_gemm[:M_chunk, :]

        chunk_start = chunk_end

    return out


# =============================================================================
# ModelNew entrypoint
# =============================================================================

class ModelNew(nn.Module):
    """
    Optimized 2D convolution using AveLang DSL with im2col + MFMA GEMM.
    Matches nn.Conv2d semantics with stride=1, padding=0, dilation=1,
    groups=1, bias=False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv2d(x, self.conv2d.weight.data)


# =============================================================================
# Test helpers
# =============================================================================

batch_size = 16
in_channels = 16
out_channels = 128
kernel_size = 3
width = 1024
height = 1024


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
