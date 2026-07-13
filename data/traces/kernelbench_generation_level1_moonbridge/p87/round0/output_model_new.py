import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Tile geometry ──────────────────────────────────────────────────────────
WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
GROUP_M = 256
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
SHM_MAX_VECS = SHM_A_VECS if SHM_A_VECS > SHM_B_VECS else SHM_B_VECS
GLOBAL_LOADS_A = SHM_A_VECS // THREADS
GLOBAL_LOADS_B = SHM_B_VECS // THREADS
ROW_U32 = A_VECS_PER_ROW * 4


@avelang.jit
def _load_global_a_to_shm(
    shm_a: al.Tensor((SHM_MAX_VECS, 4), al.u32),
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
    shm_b: al.Tensor((SHM_MAX_VECS, 4), al.u32),
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
def _fetch_mfma_operand_u32(
    dst: al.Tensor((4,), al.u32),
    shm: al.Tensor((SHM_MAX_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
):
    shm_u32 = al.view(shm, al.Tensor((SHM_MAX_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32

    dst[0] = shm_u32[row_base + k_group_u32]
    dst[1] = shm_u32[row_base + k_group_u32 + 1]
    dst[2] = shm_u32[row_base + 4 + k_group_u32]
    dst[3] = shm_u32[row_base + 5 + k_group_u32]


@avelang.jit
def conv1x1_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
    m_tiles_total: al.u32,
):
    tid = al.thread_id(0)
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

    shm_a = al.make_shared((SHM_MAX_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_MAX_VECS, 4), al.u32)
    a_reg = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    b_reg = al.make_local((N_TILES_PER_WARP, 4), al.u32)
    acc = al.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), al.f32)

    stride_m = al.grid_dim(0)
    block_n = al.convert(0, al.u32)
    start_block_m = al.block_id(0)

    for block_m in al.range(start_block_m, m_tiles_total, stride_m):
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
                _fetch_mfma_operand_u32(a_reg[i], shm_a, warp_row * M_TILES_PER_WARP + i, lane)
            for j in al.range(N_TILES_PER_WARP):
                _fetch_mfma_operand_u32(b_reg[j], shm_b, warp_col * N_TILES_PER_WARP + j, lane)

            for i in al.range(M_TILES_PER_WARP):
                for j in al.range(N_TILES_PER_WARP):
                    acc_idx = i * N_TILES_PER_WARP + j
                    a_op = al.view(a_reg[i], al.Tensor((2, 2), al.u32))
                    b_op = al.view(b_reg[j], al.Tensor((2, 2), al.u32))
                    acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_op[0], b_op[0], acc[acc_idx])
                    acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_op[1], b_op[1], acc[acc_idx])

            al.syncthreads()

        lane_group = lane // MMA_N
        lane_col = lane % MMA_N
        block_row_base = block_m * GROUP_M
        block_col_base = block_n * GROUP_N

        for j in al.range(N_TILES_PER_WARP):
            col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
            bias_f32 = al.convert(g_bias[col], al.f32)
            for i in al.range(M_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
                for t in al.range(ACC_SIZE):
                    row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                    val = acc[acc_idx, t] + bias_f32
                    g_out[row, col] = al.convert(val, al.bf16)


# ── Host wrapper ────────────────────────────────────────────────────────────

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv1x1(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    input_dtype = x.dtype

    N, C, H, W = x.shape
    C_out = weight.shape[0]

    x_2d = x.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
    x_bf16 = _prepare_bf16_cuda_contiguous(x_2d)

    w_squeezed = weight.data.reshape(C_out, C).contiguous()
    w_bf16 = _prepare_bf16_cuda_contiguous(w_squeezed)

    if bias is not None:
        bias_bf16 = _prepare_bf16_cuda_contiguous(bias.data.reshape(-1))
    else:
        bias_bf16 = torch.zeros((C_out,), device=x_bf16.device, dtype=torch.bfloat16)

    M = N * H * W
    K = C
    N_out = C_out

    out_bf16 = torch.empty((M, N_out), device=x_bf16.device, dtype=torch.bfloat16)

    m_tiles = M // GROUP_M
    num_blocks = min(m_tiles, 304)
    grid = (num_blocks, 1, 1)

    conv1x1_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, bias_bf16, out_bf16, M, N_out, K, m_tiles
    )

    out_4d = out_bf16.reshape(N, H, W, N_out).permute(0, 3, 1, 2).contiguous()

    if input_dtype != torch.bfloat16:
        out_4d = out_4d.to(dtype=input_dtype)

    return out_4d


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv1x1(x, self.conv1d.weight, self.conv1d.bias)


batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]
