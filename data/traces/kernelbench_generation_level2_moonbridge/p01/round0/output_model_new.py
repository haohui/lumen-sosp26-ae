import torch
import torch.nn as nn
import avelang
import avelang.language as al


BATCH_SIZE = 128
IN_CHANNELS = 64
OUT_CHANNELS = 128
HEIGHT = 128
WIDTH = 128
KERNEL_SIZE = 3
BIAS_SHAPE = (OUT_CHANNELS, 1, 1)

OUT_HEIGHT = HEIGHT - KERNEL_SIZE + 1
OUT_WIDTH = WIDTH - KERNEL_SIZE + 1
K_TOTAL = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE
M_TOTAL = BATCH_SIZE * OUT_HEIGHT * OUT_WIDTH

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

IM2COL_BLOCK = 256
IM2COL_ROWS_PER_BLOCK = 8


@avelang.jit
def im2col_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.u32,
    C: al.u32,
    H: al.u32,
    W: al.u32,
    OH: al.u32,
    OW: al.u32,
    KH: al.u32,
    KW: al.u32,
    K_total: al.u32,
    M_total: al.u32,
):
    tid = al.thread_id(0)
    block_idx = al.block_id(0)

    input_flat = al.make_tensor(input_ptr, al.bf16, al.make_layout((N * C * H * W,), (1,)))
    output_mat = al.make_tensor(output_ptr, al.bf16, al.make_layout((M_total, K_total), (K_total, 1)))

    C_H_W = C * H * W
    H_W = H * W
    kh_kw = KH * KW

    base_m = block_idx * al.convert(IM2COL_ROWS_PER_BLOCK, al.u32)

    for row_off in al.range(al.convert(IM2COL_ROWS_PER_BLOCK, al.u32)):
        m = base_m + row_off
        if m < M_total:
            n = m // (OH * OW)
            rest = m % (OH * OW)
            oh = rest // OW
            ow = rest % OW

            input_base = n * C_H_W + oh * W + ow

            for iter_idx in al.range(al.convert(3, al.u32)):
                k = tid + iter_idx * al.convert(IM2COL_BLOCK, al.u32)
                if k < K_total:
                    c = k // kh_kw
                    rest_k = k % kh_kw
                    kh = rest_k // KW
                    kw = rest_k % KW

                    input_idx = input_base + c * H_W + kh * W + kw
                    output_mat[m, k] = input_flat[input_idx]


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


@avelang.jit
def conv_relu_bias_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    extra_bias_ptr: al.Pointer(al.bf16),
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
    g_conv_bias = al.make_tensor(conv_bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_extra_bias = al.make_tensor(extra_bias_ptr, al.bf16, al.make_layout((n,), (1,)))
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
                a_full = al.view(a_reg[i], al.Tensor((4,), al.u32))
                a_pair = al.view(a_full, al.Tensor((2, 2), al.u32))
                b_full = al.view(b_reg[j], al.Tensor((4,), al.u32))
                b_pair = al.view(b_full, al.Tensor((2, 2), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_pair[0], b_pair[0], acc[acc_idx])
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_pair[1], b_pair[1], acc[acc_idx])

        al.syncthreads()

    zero = al.convert(0.0, al.f32)
    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    for j in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        c_bias = al.convert(g_conv_bias[col], al.f32)
        e_bias = al.convert(g_extra_bias[col], al.f32)
        for i in al.range(M_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            for t in al.range(ACC_SIZE):
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                result = acc[acc_idx, t] + c_bias
                if result < zero:
                    result = zero
                result = result + e_bias
                g_out[row, col] = al.convert(result, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_relu_bias(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    extra_bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    _N, _C, _H, _W = x.shape
    OC, IC, KH, KW = weight.shape

    OH = _H - KH + 1
    OW = _W - KW + 1
    K_total = IC * KH * KW
    M_total = _N * OH * OW

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight.reshape(OC, K_total))
    conv_bias_bf16 = _prepare_bf16_cuda_contiguous(conv_bias)
    extra_bias_bf16 = _prepare_bf16_cuda_contiguous(extra_bias)

    im2col_out = torch.empty((M_total, K_total), device=x_bf16.device, dtype=torch.bfloat16)
    im2col_grid = ((M_total + IM2COL_ROWS_PER_BLOCK - 1) // IM2COL_ROWS_PER_BLOCK, 1, 1)
    im2col_kernel[lambda: (im2col_grid, (IM2COL_BLOCK, 1, 1))](
        x_bf16, im2col_out,
        _N, IC, _H, _W, OH, OW, KH, KW, K_total, M_total,
    )

    out_2d = torch.empty((M_total, OC), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (OC // GROUP_N, M_total // GROUP_M, 1)
    conv_relu_bias_kernel[lambda: (grid, (THREADS, 1, 1))](
        im2col_out, weight_bf16, conv_bias_bf16, extra_bias_bf16, out_2d,
        M_total, OC, K_total,
    )

    out = out_2d.reshape(_N, OH, OW, OC).permute(0, 3, 1, 2).contiguous()
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        weight = self.conv.weight
        c_bias = self.conv.bias
        e_bias = self.bias.reshape(-1)
        return avelang_conv_relu_bias(x, weight, c_bias, e_bias)


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]


def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, BIAS_SHAPE]
