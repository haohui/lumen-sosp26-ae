import torch
import torch.nn as nn
import avelang
import avelang.language as al

# --- GEMM constants ---
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

# --- BN constants ---
BN_BLOCK_SIZE = 256
BN_COL_TILE = 256


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


@avelang.jit
def linear_bf16_kernel(
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

        _fetch_mfma_operand_32x32x16(a_reg[0], shm_a, warp_row * M_TILES_PER_WARP + 0, lane)
        _fetch_mfma_operand_32x32x16(a_reg[1], shm_a, warp_row * M_TILES_PER_WARP + 1, lane)
        _fetch_mfma_operand_32x32x16(b_reg[0], shm_b, warp_col * N_TILES_PER_WARP + 0, lane)
        _fetch_mfma_operand_32x32x16(b_reg[1], shm_b, warp_col * N_TILES_PER_WARP + 1, lane)

        a00 = al.view(a_reg[0, 0], al.Tensor((2,), al.u32))
        a01 = al.view(a_reg[0, 1], al.Tensor((2,), al.u32))
        a10 = al.view(a_reg[1, 0], al.Tensor((2,), al.u32))
        a11 = al.view(a_reg[1, 1], al.Tensor((2,), al.u32))
        b00 = al.view(b_reg[0, 0], al.Tensor((2,), al.u32))
        b01 = al.view(b_reg[0, 1], al.Tensor((2,), al.u32))
        b10 = al.view(b_reg[1, 0], al.Tensor((2,), al.u32))
        b11 = al.view(b_reg[1, 1], al.Tensor((2,), al.u32))

        acc[0] = al.amdgpu.mfma_f32_32x32x8_bf16(a00, b00, acc[0])
        acc[0] = al.amdgpu.mfma_f32_32x32x8_bf16(a01, b01, acc[0])
        acc[1] = al.amdgpu.mfma_f32_32x32x8_bf16(a00, b10, acc[1])
        acc[1] = al.amdgpu.mfma_f32_32x32x8_bf16(a01, b11, acc[1])
        acc[2] = al.amdgpu.mfma_f32_32x32x8_bf16(a10, b00, acc[2])
        acc[2] = al.amdgpu.mfma_f32_32x32x8_bf16(a11, b01, acc[2])
        acc[3] = al.amdgpu.mfma_f32_32x32x8_bf16(a10, b10, acc[3])
        acc[3] = al.amdgpu.mfma_f32_32x32x8_bf16(a11, b11, acc[3])

        al.syncthreads()

    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    col0 = block_col_base + (warp_col * N_TILES_PER_WARP + 0) * MMA_N + lane_col
    col1 = block_col_base + (warp_col * N_TILES_PER_WARP + 1) * MMA_N + lane_col
    bias0 = al.convert(g_bias[col0], al.f32)
    bias1 = al.convert(g_bias[col1], al.f32)

    row_base0 = block_row_base + (warp_row * M_TILES_PER_WARP + 0) * MMA_M
    row_base1 = block_row_base + (warp_row * M_TILES_PER_WARP + 1) * MMA_M

    for t in al.range(ACC_SIZE):
        row0 = row_base0 + (t // 4) * 8 + lane_group * 4 + (t % 4)
        row1 = row_base1 + (t // 4) * 8 + lane_group * 4 + (t % 4)
        g_out[row0, col0] = al.convert(acc[0, t] + bias0, al.bf16)
        g_out[row0, col1] = al.convert(acc[1, t] + bias1, al.bf16)
        g_out[row1, col0] = al.convert(acc[2, t] + bias0, al.bf16)
        g_out[row1, col1] = al.convert(acc[3, t] + bias1, al.bf16)


@avelang.jit
def batchnorm_reduce_kernel(
    in_ptr: al.Pointer(al.bf16),
    mean_out_ptr: al.Pointer(al.f32),
    rstd_out_ptr: al.Pointer(al.f32),
    m: al.i32,
    n: al.i32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    layout_in = al.make_layout((m, n), (n, 1))
    g_in = al.make_tensor(in_ptr, al.bf16, layout_in)

    col_start = bid * BN_COL_TILE

    idx = tid
    for _ in al.range(0, BN_COL_TILE, BN_BLOCK_SIZE):
        col = col_start + idx
        if col < n:
            local_sum = al.convert(0.0, al.f32)
            local_sq = al.convert(0.0, al.f32)
            for row in al.range(m):
                val = al.convert(g_in[row, col], al.f32)
                local_sum = local_sum + val
                local_sq = local_sq + val * val

            m_f32 = al.convert(m, al.f32)
            mean = local_sum / m_f32
            var = local_sq / m_f32 - mean * mean
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)

            layout_s = al.make_layout((n,), (1,))
            mo = al.make_tensor(mean_out_ptr, al.f32, layout_s)
            ro = al.make_tensor(rstd_out_ptr, al.f32, layout_s)
            mo[col] = mean
            ro[col] = rstd

        idx += BN_BLOCK_SIZE


@avelang.jit
def batchnorm_apply_swish_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    rstd_ptr: al.Pointer(al.f32),
    bn_weight_ptr: al.Pointer(al.bf16),
    bn_bias_ptr: al.Pointer(al.bf16),
    extra_bias: al.f32,
    divide_value: al.f32,
    m: al.i32,
    n: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    layout_in = al.make_layout((m, n), (n, 1))
    g_in = al.make_tensor(in_ptr, al.bf16, layout_in)
    g_out = al.make_tensor(out_ptr, al.bf16, layout_in)

    layout_s = al.make_layout((n,), (1,))
    g_mean = al.make_tensor(mean_ptr, al.f32, layout_s)
    g_rstd = al.make_tensor(rstd_ptr, al.f32, layout_s)
    g_bn_w = al.make_tensor(bn_weight_ptr, al.bf16, layout_s)
    g_bn_b = al.make_tensor(bn_bias_ptr, al.bf16, layout_s)

    col_start = bid * BN_COL_TILE

    idx = tid
    for _ in al.range(0, BN_COL_TILE, BN_BLOCK_SIZE):
        col = col_start + idx
        if col < n:
            mean = g_mean[col]
            rstd = g_rstd[col]
            bn_w = al.convert(g_bn_w[col], al.f32)
            bn_b = al.convert(g_bn_b[col], al.f32)

            for row in al.range(m):
                x_val = al.convert(g_in[row, col], al.f32)
                normed = (x_val - mean) * rstd
                result = normed * bn_w + bn_b
                result = result + extra_bias
                result = result / divide_value
                one = al.convert(1.0, al.f32)
                zero = al.convert(0.0, al.f32)
                neg_result = zero - result
                sigmoid_val = one / (one + al.exp(neg_result))
                result = result * sigmoid_val
                g_out[row, col] = al.convert(result, al.bf16)

        idx += BN_BLOCK_SIZE


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication, batch normalization,
    bias addition, division, and Swish activation using AveLang DSL.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1,
                 bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.divide_value = divide_value

        self.linear_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.linear_bias = nn.Parameter(torch.empty(out_features))
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))

        nn.init.kaiming_uniform_(self.linear_weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear_weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.linear_bias, -bound, bound)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._cached_extra_bias = self.bias.item()
        self._cached_divide = divide_value

    def forward(self, x):
        x_bf16 = _prepare_bf16_cuda_contiguous(x)
        lw_bf16 = _prepare_bf16_cuda_contiguous(self.linear_weight)
        lb_bf16 = _prepare_bf16_cuda_contiguous(self.linear_bias)
        bnw_bf16 = _prepare_bf16_cuda_contiguous(self.bn_weight)
        bnb_bf16 = _prepare_bf16_cuda_contiguous(self.bn_bias)

        m, k = x_bf16.shape
        n, _ = lw_bf16.shape

        # Phase 1: GEMM (x @ w.T + linear_bias) -> gemm_out (BF16)
        gemm_out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
        grid = (n // GROUP_N, m // GROUP_M, 1)
        linear_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
            x_bf16, lw_bf16, lb_bf16, gemm_out, m, n, k
        )

        # Phase 2: BN stats (or eval-mode defaults)
        num_col_tiles = (n + BN_COL_TILE - 1) // BN_COL_TILE
        mean_buf = torch.empty((n,), device=x_bf16.device, dtype=torch.float32)
        rstd_buf = torch.empty((n,), device=x_bf16.device, dtype=torch.float32)

        if not self.training:
            # Eval mode: running_mean=0, running_var=1 for fresh BN
            mean_buf.zero_()
            rstd_buf.fill_(1.0 / (1.0 + self.bn_eps) ** 0.5)
        else:
            batchnorm_reduce_kernel[lambda: ((num_col_tiles, 1, 1), (BN_BLOCK_SIZE, 1, 1))](
                gemm_out, mean_buf, rstd_buf, m, n, self.bn_eps
            )

        # Phase 3: BN apply + bias + divide + swish -> final_out (BF16)
        final_out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)

        batchnorm_apply_swish_kernel[lambda: ((num_col_tiles, 1, 1), (BN_BLOCK_SIZE, 1, 1))](
            gemm_out, final_out, mean_buf, rstd_buf, bnw_bf16, bnb_bf16,
            self._cached_extra_bias, self._cached_divide, m, n
        )

        if x.dtype != torch.bfloat16:
            return final_out.to(dtype=x.dtype)
        return final_out
