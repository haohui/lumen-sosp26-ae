import torch
import torch.nn as nn
import avelang
import avelang.language as al

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
NUM_GROUPS = 256
MULTIPLY_WEIGHT_SHAPE = (OUT_FEATURES,)
CHANNELS_PER_GROUP = OUT_FEATURES // NUM_GROUPS

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

GN_BLOCK_SIZE = 256


@avelang.jit
def gemm_kernel(
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
    a0 = al.make_local((M_TILES_PER_WARP, 2), al.u32)
    a1 = al.make_local((M_TILES_PER_WARP, 2), al.u32)
    b0 = al.make_local((N_TILES_PER_WARP, 2), al.u32)
    b1 = al.make_local((N_TILES_PER_WARP, 2), al.u32)
    acc = al.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), al.f32)

    for i in al.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for j in al.range(ACC_SIZE):
            acc[i, j] = 0

    k_tiles = k // GROUP_K
    zero_u32 = al.convert(0, al.u32)
    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K

        idx = tid
        for _ in al.range(GLOBAL_LOADS_A):
            row = idx // A_VECS_PER_ROW
            col_vec = idx % A_VECS_PER_ROW
            off = ((block_m * GROUP_M + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
            shm_a[idx] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero_u32, off, 0)
            idx += THREADS

        idx = tid
        for _ in al.range(GLOBAL_LOADS_B):
            row = idx // B_VECS_PER_ROW
            col_vec = idx % B_VECS_PER_ROW
            off = ((block_n * GROUP_N + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
            shm_b[idx] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero_u32, off, 0)
            idx += THREADS

        al.syncthreads()

        shm_a_flat = al.view(shm_a, al.Tensor((SHM_A_VECS * 4,), al.u32))
        shm_b_flat = al.view(shm_b, al.Tensor((SHM_B_VECS * 4,), al.u32))

        for i in al.range(M_TILES_PER_WARP):
            tile_r = warp_row * M_TILES_PER_WARP + i
            row = tile_r * MMA_M + (lane % MMA_M)
            kg = (lane // MMA_M) * 2
            rb = row * ROW_U32

            a0[i, 0] = shm_a_flat[rb + kg]
            a0[i, 1] = shm_a_flat[rb + kg + 1]
            a1[i, 0] = shm_a_flat[rb + 4 + kg]
            a1[i, 1] = shm_a_flat[rb + 5 + kg]

        for j in al.range(N_TILES_PER_WARP):
            tile_c = warp_col * N_TILES_PER_WARP + j
            row = tile_c * MMA_M + (lane % MMA_M)
            kg = (lane // MMA_M) * 2
            rb = row * ROW_U32

            b0[j, 0] = shm_b_flat[rb + kg]
            b0[j, 1] = shm_b_flat[rb + kg + 1]
            b1[j, 0] = shm_b_flat[rb + 4 + kg]
            b1[j, 1] = shm_b_flat[rb + 5 + kg]

        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a0[i], b0[j], acc[acc_idx])
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a1[i], b1[j], acc[acc_idx])

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
def groupnorm_swish_kernel(
    in_ptr: al.Pointer(al.bf16),
    gn_weight_ptr: al.Pointer(al.bf16),
    gn_bias_ptr: al.Pointer(al.bf16),
    multiply_weight_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    channels: al.i32,
    num_groups: al.i32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    batch = al.block_id(0)

    if batch < batch_size:
        group_id = tid
        base = group_id * CHANNELS_PER_GROUP

        layout_2d = al.make_layout((batch_size, channels), (channels, 1))
        x = al.make_tensor(in_ptr, al.bf16, layout_2d)

        layout_1d = al.make_layout((channels,), (1,))
        gn_w = al.make_tensor(gn_weight_ptr, al.bf16, layout_1d)
        gn_b = al.make_tensor(gn_bias_ptr, al.bf16, layout_1d)
        mw = al.make_tensor(multiply_weight_ptr, al.bf16, layout_1d)

        out_t = al.make_tensor(out_ptr, al.bf16, layout_2d)

        vals = al.make_local((CHANNELS_PER_GROUP,), al.f32)

        sum_val = al.convert(0.0, al.f32)
        for j in al.range(CHANNELS_PER_GROUP):
            v = al.convert(x[batch, base + j], al.f32)
            vals[j] = v
            sum_val = sum_val + v

        n_f32 = al.convert(CHANNELS_PER_GROUP, al.f32)
        mean = sum_val / n_f32

        var = al.convert(0.0, al.f32)
        for j in al.range(CHANNELS_PER_GROUP):
            diff = vals[j] - mean
            var = var + diff * diff
        var = var / n_f32

        rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)

        one = al.convert(1.0, al.f32)
        half = al.convert(0.5, al.f32)
        for j in al.range(CHANNELS_PER_GROUP):
            xn = (vals[j] - mean) * rstd
            w_val = al.convert(gn_w[base + j], al.f32)
            b_val = al.convert(gn_b[base + j], al.f32)
            y = xn * w_val + b_val
            y = y * half * (one + al.tanh(y * half))
            mw_val = al.convert(mw[base + j], al.f32)
            y = y * mw_val
            y = y * half * (one + al.tanh(y * half))
            out_t[batch, base + j] = al.convert(y, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)
    b_bf16 = _prepare_bf16_cuda_contiguous(bias)

    m_size, k_size = x_bf16.shape
    n_size, w_k = w_bf16.shape

    if w_k != k_size:
        raise ValueError(f"Weight/input K mismatch")
    if m_size % GROUP_M != 0 or n_size % GROUP_N != 0 or k_size % GROUP_K != 0:
        raise ValueError(f"Shape constraints: m={m_size}, n={n_size}, k={k_size}")

    out = torch.empty((m_size, n_size), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (n_size // GROUP_N, m_size // GROUP_M, 1)
    gemm_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, b_bf16, out, m_size, n_size, k_size
    )
    return out


def avelang_groupnorm_swish(
    x: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    multiply_weight: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    gn_w_bf16 = _prepare_bf16_cuda_contiguous(gn_weight)
    gn_b_bf16 = _prepare_bf16_cuda_contiguous(gn_bias)
    mw_bf16 = _prepare_bf16_cuda_contiguous(multiply_weight)

    batch_size, channels = x_bf16.shape

    if channels % num_groups != 0:
        raise ValueError(f"channels {channels} not divisible by num_groups {num_groups}")
    if num_groups != GN_BLOCK_SIZE:
        raise ValueError(f"Expected num_groups={GN_BLOCK_SIZE}, got {num_groups}")

    eps = 1e-5
    out = torch.empty_like(x_bf16)

    groupnorm_swish_kernel[lambda: ((batch_size, 1, 1), (GN_BLOCK_SIZE, 1, 1))](
        x_bf16, gn_w_bf16, gn_b_bf16, mw_bf16, out,
        batch_size, channels, num_groups, eps,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        w = self.gemm.weight.data
        b = self.gemm.bias.data
        gn_w = self.group_norm.weight.data
        gn_b = self.group_norm.bias.data
        mw = self.multiply_weight.data

        input_dtype = x.dtype

        gemm_out = avelang_gemm(x, w, b)
        result = avelang_groupnorm_swish(gemm_out, gn_w, gn_b, mw, NUM_GROUPS)

        if result.dtype != input_dtype:
            result = result.to(input_dtype)
        return result


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_FEATURES)]


def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES, NUM_GROUPS, MULTIPLY_WEIGHT_SHAPE]
