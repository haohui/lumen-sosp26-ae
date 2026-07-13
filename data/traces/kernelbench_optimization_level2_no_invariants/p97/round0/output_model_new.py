import torch
import torch.nn as nn
import avelang
import avelang.language as al

EPS = 1e-05
DIVIDE_VALUE = 1.0

WARP_SIZE = 64
NUM_WARPS = 4
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
THREADS = WARP_SIZE * NUM_WARPS
GLOBAL_LOADS_A = SHM_A_VECS // THREADS
GLOBAL_LOADS_B = SHM_B_VECS // THREADS
ROW_U32 = A_VECS_PER_ROW * 4


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
def matmul_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias0_ptr: al.Pointer(al.bf16),
    bn_w_ptr: al.Pointer(al.bf16),
    bn_b_ptr: al.Pointer(al.bf16),
    running_mean_ptr: al.Pointer(al.bf16),
    running_var_ptr: al.Pointer(al.bf16),
    extra_bias: al.Tensor((1,), al.bf16),
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
    g_bias = al.make_tensor(bias0_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_bn_w = al.make_tensor(bn_w_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_bn_b = al.make_tensor(bn_b_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_running_mean = al.make_tensor(running_mean_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_running_var = al.make_tensor(running_var_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_out = al.make_tensor(y_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

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
                a_packed = al.view(a_reg[i], al.Tensor((4,), al.u32))
                b_packed = al.view(b_reg[j], al.Tensor((4,), al.u32))
                a_2d = al.view(a_packed, al.Tensor((2, 2), al.u32))
                b_2d = al.view(b_packed, al.Tensor((2, 2), al.u32))
                for step in al.range(2):
                    a_vec = al.view(a_2d[step], al.Tensor((2,), al.u32))
                    b_vec = al.view(b_2d[step], al.Tensor((2,), al.u32))
                    acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_vec, b_vec, acc[acc_idx])

        al.syncthreads()

    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M
    block_col_base = block_n * GROUP_N

    extra_val = al.convert(extra_bias[0], al.f32)
    one = al.convert(1.0, al.f32)
    div_val = al.convert(DIVIDE_VALUE, al.f32)
    eps_f32 = al.convert(EPS, al.f32)

    for j in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        bias0_val = al.convert(g_bias[col], al.f32)
        bn_w_val = al.convert(g_bn_w[col], al.f32)
        bn_b_val = al.convert(g_bn_b[col], al.f32)
        rm_val = al.convert(g_running_mean[col], al.f32)
        rv_val = al.convert(g_running_var[col], al.f32)
        rdenom = al.sqrt(rv_val + eps_f32)
        for i in al.range(M_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            for t in al.range(ACC_SIZE):
                row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
                result = acc[acc_idx, t] + bias0_val
                result = (result - rm_val) / rdenom * bn_w_val + bn_b_val
                result = (result + extra_val) / div_val
                result = result * (one / (one + al.exp(-result)))
                g_out[row, col] = al.convert(result, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


class ModelNew(nn.Module):

    def __init__(self, in_features, out_features, bn_eps=1e-05, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, x):
        m_val = x.shape[0]
        k_val = x.shape[1]
        n_val = self.matmul.out_features

        if m_val % GROUP_M != 0 or n_val % GROUP_N != 0 or k_val % GROUP_K != 0:
            raise RuntimeError(
                f"Expected m % {GROUP_M} == 0, n % {GROUP_N} == 0, k % {GROUP_K} == 0 "
                f"(got m={m_val}, n={n_val}, k={k_val})"
            )

        x_bf16 = _prepare_bf16_cuda_contiguous(x)
        weight_bf16 = _prepare_bf16_cuda_contiguous(self.matmul.weight)
        bias0_bf16 = _prepare_bf16_cuda_contiguous(self.matmul.bias)
        bn_w_bf16 = _prepare_bf16_cuda_contiguous(self.bn.weight)
        bn_b_bf16 = _prepare_bf16_cuda_contiguous(self.bn.bias)
        running_mean_bf16 = _prepare_bf16_cuda_contiguous(self.bn.running_mean)
        running_var_bf16 = _prepare_bf16_cuda_contiguous(self.bn.running_var)
        extra_b_bf16 = _prepare_bf16_cuda_contiguous(self.bias)

        y = torch.empty((m_val, n_val), device=x_bf16.device, dtype=torch.bfloat16)

        grid_mm = (n_val // GROUP_N, m_val // GROUP_M, 1)
        matmul_kernel[lambda: (grid_mm, (THREADS, 1, 1))](
            x_bf16, weight_bf16, bias0_bf16, bn_w_bf16, bn_b_bf16,
            running_mean_bf16, running_var_bf16, extra_b_bf16,
            y, m_val, n_val, k_val
        )

        return y
