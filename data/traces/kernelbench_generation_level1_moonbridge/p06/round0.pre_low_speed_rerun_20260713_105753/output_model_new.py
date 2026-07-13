import torch
import torch.nn as nn
import avelang
import avelang.language as al

WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
GROUP_M = 64
GROUP_N = 64
GROUP_K = 128
MMA_M = 32
MMA_N = 32
VEC_ELEMS = 8
BF16_BYTES = 2
ACC_SIZE = 16
WARPS_M = 2
WARPS_N = 2
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW
GLOBAL_LOADS_A = SHM_A_VECS // THREADS
GLOBAL_LOADS_B = SHM_B_VECS // THREADS
ROW_U32 = A_VECS_PER_ROW * 4
K_SLICES = GROUP_K // 8


@avelang.jit
def matmul_bf16_kernel(
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

    a_memref = al.make_tensor(a_ptr, al.bf16, al.make_layout((m * k,), (1,)))
    b_memref = al.make_tensor(b_ptr, al.bf16, al.make_layout((n * k,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    a_rsrc = al.amdgpu.make_rsrc(a_memref, m * k * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    a_op = al.make_local((2,), al.u32)
    b_op = al.make_local((2,), al.u32)
    acc = al.make_local((ACC_SIZE,), al.f32)

    zero_u32 = al.convert(0, al.u32)

    for t in al.range(ACC_SIZE):
        acc[t] = 0

    shm_a_flat = al.view(shm_a, al.Tensor((SHM_A_VECS * 4,), al.u32))
    shm_b_flat = al.view(shm_b, al.Tensor((SHM_B_VECS * 4,), al.u32))

    k_group_u32 = (lane // MMA_M) * 2
    a_row = warp_row * MMA_M + (lane % MMA_M)
    a_row_base = a_row * ROW_U32
    b_row = warp_col * MMA_M + (lane % MMA_M)
    b_row_base = b_row * ROW_U32

    k_tiles = k // GROUP_K
    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K

        idx = tid
        for _ in al.range(GLOBAL_LOADS_A):
            row = idx // A_VECS_PER_ROW
            col_vec = idx % A_VECS_PER_ROW
            off = ((block_m * GROUP_M + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
            shm_a[idx] = al.amdgpu.raw_buffer_load_x4(a_rsrc, zero_u32, off, 0)
            idx += THREADS

        idx = tid
        for _ in al.range(GLOBAL_LOADS_B):
            row = idx // B_VECS_PER_ROW
            col_vec = idx % B_VECS_PER_ROW
            off = ((block_n * GROUP_N + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
            shm_b[idx] = al.amdgpu.raw_buffer_load_x4(b_rsrc, zero_u32, off, 0)
            idx += THREADS

        al.syncthreads()

        for ks in al.range(K_SLICES):
            k_off = ks * 4
            a_op[0] = shm_a_flat[a_row_base + k_off + k_group_u32]
            a_op[1] = shm_a_flat[a_row_base + k_off + k_group_u32 + 1]
            b_op[0] = shm_b_flat[b_row_base + k_off + k_group_u32]
            b_op[1] = shm_b_flat[b_row_base + k_off + k_group_u32 + 1]

            a0 = al.view(a_op, al.Tensor((2,), al.u32))
            b0 = al.view(b_op, al.Tensor((2,), al.u32))
            ac = al.view(acc, al.Tensor((ACC_SIZE,), al.f32))
            ac = al.amdgpu.mfma_f32_32x32x8_bf16(a0, b0, ac)

        al.syncthreads()

    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    row_base = block_m * GROUP_M + warp_row * MMA_M
    col = block_n * GROUP_N + warp_col * MMA_N + lane_col

    for t in al.range(ACC_SIZE):
        row = row_base + (t // 4) * 8 + lane_group * 4 + (t % 4)
        result = acc[t]
        g_out[row, col] = al.convert(result, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    a_bf16 = _prepare_bf16_cuda_contiguous(a)
    b_bf16 = _prepare_bf16_cuda_contiguous(b)

    m, k_a = a_bf16.shape
    k_b, n = b_bf16.shape
    if k_a != k_b:
        raise ValueError(
            f"Inner dimension mismatch: A has K={k_a}, B has K={k_b}"
        )
    k = k_a
    if m % GROUP_M != 0 or n % GROUP_N != 0 or k % GROUP_K != 0:
        raise ValueError(
            f"Expected m % {GROUP_M} == 0, n % {GROUP_N} == 0, k % {GROUP_K} == 0 "
            f"(got m={m}, n={n}, k={k})"
        )

    b_t = b_bf16.t().contiguous()

    out = torch.empty((m, n), device=a_bf16.device, dtype=torch.bfloat16)
    grid = (n // GROUP_N, m // GROUP_M, 1)
    matmul_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
        a_bf16, b_t, out, m, n, k
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return avelang_matmul(A, B)
