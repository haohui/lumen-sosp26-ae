import torch
import torch.nn as nn
import avelang
import avelang.language as al


BLOCK_M = 128
BLOCK_N = 128
TILE_K = 64
THREAD_M = 16
THREAD_N = 16
BF16_BYTES = 2

ELEMS_PER_THREAD_M = BLOCK_M // THREAD_M
ELEMS_PER_THREAD_N = BLOCK_N // THREAD_N


@avelang.jit
def matmul_bf16_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid_x = al.thread_id(0)
    tid_y = al.thread_id(1)
    block_x = al.block_id(0)
    block_y = al.block_id(1)

    a = al.make_tensor(a_ptr, al.bf16, al.make_layout((m, k), (k, 1)))
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((n, k), (k, 1)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    shm_a = al.make_shared((BLOCK_M, TILE_K), al.bf16)
    shm_b = al.make_shared((BLOCK_N, TILE_K), al.bf16)

    row_off = block_y * BLOCK_M
    col_off = block_x * BLOCK_N

    acc = al.make_local((ELEMS_PER_THREAD_M, ELEMS_PER_THREAD_N), al.f32)
    for i in al.range(ELEMS_PER_THREAD_M):
        for j in al.range(ELEMS_PER_THREAD_N):
            acc[i, j] = 0

    num_k_tiles = k // TILE_K
    for kt in al.range(num_k_tiles):
        k_off = kt * TILE_K

        for r in al.range(ELEMS_PER_THREAD_M):
            shm_row = r * THREAD_M + tid_y
            g_row = row_off + shm_row
            for c in al.range(TILE_K // THREAD_N):
                shm_col = c * THREAD_N + tid_x
                if g_row < m and k_off + shm_col < k:
                    shm_a[shm_row, shm_col] = a[g_row, k_off + shm_col]

        for r in al.range(ELEMS_PER_THREAD_N):
            shm_row = r * THREAD_M + tid_y
            g_row = col_off + shm_row
            for c in al.range(TILE_K // THREAD_N):
                shm_col = c * THREAD_N + tid_x
                if g_row < n and k_off + shm_col < k:
                    shm_b[shm_row, shm_col] = b[g_row, k_off + shm_col]

        al.syncthreads()

        for kk in al.range(TILE_K):
            for i in al.range(ELEMS_PER_THREAD_M):
                a_row = i * THREAD_M + tid_y
                a_val = al.convert(shm_a[a_row, kk], al.f32)
                for j in al.range(ELEMS_PER_THREAD_N):
                    b_row = j * THREAD_N + tid_x
                    b_val = al.convert(shm_b[b_row, kk], al.f32)
                    acc[i, j] = acc[i, j] + a_val * b_val

        al.syncthreads()

    for i in al.range(ELEMS_PER_THREAD_M):
        g_row = row_off + i * THREAD_M + tid_y
        for j in al.range(ELEMS_PER_THREAD_N):
            g_col = col_off + j * THREAD_N + tid_x
            if g_row < m and g_col < n:
                g_out[g_row, g_col] = al.convert(acc[i, j], al.bf16)


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

    m_val, k_val = a_bf16.shape
    n_val, b_k = b_bf16.shape
    if b_k != k_val:
        raise ValueError(f"K mismatch")

    out = torch.empty((m_val, n_val), device=a_bf16.device, dtype=torch.bfloat16)
    grid = (n_val // BLOCK_N, m_val // BLOCK_M, 1)
    matmul_bf16_kernel[lambda: (grid, (THREAD_N, THREAD_M, 1))](
        a_bf16, b_bf16, out, m_val, n_val, k_val
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return avelang_matmul(A, B)
