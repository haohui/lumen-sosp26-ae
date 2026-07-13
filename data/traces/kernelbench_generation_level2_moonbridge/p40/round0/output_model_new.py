import torch
import torch.nn as nn
import avelang
import avelang.language as al

TM = 64
TN = 64
BK = 32
TM_THREADS = 16
TN_THREADS = 16
THREADS = TM_THREADS * TN_THREADS
ELEMS_PER_THREAD_M = TM // TM_THREADS
ELEMS_PER_THREAD_N = TN // TN_THREADS
A_VECS_PER_ROW = BK // 8
B_VECS_PER_ROW = BK // 8
SHM_A_VECS = TM * A_VECS_PER_ROW
SHM_B_VECS = TN * B_VECS_PER_ROW
VEC_ELEMS = 8
BF16_BYTES = 2


@avelang.jit
def matmul_scale_residual_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
    scale: al.f32,
):
    tid_x = al.thread_id(0)
    tid_y = al.thread_id(1)
    block_m = al.block_id(0)
    block_n = al.block_id(1)

    x_memref = al.make_tensor(x_ptr, al.bf16, al.make_layout((m * k,), (1,)))
    w_memref = al.make_tensor(w_ptr, al.bf16, al.make_layout((n * k,), (1,)))
    g_bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    x_rsrc = al.amdgpu.make_rsrc(x_memref, m * k * BF16_BYTES)
    w_rsrc = al.amdgpu.make_rsrc(w_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    shm_a_flat = al.view(shm_a, al.Tensor((SHM_A_VECS * 4,), al.u32))
    shm_b_flat = al.view(shm_b, al.Tensor((SHM_B_VECS * 4,), al.u32))
    shm_a_bf16 = al.view(shm_a_flat, al.Tensor((TM * BK,), al.bf16))
    shm_b_bf16 = al.view(shm_b_flat, al.Tensor((TN * BK,), al.bf16))

    acc = al.make_local((ELEMS_PER_THREAD_M, ELEMS_PER_THREAD_N), al.f32)
    for ri in al.range(ELEMS_PER_THREAD_M):
        for ci in al.range(ELEMS_PER_THREAD_N):
            acc[ri, ci] = 0

    zero_u32 = al.convert(0, al.u32)
    row_start = block_m * TM
    col_start = block_n * TN
    flat_tid = tid_y * TN_THREADS + tid_x

    k_tiles = k // BK
    for kt in al.range(k_tiles):
        k_base = kt * BK

        idx = flat_tid
        for _ in al.range(SHM_A_VECS // THREADS):
            if idx < SHM_A_VECS:
                row = idx // A_VECS_PER_ROW
                col_vec = idx % A_VECS_PER_ROW
                off = ((row_start + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
                shm_a[idx] = al.amdgpu.raw_buffer_load_x4(x_rsrc, zero_u32, off, 0)
            idx += THREADS

        idx = flat_tid
        for _ in al.range(SHM_B_VECS // THREADS):
            if idx < SHM_B_VECS:
                row = idx // B_VECS_PER_ROW
                col_vec = idx % B_VECS_PER_ROW
                off = ((col_start + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
                shm_b[idx] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero_u32, off, 0)
            idx += THREADS

        al.syncthreads()

        for ri in al.range(ELEMS_PER_THREAD_M):
            a_row = tid_y * ELEMS_PER_THREAD_M + ri
            a_base = a_row * BK
            for kk in al.range(BK):
                a_val = al.convert(shm_a_bf16[a_base + kk], al.f32)
                for ci in al.range(ELEMS_PER_THREAD_N):
                    b_row = tid_x * ELEMS_PER_THREAD_N + ci
                    b_val = al.convert(shm_b_bf16[b_row * BK + kk], al.f32)
                    acc[ri, ci] = acc[ri, ci] + a_val * b_val

        al.syncthreads()

    for ri in al.range(ELEMS_PER_THREAD_M):
        row = row_start + tid_y * ELEMS_PER_THREAD_M + ri
        for ci in al.range(ELEMS_PER_THREAD_N):
            col = col_start + tid_x * ELEMS_PER_THREAD_N + ci
            if row < m and col < n:
                bias_val = al.convert(g_bias[col], al.f32)
                result = acc[ri, ci] + bias_val
                result = result * scale + result
                g_out[row, col] = al.convert(result, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_matmul_scale_residual(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    m, k = x_bf16.shape
    n, weight_k = weight_bf16.shape
    if weight_k != k:
        raise ValueError(f"Weight/input K mismatch: x has K={k}, weight has K={weight_k}")
    if m % TM != 0 or n % TN != 0 or k % BK != 0:
        raise ValueError(f"Expected m % {TM} == 0, n % {TN} == 0, k % {BK} == 0 (got m={m}, n={n}, k={k})")

    out = torch.empty((m, n), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (m // TM, n // TN, 1)
    matmul_scale_residual_bf16_kernel[lambda: (grid, (TN_THREADS, TM_THREADS, 1))](
        x_bf16, weight_bf16, bias_bf16, out, m, n, k, scale
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, scaling_factor: float):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_matmul_scale_residual(
            x, self.matmul.weight, self.matmul.bias, self.scaling_factor
        )


def get_inputs():
    return [torch.rand(16384, 4096)]


def get_init_inputs():
    return [4096, 4096, 0.5]
