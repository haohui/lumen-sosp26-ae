import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def _load_global_a_to_shm(
    shm_a: al.Tensor((256, 4), al.u32),
    a_rsrc: al.Tensor((4,), al.u32),
    block_m: al.u32,
    k_base: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(1):
        row = idx // 2
        col_vec = idx % 2
        off = ((block_m * 128 + row) * k + k_base + col_vec * 8) * 2
        shm_a[idx] = al.amdgpu.raw_buffer_load_x4(a_rsrc, zero, off, 0)
        idx += 256


@avelang.jit
def _load_global_b_to_shm(
    shm_b: al.Tensor((256, 4), al.u32),
    b_rsrc: al.Tensor((4,), al.u32),
    block_n: al.u32,
    k_base: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(1):
        row = idx // 2
        col_vec = idx % 2
        off = ((block_n * 128 + row) * k + k_base + col_vec * 8) * 2
        shm_b[idx] = al.amdgpu.raw_buffer_load_x4(b_rsrc, zero, off, 0)
        idx += 256


@avelang.jit
def gemm_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid = al.thread_id(0)
    block_n = al.block_id(0)
    block_m = al.block_id(1)
    wid = tid // 64
    lane = tid % 64
    warp_row = wid // 2
    warp_col = wid % 2

    a_memref = al.make_tensor(a_ptr, al.bf16, al.make_layout((m * k,), (1,)))
    b_memref = al.make_tensor(b_ptr, al.bf16, al.make_layout((n * k,), (1,)))
    g_out = al.make_tensor(c_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    a_rsrc = al.amdgpu.make_rsrc(a_memref, m * k * 2)
    b_rsrc = al.amdgpu.make_rsrc(b_memref, n * k * 2)

    shm_a = al.make_shared((256, 4), al.u32)
    shm_b = al.make_shared((256, 4), al.u32)
    a_data = al.make_local((2, 4), al.u32)
    b_data = al.make_local((2, 4), al.u32)
    acc = al.make_local((4, 16), al.f32)

    for i in al.range(4):
        for j in al.range(16):
            acc[i, j] = al.convert(0.0, al.f32)

    k_tiles = k // 16
    for kt in al.range(k_tiles):
        k_base = kt * 16
        _load_global_a_to_shm(shm_a, a_rsrc, block_m, k_base, k, tid)
        _load_global_b_to_shm(shm_b, b_rsrc, block_n, k_base, k, tid)
        al.syncthreads()

        shm_a_u32 = al.view(shm_a, al.Tensor((1024,), al.u32))
        shm_b_u32 = al.view(shm_b, al.Tensor((1024,), al.u32))

        for i in al.range(2):
            a_row = (warp_row * 2 + i) * 32 + (lane % 32)
            a_k_group = (lane // 32) * 2
            a_base = a_row * 8
            a_data[i, 0] = shm_a_u32[a_base + a_k_group]
            a_data[i, 1] = shm_a_u32[a_base + a_k_group + 1]
            a_data[i, 2] = shm_a_u32[a_base + 4 + a_k_group]
            a_data[i, 3] = shm_a_u32[a_base + 5 + a_k_group]

        for j in al.range(2):
            b_row = (warp_col * 2 + j) * 32 + (lane % 32)
            b_k_group = (lane // 32) * 2
            b_base = b_row * 8
            b_data[j, 0] = shm_b_u32[b_base + b_k_group]
            b_data[j, 1] = shm_b_u32[b_base + b_k_group + 1]
            b_data[j, 2] = shm_b_u32[b_base + 4 + b_k_group]
            b_data[j, 3] = shm_b_u32[b_base + 5 + b_k_group]

        for i in al.range(2):
            for j in al.range(2):
                a_frag = al.view(a_data[i], al.Tensor((2, 2, 1), al.u32))
                b_frag = al.view(b_data[j], al.Tensor((2, 2, 1), al.u32))
                acc_idx = i * 2 + j
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(
                    a_frag[0], b_frag[0], acc[acc_idx],
                )
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(
                    a_frag[1], b_frag[1], acc[acc_idx],
                )

        al.syncthreads()

    lane_group = lane // 32
    lane_col = lane % 32
    block_row_base = block_m * 128
    block_col_base = block_n * 128

    for j in al.range(2):
        col = block_col_base + (warp_col * 2 + j) * 32 + lane_col
        for i in al.range(2):
            acc_idx = i * 2 + j
            row_base = block_row_base + (warp_row * 2 + i) * 32
            for t in al.range(16):
                row = row_base + 8 * (t // 4) + 4 * lane_group + (t % 4)
                if row < m and col < n:
                    g_out[row, col] = al.convert(acc[acc_idx, t], al.bf16)


def _prepare_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    a_bf16 = _prepare_bf16_contiguous(a)
    b_bf16 = _prepare_bf16_contiguous(b)

    m_val, k_val = a_bf16.shape
    k_val_b, n_val = b_bf16.shape

    if k_val != k_val_b:
        raise ValueError(f"K dimension mismatch: A has K={k_val}, B has K={k_val_b}")
    if m_val % 128 != 0 or n_val % 128 != 0 or k_val % 16 != 0:
        raise ValueError(
            f"Expected m % 128 == 0, n % 128 == 0, k % 16 == 0 "
            f"(got m={m_val}, n={n_val}, k={k_val})"
        )

    b_t = b_bf16.T.contiguous()

    out = torch.empty((m_val, n_val), device=a_bf16.device, dtype=torch.bfloat16)
    grid = (n_val // 128, m_val // 128, 1)
    gemm_kernel[lambda: (grid, (256, 1, 1))](a_bf16, b_t, out, m_val, n_val, k_val)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return avelang_gemm(A, B)


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2


def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]


def get_init_inputs():
    return []
