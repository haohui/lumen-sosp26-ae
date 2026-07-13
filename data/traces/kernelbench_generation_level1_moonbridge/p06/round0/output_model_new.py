import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32
THREADS = 64


@avelang.jit
def matmul_bf16_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    a_bf16 = al.make_tensor(a_ptr, al.bf16, al.make_layout((m, k), (k, 1)))
    b_bf16 = al.make_tensor(b_ptr, al.bf16, al.make_layout((n, k), (k, 1)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    k_i32 = k >> 1
    k_vecs = k >> 3
    a_vec = al.view(
        a_bf16,
        al.i32,
        al.make_layout((m, k_vecs, 4), (k_i32, 4, 1)),
    )
    b_vec = al.view(
        b_bf16,
        al.i32,
        al.make_layout((n, k_vecs, 4), (k_i32, 4, 1)),
    )

    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5

    block_m = al.block_id(1) * BLOCK_M
    block_n = al.block_id(0) * BLOCK_N

    k_groups_per_step = BLOCK_K >> 3
    # Shared memory: (rows * groups_per_step) x 4 i32 per group.
    a_smem = al.make_shared((BLOCK_M * k_groups_per_step, 4), al.i32)
    b_smem = al.make_shared((BLOCK_N * k_groups_per_step, 4), al.i32)

    acc = al.full((16,), 0.0, al.f32)

    k_tiles = k // BLOCK_K
    for kt in al.range(k_tiles):
        k_base = kt * k_groups_per_step

        # Each lane loads two 4-i32 vectors: K-group lane_group and lane_group+2.
        smem_row = lane_col * k_groups_per_step + lane_group
        a_smem[smem_row] = a_vec[block_m + lane_col, k_base + lane_group]
        b_smem[smem_row] = b_vec[block_n + lane_col, k_base + lane_group]

        smem_row2 = lane_col * k_groups_per_step + lane_group + 2
        a_smem[smem_row2] = a_vec[block_m + lane_col, k_base + lane_group + 2]
        b_smem[smem_row2] = b_vec[block_n + lane_col, k_base + lane_group + 2]

        al.syncthreads()

        # Sub-iteration 0: process K-groups 0 (lanes 0-31) and 1 (lanes 32-63).
        smem_idx = lane_col * k_groups_per_step + lane_group
        a_words0 = a_smem[smem_idx]
        b_words0 = b_smem[smem_idx]
        a_frag0 = al.view(a_words0, al.Tensor((2, 2, 1), al.i32))
        b_frag0 = al.view(b_words0, al.Tensor((2, 2, 1), al.i32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        # Sub-iteration 1: process K-groups 2 (lanes 0-31) and 3 (lanes 32-63).
        smem_idx2 = lane_col * k_groups_per_step + lane_group + 2
        a_words1 = a_smem[smem_idx2]
        b_words1 = b_smem[smem_idx2]
        a_frag1 = al.view(a_words1, al.Tensor((2, 2, 1), al.i32))
        b_frag1 = al.view(b_words1, al.Tensor((2, 2, 1), al.i32))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

        al.syncthreads()

    for r in al.range(16):
        row = block_m + ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        col = block_n + lane_col
        g_out[row, col] = al.convert(acc[r], al.bf16)


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
            f"K dimension mismatch: A has K={k_a}, B has K={k_b}"
        )
    if m % BLOCK_M != 0 or n % BLOCK_N != 0 or k_a % BLOCK_K != 0:
        raise ValueError(
            f"Expected m % {BLOCK_M} == 0, n % {BLOCK_N} == 0, k % {BLOCK_K} == 0 "
            f"(got m={m}, n={n}, k={k_a})"
        )

    b_t = b_bf16.t().contiguous()

    out = torch.empty((m, n), device=a_bf16.device, dtype=torch.bfloat16)
    grid = (n // BLOCK_N, m // BLOCK_M, 1)
    matmul_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
        a_bf16, b_t, out, m, n, k_a
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return avelang_matmul(A, B)
