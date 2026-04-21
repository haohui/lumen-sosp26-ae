import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH = 8
I = 256
J = 512
L = 256
K = 768

M = BATCH * I * J
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 16
LOAD_THREADS = 64
THREADS_PER_BLOCK = BLOCK_M * BLOCK_N
A_RANGE_BYTES = M * L * 2
B_RANGE_BYTES = L * K * 2


@substrate.jit
def einsum4d_pipelined_kernel(
    A: S.Tensor((BATCH, I, J, L), S.bf16),
    B: S.Tensor((L, K), S.bf16),
    C: S.Tensor((BATCH, I, J, K), S.bf16),
):
    tid = S.thread_id(0)
    tx = tid % BLOCK_N
    ty = tid // BLOCK_N

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N
    row = block_row + ty
    col = block_col + tx

    a_flat = S.view(A, S.Tensor((M, L), S.bf16))
    c_flat = S.view(C, S.Tensor((M, K), S.bf16))

    a_u32 = S.view(a_flat, S.Tensor((M, L // 2), S.u32))
    b_u32 = S.view(B, S.Tensor((L, K // 2), S.u32))

    # Keep the raw-buffer range explicit so speculative accesses are
    # zero-filled by hardware if a future shape variant crosses the boundary.
    a_rsrc = S.amdgpu.make_rsrc(a_u32, A_RANGE_BYTES)
    b_rsrc = S.amdgpu.make_rsrc(b_u32, B_RANGE_BYTES)

    a_shared_u32 = S.make_shared((2, BLOCK_M, BLOCK_K // 2), S.u32)
    b_shared_u32 = S.make_shared((2, BLOCK_K, BLOCK_N // 2), S.u32)

    a_shared = S.view(a_shared_u32, S.Tensor((2, BLOCK_M, BLOCK_K), S.bf16))
    b_shared = S.view(b_shared_u32, S.Tensor((2, BLOCK_K, BLOCK_N), S.bf16))

    next_a = S.make_local((4,), S.u32)
    next_b = S.make_local((4,), S.u32)
    acc = S.convert(0.0, S.f32)

    if tid < LOAD_THREADS:
        load_idx = tid
        if load_idx < (BLOCK_M * BLOCK_K) // 8:
            a_row = load_idx // (BLOCK_K // 8)
            a_col = (load_idx % (BLOCK_K // 8)) * 8
            a_base_u32 = ((block_row + a_row) * (L // 2) + (a_col // 2)) * 4
            a_vec = S.amdgpu.raw_buffer_load_x4(
                a_rsrc, S.convert(a_base_u32, S.i32), 0, 0
            )
            for i in S.range(4):
                a_shared_u32[0, a_row, a_col // 2 + i] = a_vec[i]
        else:
            b_idx = load_idx - (BLOCK_M * BLOCK_K) // 8
            b_row = b_idx // (BLOCK_N // 8)
            b_col = (b_idx % (BLOCK_N // 8)) * 8
            b_base_u32 = ((b_row) * (K // 2) + ((block_col + b_col) // 2)) * 4
            b_vec = S.amdgpu.raw_buffer_load_x4(
                b_rsrc, S.convert(b_base_u32, S.i32), 0, 0
            )
            for i in S.range(4):
                b_shared_u32[0, b_row, b_col // 2 + i] = b_vec[i]
    S.syncthreads()

    current = 0
    for k_base in S.range(0, L, BLOCK_K):
        has_next = k_base + BLOCK_K < L

        if tid < LOAD_THREADS and has_next:
            load_idx = tid
            if load_idx < (BLOCK_M * BLOCK_K) // 8:
                a_row = load_idx // (BLOCK_K // 8)
                a_col = (load_idx % (BLOCK_K // 8)) * 8
                a_base_u32 = (
                    ((block_row + a_row) * (L // 2) + ((k_base + BLOCK_K + a_col) // 2))
                    * 4
                )
                next_a = S.amdgpu.raw_buffer_load_x4(
                    a_rsrc, S.convert(a_base_u32, S.i32), 0, 0
                )
            else:
                b_idx = load_idx - (BLOCK_M * BLOCK_K) // 8
                b_row = b_idx // (BLOCK_N // 8)
                b_col = (b_idx % (BLOCK_N // 8)) * 8
                b_base_u32 = (
                    (((k_base + BLOCK_K + b_row) * (K // 2)) + ((block_col + b_col) // 2))
                    * 4
                )
                next_b = S.amdgpu.raw_buffer_load_x4(
                    b_rsrc, S.convert(b_base_u32, S.i32), 0, 0
                )

        for kk in S.range(0, BLOCK_K // 2):
            acc += S.convert(a_shared[current, ty, kk], S.f32) * S.convert(
                b_shared[current, kk, tx], S.f32
            )

        if tid < LOAD_THREADS and has_next:
            load_idx = tid
            if load_idx < (BLOCK_M * BLOCK_K) // 8:
                a_row = load_idx // (BLOCK_K // 8)
                a_col = (load_idx % (BLOCK_K // 8)) * 8
                for i in S.range(4):
                    a_shared_u32[1 - current, a_row, a_col // 2 + i] = next_a[i]
            else:
                b_idx = load_idx - (BLOCK_M * BLOCK_K) // 8
                b_row = b_idx // (BLOCK_N // 8)
                b_col = (b_idx % (BLOCK_N // 8)) * 8
                for i in S.range(4):
                    b_shared_u32[1 - current, b_row, b_col // 2 + i] = next_b[i]

        for kk in S.range(BLOCK_K // 2, BLOCK_K):
            acc += S.convert(a_shared[current, ty, kk], S.f32) * S.convert(
                b_shared[current, kk, tx], S.f32
            )

        if has_next:
            S.syncthreads()
            current = 1 - current

    c_flat[row, col] = S.convert(acc, S.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (BATCH, I, J, L) or tuple(B.shape) != (L, K):
            raise ValueError("ModelNew only supports the benchmark input shape.")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise ValueError("ModelNew expects bfloat16 inputs.")

        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((BATCH, I, J, K), device=A.device, dtype=A.dtype)
        einsum4d_pipelined_kernel[lambda: ((K // BLOCK_N, M // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))](
            A, B, C, num_warps=4
        )
        return C
