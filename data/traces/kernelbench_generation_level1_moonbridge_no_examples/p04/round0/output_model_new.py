import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def gemv_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K: al.i32,
    BLOCK_K: al.constexpr,
):
    row = al.block_id(0)
    tid = al.thread_id(0)

    if row >= M:
        return

    A_layout = al.make_layout((M, K), (K, 1))
    A = al.make_tensor(A_ptr, al.bf16, A_layout)

    B_layout = al.make_layout((K, 1), (1, 1))
    B = al.make_tensor(B_ptr, al.bf16, B_layout)

    # FP32 accumulator — each thread processes 4 elements per iteration
    acc = al.convert(0.0, al.f32)
    stride = BLOCK_K * 4
    for k in al.range(tid * 4, K, stride):
        a0 = al.convert(A[row, k + 0], al.f32)
        a1 = al.convert(A[row, k + 1], al.f32)
        a2 = al.convert(A[row, k + 2], al.f32)
        a3 = al.convert(A[row, k + 3], al.f32)
        b0 = al.convert(B[k + 0, 0], al.f32)
        b1 = al.convert(B[k + 1, 0], al.f32)
        b2 = al.convert(B[k + 2, 0], al.f32)
        b3 = al.convert(B[k + 3, 0], al.f32)
        acc = acc + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3

    # Shared-memory tree reduction (8 steps for BLOCK_K=256)
    partial = al.make_shared((BLOCK_K,), al.f32)
    partial[tid] = acc
    al.syncthreads()

    if tid < 128:
        partial[tid] = partial[tid] + partial[tid + 128]
    al.syncthreads()
    if tid < 64:
        partial[tid] = partial[tid] + partial[tid + 64]
    al.syncthreads()
    if tid < 32:
        partial[tid] = partial[tid] + partial[tid + 32]
    al.syncthreads()
    if tid < 16:
        partial[tid] = partial[tid] + partial[tid + 16]
    al.syncthreads()
    if tid < 8:
        partial[tid] = partial[tid] + partial[tid + 8]
    al.syncthreads()
    if tid < 4:
        partial[tid] = partial[tid] + partial[tid + 4]
    al.syncthreads()
    if tid < 2:
        partial[tid] = partial[tid] + partial[tid + 2]
    al.syncthreads()
    if tid < 1:
        partial[tid] = partial[tid] + partial[tid + 1]
    al.syncthreads()

    if tid == 0:
        C_layout = al.make_layout((M, 1), (1, 1))
        C = al.make_tensor(C_ptr, al.bf16, C_layout)
        C[row, 0] = al.convert(partial[0], al.bf16)


def avelang_gemv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    A = A.contiguous().to(torch.bfloat16)
    B = B.contiguous().to(torch.bfloat16)

    M, K = A.shape

    C = torch.empty(M, 1, dtype=torch.bfloat16, device=A.device)

    BLOCK_K = 256

    gemv_kernel[lambda: ((M, 1, 1), (BLOCK_K, 1, 1))](
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        M, K, BLOCK_K,
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return avelang_gemv(A, B)
