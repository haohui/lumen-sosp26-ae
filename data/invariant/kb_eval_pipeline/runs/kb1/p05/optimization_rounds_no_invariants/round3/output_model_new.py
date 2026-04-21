import torch
import torch.nn as nn

import substrate
import substrate.language as S


M = 65536
N = 16384
TOTAL_ELEMS = M * N
ELEMENT_BYTES = 2
PACKED_ELEMS = 8
PACKED_BYTES = 16
TOTAL_BYTES = TOTAL_ELEMS * ELEMENT_BYTES
BLOCK_THREADS = 256
GRID_BLOCKS = 1024


@substrate.jit
def scale_kernel_pipelined(
    A: S.Tensor((65536, 16384), S.bf16),
    C: S.Tensor((65536, 16384), S.bf16),
    scalar_bits: S.u32,
):
    tid = S.thread_id(0)
    bid = S.block_id(0)
    grid = S.grid_dim(0)
    scalar = S.convert(S.bitcast(scalar_bits, S.f32), S.bf16)

    a_flat = S.view(A, S.Tensor((TOTAL_ELEMS,), S.bf16))
    c_flat = S.view(C, S.Tensor((TOTAL_ELEMS,), S.bf16))
    a_rsrc = S.amdgpu.make_rsrc(a_flat, TOTAL_BYTES)
    c_rsrc = S.amdgpu.make_rsrc(c_flat, TOTAL_BYTES)

    packed0 = S.make_local((4,), S.u32)
    packed1 = S.make_local((4,), S.u32)
    vals0 = S.view(packed0, S.Tensor((PACKED_ELEMS,), S.bf16))
    vals1 = S.view(packed1, S.Tensor((PACKED_ELEMS,), S.bf16))

    byte0 = (bid * S.block_dim(0) + tid) * PACKED_BYTES
    stride_bytes = S.block_dim(0) * grid * PACKED_BYTES
    byte1 = byte0 + stride_bytes

    while byte0 < TOTAL_BYTES:
        load0 = S.amdgpu.raw_buffer_load_x4(a_rsrc, byte0, 0, 0)
        packed0[0] = load0[0]
        packed0[1] = load0[1]
        packed0[2] = load0[2]
        packed0[3] = load0[3]
        vals0[0] = vals0[0] * scalar
        vals0[1] = vals0[1] * scalar
        vals0[2] = vals0[2] * scalar
        vals0[3] = vals0[3] * scalar
        vals0[4] = vals0[4] * scalar
        vals0[5] = vals0[5] * scalar
        vals0[6] = vals0[6] * scalar
        vals0[7] = vals0[7] * scalar
        S.amdgpu.raw_buffer_store_x4(packed0, c_rsrc, byte0, 0, 0)

        load1 = S.amdgpu.raw_buffer_load_x4(a_rsrc, byte1, 0, 0)
        packed1[0] = load1[0]
        packed1[1] = load1[1]
        packed1[2] = load1[2]
        packed1[3] = load1[3]
        vals1[0] = vals1[0] * scalar
        vals1[1] = vals1[1] * scalar
        vals1[2] = vals1[2] * scalar
        vals1[3] = vals1[3] * scalar
        vals1[4] = vals1[4] * scalar
        vals1[5] = vals1[5] * scalar
        vals1[6] = vals1[6] * scalar
        vals1[7] = vals1[7] * scalar
        S.amdgpu.raw_buffer_store_x4(packed1, c_rsrc, byte1, 0, 0)

        byte0 = byte0 + 2 * stride_bytes
        byte1 = byte1 + 2 * stride_bytes


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if tuple(A.shape) != (M, N):
            return A * B
        A = A.contiguous()
        scalar = torch.tensor(float(B), device="cpu", dtype=torch.float32).view(torch.int32).item()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        scale_kernel_pipelined[lambda: ((GRID_BLOCKS, 1, 1), (BLOCK_THREADS, 1, 1))](
            A,
            C,
            scalar,
            num_warps=4,
        )
        return C
