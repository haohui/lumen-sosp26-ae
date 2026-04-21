import torch
import torch.nn as nn
import substrate
import substrate.language as S


M = 4096
N = 4096
THREADS = 256
ELEMS_PER_THREAD = 8
PACKED_U32_PER_THREAD = ELEMS_PER_THREAD // 2
BLOCK_ELEMS = THREADS * ELEMS_PER_THREAD
TOTAL_ELEMS = N * M
TOTAL_PACKED_U32 = TOTAL_ELEMS // 2


@substrate.jit
def diag_scale_kernel(
    a: S.Tensor((N,), S.bf16),
    b: S.Tensor((N, M), S.bf16),
    out: S.Tensor((N, M), S.bf16),
):
    block = S.block_id(0)
    lane = S.thread_id(0)

    packed_idx = (block * THREADS + lane) * PACKED_U32_PER_THREAD
    elem_idx = packed_idx * 2
    row = elem_idx // M
    scale = a[row]

    b_u32 = S.view(b, S.Tensor((TOTAL_PACKED_U32,), S.u32))
    out_u32 = S.view(out, S.Tensor((TOTAL_PACKED_U32,), S.u32))

    range_bytes = TOTAL_PACKED_U32 * 4
    byte_offset = packed_idx * 4

    b_rsrc = S.amdgpu.make_rsrc(b_u32, range_bytes)
    out_rsrc = S.amdgpu.make_rsrc(out_u32, range_bytes)

    packed_vals = S.amdgpu.raw_buffer_load_x4(b_rsrc, byte_offset, 0, 0)
    vals = S.view(packed_vals, S.Tensor((ELEMS_PER_THREAD,), S.bf16))

    scaled = S.full((ELEMS_PER_THREAD,), 0.0, S.bf16)
    for i in S.range(ELEMS_PER_THREAD):
        scaled[i] = vals[i] * scale

    packed_out = S.view(scaled, S.Tensor((PACKED_U32_PER_THREAD,), S.u32))
    S.amdgpu.raw_buffer_store_x4(packed_out, out_rsrc, byte_offset, 0, 0)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._grid = ((TOTAL_ELEMS + BLOCK_ELEMS - 1) // BLOCK_ELEMS, 1, 1)
        self._block = (THREADS, 1, 1)

    def forward(self, A, B):
        a = A.contiguous()
        b = B.contiguous()
        if tuple(a.shape) != (N,) or tuple(b.shape) != (N, M):
            return b * a.unsqueeze(1)
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            return b * a.view(N, 1)
        if not a.is_cuda or not b.is_cuda:
            return b * a.view(N, 1)

        out = torch.empty_like(b)
        diag_scale_kernel[lambda: (self._grid, self._block)](a, b, out)
        return out
