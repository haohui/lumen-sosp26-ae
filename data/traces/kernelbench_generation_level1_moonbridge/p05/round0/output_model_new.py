import struct

import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_ELEMS = 131072
THREADS = 256


def _f32_to_i32_bits(f: float) -> int:
    return struct.unpack("<i", struct.pack("<f", f))[0]


@avelang.jit
def mat_scalar_mul_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_elements: al.i32,
    scalar_bits: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    scalar_val = al.bitcast(scalar_bits, al.f32)

    layout = al.make_layout((total_elements,), (1,))
    in_tensor = al.make_tensor(in_ptr, al.bf16, layout)
    out_tensor = al.make_tensor(out_ptr, al.bf16, layout)

    block_start = bid * TILE_ELEMS

    for v in al.range(TILE_ELEMS // THREADS):
        idx = block_start + tid + v * THREADS
        if idx < total_elements:
            val = al.convert(in_tensor[idx], al.f32)
            out_tensor[idx] = al.convert(val * scalar_val, al.bf16)


def avelang_mat_scalar_mul(A: torch.Tensor, s: float) -> torch.Tensor:
    if not A.is_cuda:
        raise RuntimeError("Input tensor must be on CUDA/HIP device.")

    original_dtype = A.dtype
    A_contiguous = A.contiguous()
    if A_contiguous.dtype != torch.bfloat16:
        A_bf16 = A_contiguous.to(torch.bfloat16)
    else:
        A_bf16 = A_contiguous

    out = torch.empty_like(A_bf16)
    total_elements = A_bf16.numel()
    num_blocks = (total_elements + TILE_ELEMS - 1) // TILE_ELEMS
    scalar_bits = _f32_to_i32_bits(s)

    mat_scalar_mul_kernel[lambda: ((num_blocks, 1, 1), (THREADS, 1, 1))](
        A_bf16, out, total_elements, scalar_bits
    )

    if out.dtype != original_dtype:
        out = out.to(original_dtype)

    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return avelang_mat_scalar_mul(A, s)
