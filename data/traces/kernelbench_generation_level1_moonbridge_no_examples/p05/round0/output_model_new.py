import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256
ELEMS_PER_THREAD = 256


@avelang.jit
def mat_scalar_mul_kernel(
    a_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_elements: al.i32,
    s: al.f32,
):
    tid = al.block_id(0) * al.block_dim(0) + al.thread_id(0)
    stride = al.grid_dim(0) * al.block_dim(0)

    layout = al.make_layout((total_elements,), (1,))
    a = al.make_tensor(a_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    for i in al.range(tid, total_elements, stride):
        val = al.convert(a[i], al.f32)
        out[i] = al.convert(val * s, al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        A = A.contiguous().to(torch.bfloat16)
        M_cur, N_cur = A.shape
        total_elements = M_cur * N_cur

        out = torch.empty_like(A)

        work_per_block = BLOCK_SIZE * ELEMS_PER_THREAD
        grid_x = (total_elements + work_per_block - 1) // work_per_block
        grid = (grid_x, 1, 1)
        block = (BLOCK_SIZE, 1, 1)

        mat_scalar_mul_kernel[lambda: (grid, block)](A, out, total_elements, s)

        return out
