import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def hardsigmoid_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    n: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_dim = al.block_dim(0)
    grid_dim = al.grid_dim(0)

    start = bid * block_dim + tid
    stride = grid_dim * block_dim

    layout_1d = al.make_layout((n,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, layout_1d)
    out = al.make_tensor(out_ptr, al.bf16, layout_1d)

    for i in al.range(start, n, stride):
        val = al.convert(x[i], al.f32)
        # HardSigmoid: clamp(x / 6 + 0.5, 0, 1)
        val = val / al.convert(6.0, al.f32) + al.convert(0.5, al.f32)

        zero = al.convert(0.0, al.f32)
        one = al.convert(1.0, al.f32)

        if val < zero:
            val = zero
        if val > one:
            val = one

        out[i] = al.convert(val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x_bf16 = x.to(torch.bfloat16).contiguous()
        n = x_bf16.numel()
        out = torch.empty_like(x_bf16)

        BLOCK_SIZE = 256
        GRID_SIZE = 304 * 160  # 48640 blocks for MI300X 304 CUs

        hardsigmoid_kernel[lambda: ((GRID_SIZE, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_bf16.data_ptr(), out.data_ptr(), n
        )

        return out.to(original_dtype)
