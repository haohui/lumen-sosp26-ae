import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def elu_kernel(
    x: al.Pointer(al.bf16),
    out: al.Pointer(al.bf16),
    numel: al.i32,
    alpha: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_size = al.block_dim(0)
    grid_size = al.grid_dim(0)

    # Create u32 view: each u32 word packs 2 BF16 values for 2x load throughput
    words = numel // 2
    u32_layout = al.make_layout((words,), (1,))
    x_u32 = al.make_tensor(x, al.u32, u32_layout)

    # Output as scalar BF16
    bf16_layout = al.make_layout((numel,), (1,))
    out_view = al.make_tensor(out, al.bf16, bf16_layout)

    idx = bid * block_size + tid
    stride = grid_size * block_size
    zero_f32 = al.convert(0.0, al.f32)
    one_f32 = al.convert(1.0, al.f32)

    # Main loop: process 2 BF16 elements per u32 load
    for i in al.range(idx, words, stride):
        w = x_u32[i]
        pair = al.view(w, al.Tensor((2,), al.bf16))
        v0 = pair[0]
        v1 = pair[1]

        f0 = al.convert(v0, al.f32)
        if f0 > zero_f32:
            out_view[i * 2] = v0
        else:
            out_view[i * 2] = al.convert(alpha * (al.exp(f0) - one_f32), al.bf16)

        f1 = al.convert(v1, al.f32)
        if f1 > zero_f32:
            out_view[i * 2 + 1] = v1
        else:
            out_view[i * 2 + 1] = al.convert(alpha * (al.exp(f1) - one_f32), al.bf16)


def avelang_elu(x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    x_bf16 = x.to(torch.bfloat16).contiguous()
    out = torch.empty_like(x_bf16)
    numel = x_bf16.numel()

    BLOCK_SIZE = 256
    MAX_GRID = 65536
    grid_size = min((numel + BLOCK_SIZE - 1) // BLOCK_SIZE, MAX_GRID)

    elu_kernel[lambda: ((grid_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, numel, alpha
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_elu(x, self.alpha)
