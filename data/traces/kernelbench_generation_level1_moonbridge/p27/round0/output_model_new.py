import torch
import torch.nn as nn
import avelang
import avelang.language as al

# SELU constants from Klambauer et al. (2017)
_SELU_ALPHA = 1.6732632423543772848170429916717
_SELU_SCALE = 1.0507009873554804934193349852946

BLOCK_SIZE = 256
MAX_BLOCKS = 65536


@avelang.jit
def selu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
):
    """
    SELU activation: scale * where(x > 0, x, alpha * (exp(x) - 1))

    Processes 4 bf16 elements per loop iteration via u32-packed access
    for improved memory bandwidth utilization.
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)
    grid_size = al.grid_dim(0)

    # 4 bf16 per iteration = 2 u32 loads/stores
    quarter = numel // 4
    layout_u32 = al.make_layout((numel // 2,), (1,))
    x_u32 = al.make_tensor(x_ptr, al.u32, layout_u32)
    out_u32 = al.make_tensor(out_ptr, al.u32, layout_u32)

    alpha = al.convert(_SELU_ALPHA, al.f32)
    scale = al.convert(_SELU_SCALE, al.f32)
    zero = al.convert(0.0, al.f32)
    one = al.convert(1.0, al.f32)

    idx = bid * BLOCK_SIZE + tid
    stride = BLOCK_SIZE * grid_size

    for i in al.range(idx, quarter, stride):
        base = i * 2  # u32 index for 4 bf16 values
        p0 = x_u32[base]
        p1 = x_u32[base + 1]

        pair0 = al.view(p0, al.Tensor((2,), al.bf16))
        pair1 = al.view(p1, al.Tensor((2,), al.bf16))

        v0 = al.convert(pair0[0], al.f32)
        v1 = al.convert(pair0[1], al.f32)
        v2 = al.convert(pair1[0], al.f32)
        v3 = al.convert(pair1[1], al.f32)

        e0 = al.exp(v0)
        e1 = al.exp(v1)
        e2 = al.exp(v2)
        e3 = al.exp(v3)

        r0 = al.select(v0 > zero, scale * v0, scale * alpha * (e0 - one))
        r1 = al.select(v1 > zero, scale * v1, scale * alpha * (e1 - one))
        r2 = al.select(v2 > zero, scale * v2, scale * alpha * (e2 - one))
        r3 = al.select(v3 > zero, scale * v3, scale * alpha * (e3 - one))

        op0 = al.make_local((2,), al.bf16)
        op0[0] = al.convert(r0, al.bf16)
        op0[1] = al.convert(r1, al.bf16)
        out_u32[base] = al.view(op0, al.Tensor((1,), al.u32))[0]

        op1 = al.make_local((2,), al.bf16)
        op1[0] = al.convert(r2, al.bf16)
        op1[1] = al.convert(r3, al.bf16)
        out_u32[base + 1] = al.view(op1, al.Tensor((1,), al.u32))[0]


def avelang_selu(x: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = x.contiguous().to(dtype=torch.bfloat16)
    numel = x_bf16.numel()

    out = torch.empty_like(x_bf16)

    # Each thread processes 4 bf16 values per iteration
    num_blocks = (numel // 4 + BLOCK_SIZE - 1) // BLOCK_SIZE
    if num_blocks > MAX_BLOCKS:
        num_blocks = MAX_BLOCKS

    selu_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, out, numel
    )

    return out.to(dtype=x.dtype)


class ModelNew(nn.Module):
    """
    Simple model that performs a SELU activation using an AveLang DSL kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_selu(x)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
