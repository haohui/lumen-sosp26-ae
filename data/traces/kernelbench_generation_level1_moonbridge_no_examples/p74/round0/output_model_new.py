import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv_transpose1d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.constexpr,
    C_in: al.constexpr,
    C_out: al.constexpr,
    L: al.constexpr,
    L_out: al.constexpr,
    K: al.constexpr,
    dilation: al.constexpr,
):
    """2D grid: each block handles all L_out for one (b, oc).
    Weight slice loaded into shared memory once per block."""
    b = al.block_id(0)
    oc = al.block_id(1)
    tid = al.thread_id(0)
    BLOCK_SIZE = 256

    in_layout = al.make_layout((B, C_in, L), (C_in * L, L, 1))
    inp = al.make_tensor(input_ptr, al.bf16, in_layout)

    wt_layout = al.make_layout((C_in, C_out, K), (C_out * K, K, 1))
    wt = al.make_tensor(weight_ptr, al.bf16, wt_layout)

    out_layout = al.make_layout((B, C_out, L_out), (C_out * L_out, L_out, 1))
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    # Load per-block weight slice (C_in x K) into shared memory.
    wt_slice = al.make_shared((C_in, K), al.bf16)
    wt_total = C_in * K
    for i in al.range(tid, wt_total, BLOCK_SIZE):
        ic = i // K
        k = i - ic * K
        wt_slice[ic, k] = wt[ic, oc, k]
    al.syncthreads()

    for po in al.range(tid, L_out, BLOCK_SIZE):
        acc = al.convert(0.0, al.f32)
        for k in al.range(K):
            kd = k * dilation
            if po >= kd:
                ipos = po - kd
                if ipos < L:
                    for ic in al.range(C_in):
                        in_val = al.convert(inp[b, ic, ipos], al.f32)
                        w_val = al.convert(wt_slice[ic, k], al.f32)
                        acc = acc + in_val * w_val
        out[b, oc, po] = al.convert(acc, al.bf16)


def avelang_conv_transpose1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    B, C_in, L = x.shape
    C_in_w, C_out, K = weight.shape
    if C_in != C_in_w:
        raise ValueError(f"Input channels mismatch: x has {C_in}, weight has {C_in_w}")

    L_out = (L - 1) * stride - 2 * padding + dilation * (K - 1) + 1

    out = torch.zeros(B, C_out, L_out, device=x.device, dtype=torch.bfloat16)

    BLOCK_SIZE = 256
    grid = (B, C_out, 1)
    block = (BLOCK_SIZE, 1, 1)

    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = weight.to(torch.bfloat16).contiguous()

    conv_transpose1d_kernel[lambda: (grid, block)](
        x_bf16,
        w_bf16,
        out,
        B,
        C_in,
        C_out,
        L,
        L_out,
        K,
        dilation,
        num_warps=8,
    )

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_transpose1d(
            x, self.weight, self.stride, self.padding, self.dilation
        )
