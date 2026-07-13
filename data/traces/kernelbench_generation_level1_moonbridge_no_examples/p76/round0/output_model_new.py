import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv1d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B_IC: al.i32,
    OC: al.i32,
    IC_KS: al.i32,
    L: al.i32,
    OL: al.i32,
    B_OC: al.i32,
    IC: al.i32,
    B: al.i32,
    stride: al.i32,
    dilation: al.i32,
    KS: al.constexpr,
):
    """1D convolution: each thread computes one output element.
    Input layout (B*IC, L), weight (OC, IC*KS), output (B*OC, OL).
    2D grid encodes flattened (batch, out-channel) as bo = block_id(0)*block_dim(0) + thread_id(0).
    """
    bo = al.block_id(0) * al.block_dim(0) + al.thread_id(0)
    ol = al.block_id(1) * al.block_dim(1) + al.thread_id(1)

    x_layout = al.make_layout((B_IC, L), (L, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    w_layout = al.make_layout((OC, IC_KS), (IC_KS, 1))
    wgt = al.make_tensor(w_ptr, al.bf16, w_layout)

    out_layout = al.make_layout((B_OC, OL), (OL, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    if bo < B_OC:
        if ol < OL:
            b = bo // OC
            oc = bo % OC
            accum = al.convert(0.0, al.f32)
            in_base = ol * stride
            b_ic_base = b * IC
            for ic in al.range(IC):
                x_row = b_ic_base + ic
                for k in al.range(KS):
                    x_val = al.convert(
                        x[x_row, in_base + k * dilation], al.f32
                    )
                    w_val = al.convert(wgt[oc, ic * KS + k], al.f32)
                    accum = accum + x_val * w_val
            out[bo, ol] = al.convert(accum, al.bf16)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # Match PyTorch Conv1d default weight init: kaiming_uniform.
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size)
        )
        self.reset_parameters()

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, IC, L = x.shape
        OC = self.out_channels
        KS = self.kernel_size
        stride_val = self.stride
        dilation_val = self.dilation

        OL = (L - dilation_val * (KS - 1) - 1) // stride_val + 1

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = self.weight.to(torch.bfloat16).contiguous()
        out = torch.empty(B, OC, OL, dtype=torch.bfloat16, device=x.device)

        B_IC = B * IC
        IC_KS = IC * KS
        B_OC = B * OC

        # Block size limited to 256 threads to avoid register pressure.
        BLOCK_BO = 8
        BLOCK_OL = 32

        grid_bo = (B_OC + BLOCK_BO - 1) // BLOCK_BO
        grid_ol = (OL + BLOCK_OL - 1) // BLOCK_OL

        conv1d_kernel[lambda: ((grid_bo, grid_ol, 1), (BLOCK_BO, BLOCK_OL, 1))](
            x_bf16, w_bf16, out,
            B_IC, OC, IC_KS, L, OL, B_OC,
            IC, B,
            stride_val, dilation_val,
            KS=KS,
        )

        if self.bias is not None:
            out = out + self.bias.to(torch.bfloat16).view(1, -1, 1)

        return out.to(x.dtype)
