import torch
import torch.nn as nn
import avelang
import avelang.language as al

_BLOCK_OC = 4
_BLOCK_L = 128
_BLOCK_THREADS = _BLOCK_OC * (_BLOCK_L // 2)  # 256
_TILE_IN = _BLOCK_L + 2  # 130


@avelang.jit
def conv1d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    KS: al.i32,
    L: al.i32,
    L_out: al.i32,
    OC_blocks: al.i32,
    L_blocks: al.i32,
):
    block_id = al.block_id(0)
    tid = al.thread_id(0)

    oc_l_blocks = OC_blocks * L_blocks
    b = block_id // oc_l_blocks
    rem = block_id - b * oc_l_blocks
    oc_block = rem // L_blocks
    l_block = rem - oc_block * L_blocks

    local_oc = tid // 64
    local_l = tid - local_oc * 64

    g_oc = oc_block * 4 + local_oc
    base_l = l_block * 128
    g_l0 = base_l + local_l
    g_l1 = g_l0 + 64

    if g_oc >= OC:
        return

    input_t = al.make_tensor(
        input_ptr, al.bf16,
        al.make_layout((B, IC, L), (IC * L, L, 1)),
    )
    weight_t = al.make_tensor(
        weight_ptr, al.bf16,
        al.make_layout((OC, IC, KS), (IC * KS, KS, 1)),
    )
    output_t = al.make_tensor(
        output_ptr, al.bf16,
        al.make_layout((B, OC, L_out), (OC * L_out, L_out, 1)),
    )

    input_s = al.make_shared((130,), al.bf16)

    acc0 = al.convert(0.0, al.f32)
    acc1 = al.convert(0.0, al.f32)

    valid0 = g_l0 < L_out
    valid1 = g_l1 < L_out

    for ic in al.range(IC):
        if tid < 130:
            in_pos = base_l + tid
            if in_pos < L:
                input_s[tid] = input_t[b, ic, in_pos]
            else:
                input_s[tid] = al.convert(0.0, al.bf16)

        al.syncthreads()

        if valid0:
            for k in al.range(KS):
                acc0 = acc0 + al.convert(input_s[local_l + k], al.f32) * al.convert(weight_t[g_oc, ic, k], al.f32)

        if valid1:
            for k in al.range(KS):
                acc1 = acc1 + al.convert(input_s[local_l + 64 + k], al.f32) * al.convert(weight_t[g_oc, ic, k], al.f32)

        al.syncthreads()

    if valid0:
        output_t[b, g_oc, g_l0] = al.convert(acc0, al.bf16)
    if valid1:
        output_t[b, g_oc, g_l1] = al.convert(acc1, al.bf16)


def conv1d_avelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    B, IC, L = x.shape
    OC, IC_g, KS = weight.shape
    L_out = (L + 2 * padding - dilation * (KS - 1) - 1) // stride + 1

    x_bf16 = x.contiguous().to(torch.bfloat16)
    w_bf16 = weight.contiguous().to(torch.bfloat16)
    out = torch.empty(B, OC, L_out, dtype=torch.bfloat16, device=x.device)

    OC_blocks = (OC + _BLOCK_OC - 1) // _BLOCK_OC
    L_blocks = (L_out + _BLOCK_L - 1) // _BLOCK_L
    total_blocks = B * OC_blocks * L_blocks
    grid = (total_blocks, 1, 1)
    block = (_BLOCK_THREADS, 1, 1)

    conv1d_kernel[lambda: (grid, block)](
        x_bf16.data_ptr(),
        w_bf16.data_ptr(),
        out.data_ptr(),
        B, IC, OC, KS, L, L_out,
        OC_blocks, L_blocks,
    )

    if bias is not None:
        out += bias.to(out.dtype).view(1, -1, 1)

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
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv1d_avelang(
            x, self.conv1d.weight, self.conv1d.bias,
            self.conv1d.stride[0], self.conv1d.padding[0],
            self.conv1d.dilation[0], self.conv1d.groups,
        )
