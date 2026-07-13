import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def conv3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    KS: al.constexpr,
    WT_SIZE: al.constexpr,
    BLOCK_SIZE: al.constexpr,
):
    # Shared memory for the full weight matrix
    wt_shared = al.make_shared((WT_SIZE,), al.bf16)

    # 1D flattened tensor views
    in_total = B * IC * D * H * W
    out_total = B * OC * D_out * H_out * W_out

    input_t = al.make_tensor(input_ptr, al.bf16, al.make_layout((in_total,), (1,)))
    weight_t = al.make_tensor(weight_ptr, al.bf16, al.make_layout((WT_SIZE,), (1,)))
    output_t = al.make_tensor(output_ptr, al.bf16, al.make_layout((out_total,), (1,)))

    # Cooperative load of weights into shared memory
    tid = al.thread_id(0)
    for i in al.range(tid, WT_SIZE, BLOCK_SIZE):
        wt_shared[i] = weight_t[i]

    al.syncthreads()

    # Compute output elements
    gid = al.block_id(0) * BLOCK_SIZE + tid

    if gid < out_total:
        # Decode linear index -> (b, oc, d_out, h_out, w_out)
        tmp = gid
        w_out_idx = tmp % W_out
        tmp = tmp // W_out
        h_out_idx = tmp % H_out
        tmp = tmp // H_out
        d_out_idx = tmp % D_out
        tmp = tmp // D_out
        oc = tmp % OC
        b = tmp // OC

        # Strides
        in_B_stride = IC * D * H * W
        in_IC_stride = D * H * W
        in_D_stride = H * W
        in_H_stride = W

        wt_OC_stride = IC * KS * KS * KS
        wt_IC_stride = KS * KS * KS
        wt_KD_stride = KS * KS
        wt_KH_stride = KS

        # FP32 accumulator
        acc = al.convert(0.0, al.f32)

        in_base = b * in_B_stride
        wt_base = oc * wt_OC_stride

        for ic in al.range(IC):
            in_ic_off = in_base + ic * in_IC_stride
            wt_ic_off = wt_base + ic * wt_IC_stride

            for kd in al.range(KS):
                in_d_off = in_ic_off + (d_out_idx + kd) * in_D_stride
                wt_kd_off = wt_ic_off + kd * wt_KD_stride

                for kh in al.range(KS):
                    in_h_off = in_d_off + (h_out_idx + kh) * in_H_stride
                    wt_kh_off = wt_kd_off + kh * wt_KH_stride

                    for kw in al.range(KS):
                        in_idx = in_h_off + (w_out_idx + kw)
                        wt_idx = wt_kh_off + kw

                        in_val = al.convert(input_t[in_idx], al.f32)
                        wt_val = al.convert(wt_shared[wt_idx], al.f32)
                        acc = acc + in_val * wt_val

        output_t[gid] = al.convert(acc, al.bf16)


def _compute_output_size(
    in_size: int, kernel: int, stride: int, padding: int, dilation: int
) -> int:
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def conv3d_avelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    B, IC, D, H, W = x.shape
    OC, _, KD, KH, KW = weight.shape
    KS = KD

    D_out = _compute_output_size(D, KS, stride, padding, dilation)
    H_out = _compute_output_size(H, KS, stride, padding, dilation)
    W_out = _compute_output_size(W, KS, stride, padding, dilation)

    x_bf16 = x.contiguous().to(torch.bfloat16)
    w_bf16 = weight.contiguous().to(torch.bfloat16)
    out_bf16 = torch.empty(
        B, OC, D_out, H_out, W_out, dtype=torch.bfloat16, device=x.device
    )

    out_total = B * OC * D_out * H_out * W_out
    WT_SIZE = OC * IC * KS * KS * KS
    BLOCK_SIZE = 256
    grid_x = (out_total + BLOCK_SIZE - 1) // BLOCK_SIZE

    conv3d_kernel[lambda: ((grid_x, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16,
        w_bf16,
        out_bf16,
        B,
        IC,
        OC,
        D,
        H,
        W,
        D_out,
        H_out,
        W_out,
        KS,
        WT_SIZE,
        BLOCK_SIZE,
    )

    return out_bf16


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
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv3d.weight
        return conv3d_avelang(x, weight, self.stride, self.padding, self.dilation)
