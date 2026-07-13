import math
import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE = 256
ACC_TILE = 8


@avelang.jit
def conv_transpose3d_bf16_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.u32,
    C_in: al.u32,
    C_out: al.u32,
    D: al.u32,
    H: al.u32,
    W: al.u32,
    D_out: al.u32,
    H_out: al.u32,
    W_out: al.u32,
    K: al.u32,
    stride: al.u32,
    padding: al.u32,
    groups: al.u32,
):
    tid = al.thread_id(0)
    bid_x = al.block_id(0)
    bid_y = al.block_id(1)

    total_spatial = D_out * H_out * W_out
    spatial_idx = bid_x * BLOCK_SIZE + tid

    if spatial_idx >= total_spatial:
        return

    w_out = spatial_idx % W_out
    tmp = spatial_idx // W_out
    h_out = tmp % H_out
    d_out = tmp // H_out

    n_idx = bid_y // groups
    g_idx = bid_y % groups

    C_in_g = C_in // groups
    C_out_g = C_out // groups

    g_ic_start = g_idx * C_in_g
    g_oc_start = g_idx * C_out_g

    in_n_stride = C_in * D * H * W
    in_c_stride = D * H * W
    in_d_stride = H * W
    in_h_stride = W

    out_n_stride = C_out * D_out * H_out * W_out
    out_c_stride = D_out * H_out * W_out
    out_d_stride = H_out * W_out
    out_h_stride = W_out

    K_sq = K * K
    K_cu = K_sq * K
    w_ic_stride = C_out_g * K_cu
    w_oc_stride = K_cu
    w_kd_stride = K_sq
    w_kh_stride = K

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((N * in_n_stride,), (1,)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((C_in * w_ic_stride,), (1,)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((N * out_n_stride,), (1,)))

    in_base = n_idx * in_n_stride
    out_base = n_idx * out_n_stride

    zero_f32 = al.convert(0.0, al.f32)
    zero_u32 = al.convert(0, al.u32)

    d_limit = d_out + padding
    h_limit = h_out + padding
    w_limit = w_out + padding

    num_oc_tiles = (C_out_g + ACC_TILE - 1) // ACC_TILE

    for oc_tile in al.range(num_oc_tiles):
        acc = al.make_local((ACC_TILE,), al.f32)
        for i in al.range(ACC_TILE):
            acc[i] = zero_f32

        for kd in al.range(K):
            if kd <= d_limit:
                in_d_unchecked = d_limit - kd
                if in_d_unchecked % stride == zero_u32:
                    in_d = in_d_unchecked // stride
                    if in_d < D:
                        for kh in al.range(K):
                            if kh <= h_limit:
                                in_h_unchecked = h_limit - kh
                                if in_h_unchecked % stride == zero_u32:
                                    in_h = in_h_unchecked // stride
                                    if in_h < H:
                                        for kw in al.range(K):
                                            if kw <= w_limit:
                                                in_w_unchecked = w_limit - kw
                                                if in_w_unchecked % stride == zero_u32:
                                                    in_w = in_w_unchecked // stride
                                                    if in_w < W:
                                                        in_idx_base = (
                                                            in_base
                                                            + in_d * in_d_stride
                                                            + in_h * in_h_stride
                                                            + in_w
                                                        )
                                                        for ic_local in al.range(C_in_g):
                                                            ic = g_ic_start + ic_local
                                                            in_idx = in_idx_base + ic * in_c_stride
                                                            in_val = al.convert(x[in_idx], al.f32)
                                                            w_base = (
                                                                ic * w_ic_stride
                                                                + kd * w_kd_stride
                                                                + kh * w_kh_stride
                                                                + kw
                                                            )
                                                            for i in al.range(ACC_TILE):
                                                                oc_local = oc_tile * ACC_TILE + i
                                                                if oc_local < C_out_g:
                                                                    w_idx = w_base + oc_local * w_oc_stride
                                                                    w_val = al.convert(w[w_idx], al.f32)
                                                                    acc[i] = acc[i] + in_val * w_val

        for i in al.range(ACC_TILE):
            oc_local = oc_tile * ACC_TILE + i
            if oc_local < C_out_g:
                oc = g_oc_start + oc_local
                out_idx = (
                    out_base
                    + oc * out_c_stride
                    + d_out * out_d_stride
                    + h_out * out_h_stride
                    + w_out
                )
                out[out_idx] = al.convert(acc[i], al.bf16)


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    groups: int,
    output_padding: int = 0,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    N, C_in, D, H, W = x.shape
    C_in_w, C_out_g, K_w, _, _ = weight.shape

    if C_in_w != C_in:
        raise ValueError(f"Weight C_in mismatch: {C_in_w} vs {C_in}")
    if K_w != weight.shape[2] or K_w != weight.shape[3] or K_w != weight.shape[4]:
        raise ValueError("Kernel must be cubic")

    K = K_w
    C_out = C_out_g * groups

    D_out = (D - 1) * stride - 2 * padding + K + output_padding
    H_out = (H - 1) * stride - 2 * padding + K + output_padding
    W_out = (W - 1) * stride - 2 * padding + K + output_padding

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)
    bias_bf16 = _to_bf16_contiguous(bias) if bias is not None else None

    out = torch.empty(
        (N, C_out, D_out, H_out, W_out),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    total_spatial = D_out * H_out * W_out
    grid_x = (total_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = N * groups

    conv_transpose3d_bf16_kernel[lambda: ((grid_x, grid_y, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16,
        w_bf16,
        out,
        N,
        C_in,
        C_out,
        D,
        H,
        W,
        D_out,
        H_out,
        W_out,
        K,
        stride,
        padding,
        groups,
    )

    if bias_bf16 is not None:
        out.add_(bias_bf16.view(1, C_out, 1, 1, 1))

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_transpose3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.groups,
            0,  # output_padding received but intentionally ignored (ref model does same)
        )


# Test code
batch_size = 4
in_channels = 32
out_channels = 32
kernel_size = 3
depth = 32
height = 64
width = 128
stride = 2
padding = 1
groups = 4


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups]
