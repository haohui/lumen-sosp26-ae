import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def depthwise_conv2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    K: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    stride_val: al.i32,
    pad_val: al.i32,
    dilation_val: al.i32,
    total_elems: al.i32,
):
    x_strides = al.make_layout((N, C, H, W), (C * H * W, H * W, W, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_strides)
    w_strides = al.make_layout((C, 1, K, 1), (K, K, 1, 1))
    w = al.make_tensor(w_ptr, al.bf16, w_strides)
    o_strides = al.make_layout((N, C, H_out, W_out), (C * H_out * W_out, H_out * W_out, W_out, 1))
    out = al.make_tensor(out_ptr, al.bf16, o_strides)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    bdim = al.block_dim(0)
    gid = bid * bdim + tid

    if gid < total_elems:
        w_idx = gid % W_out
        tmp = gid // W_out
        h_idx = tmp % H_out
        tmp = tmp // H_out
        c = tmp % C
        n = tmp // C

        acc = al.convert(0.0, al.f32)
        for kh in al.range(K):
            h_in = h_idx * stride_val - pad_val + kh * dilation_val
            w_in = w_idx * stride_val - pad_val
            in_bounds = h_in >= 0
            in_bounds = in_bounds and (h_in < H)
            in_bounds = in_bounds and (w_in >= 0)
            in_bounds = in_bounds and (w_in < W)
            if in_bounds:
                xv = al.convert(x[n, c, h_in, w_in], al.f32)
                wv = al.convert(w[c, 0, kh, 0], al.f32)
                acc = acc + xv * wv

        out[n, c, h_idx, w_idx] = al.convert(acc, al.bf16)


def _depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device"
    assert weight.is_cuda, "Weight must be on CUDA/HIP device"

    N, C, H, W = x.shape
    K = weight.shape[2]
    H_out = (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (1 - 1) - 1) // stride + 1

    x_bf16 = x.contiguous().to(torch.bfloat16)
    w_bf16 = weight.contiguous().to(torch.bfloat16)
    out = torch.empty(N, C, H_out, W_out, dtype=torch.bfloat16, device=x.device)

    total_elems = N * C * H_out * W_out
    BLOCK_SIZE = 256
    grid_x = (total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE

    depthwise_conv2d_kernel[lambda: ((grid_x, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, w_bf16, out,
        N, C, H, W, K, H_out, W_out,
        stride, padding, dilation,
        total_elems,
    )
    return out.to(x.dtype)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, 1))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter("bias_param", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias_param is not None:
            fan_in = self.kernel_size
            bound = 1.0 / (fan_in ** 0.5) if fan_in != 0 else 0.0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _depthwise_conv2d(x, self.weight, self.stride, self.padding, self.dilation)
        if self.use_bias and self.bias_param is not None:
            out = out + self.bias_param.to(dtype=out.dtype, device=out.device).view(1, -1, 1, 1)
        return out
