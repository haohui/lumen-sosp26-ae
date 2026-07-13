import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def layernorm_gelu_scale_kernel(
    in_ptr: al.Pointer(al.f32),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.constexpr,
    C: al.constexpr,
    D: al.constexpr,
    H: al.constexpr,
    W: al.constexpr,
    EPS: al.constexpr,
    SCALING_FACTOR: al.constexpr,
):
    sN = C * D * H * W
    sC = D * H * W
    sD = H * W
    sH = W
    in_layout = al.make_layout((N, C, D, H, W), (sN, sC, sD, sH, 1))
    x = al.make_tensor(in_ptr, al.f32, in_layout)

    wt_layout = al.make_layout((W,), (1,))
    weight = al.make_tensor(weight_ptr, al.bf16, wt_layout)
    bias_t = al.make_tensor(bias_ptr, al.bf16, wt_layout)

    out_layout = al.make_layout((N, C, D, H, W), (sN, sC, sD, sH, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    gid = al.block_id(0)
    w = al.thread_id(0)

    n = gid // (C * D * H)
    rem = gid % (C * D * H)
    c = rem // (D * H)
    rem2 = rem % (D * H)
    d = rem2 // H
    h = rem2 % H

    val = x[n, c, d, h, w]

    # Shared memory reduction for mean
    smem = al.make_shared((W,), al.f32)
    smem[w] = val
    al.syncthreads()

    if w < 32:
        smem[w] = smem[w] + smem[w + 32]
    al.syncthreads()
    if w < 16:
        smem[w] = smem[w] + smem[w + 16]
    al.syncthreads()
    if w < 8:
        smem[w] = smem[w] + smem[w + 8]
    al.syncthreads()
    if w < 4:
        smem[w] = smem[w] + smem[w + 4]
    al.syncthreads()
    if w < 2:
        smem[w] = smem[w] + smem[w + 2]
    al.syncthreads()
    if w < 1:
        smem[w] = smem[w] + smem[w + 1]
    al.syncthreads()

    mean = smem[0] / al.convert(W, al.f32)

    diff = val - mean
    smem[w] = diff * diff
    al.syncthreads()

    if w < 32:
        smem[w] = smem[w] + smem[w + 32]
    al.syncthreads()
    if w < 16:
        smem[w] = smem[w] + smem[w + 16]
    al.syncthreads()
    if w < 8:
        smem[w] = smem[w] + smem[w + 8]
    al.syncthreads()
    if w < 4:
        smem[w] = smem[w] + smem[w + 4]
    al.syncthreads()
    if w < 2:
        smem[w] = smem[w] + smem[w + 2]
    al.syncthreads()
    if w < 1:
        smem[w] = smem[w] + smem[w + 1]
    al.syncthreads()

    var_val = smem[0] / al.convert(W, al.f32)

    inv_std = al.convert(1.0, al.f32) / al.sqrt(var_val + al.convert(EPS, al.f32))
    norm_val = diff * inv_std

    w_val = al.convert(weight[w], al.f32)
    b_val = al.convert(bias_t[w], al.f32)
    norm_val = norm_val * w_val + b_val

    x3 = norm_val * norm_val * norm_val
    inner = al.convert(0.7978845608028654, al.f32) * (norm_val + al.convert(0.044715, al.f32) * x3)
    gelu_val = al.convert(0.5, al.f32) * norm_val * (al.convert(1.0, al.f32) + al.tanh(inner))

    result = gelu_val * al.convert(SCALING_FACTOR, al.f32)
    out[n, c, d, h, w] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor
        self._eps = eps

    def forward(self, x):
        N, C_in, D_in, H_in, W_in = x.shape
        device = x.device

        C_out = self.conv_transpose.out_channels
        kd, kh, kw = self.conv_transpose.kernel_size
        stride_val = self.conv_transpose.stride[0]
        padding_val = self.conv_transpose.padding[0]

        D_out = (D_in - 1) * stride_val - 2 * padding_val + kd
        H_out = (H_in - 1) * stride_val - 2 * padding_val + kh
        W_out = (W_in - 1) * stride_val - 2 * padding_val + kw

        # Use PyTorch conv_transpose3d for the convolution
        conv_out = self.conv_transpose(x)
        y = conv_out.float().contiguous()

        ln_w = self.layer_norm.weight.data.contiguous()
        ln_b = self.layer_norm.bias.data.contiguous()

        z = torch.empty(N, C_out, D_out, H_out, W_out,
                        device=device, dtype=x.dtype)

        grid = N * C_out * D_out * H_out
        layernorm_gelu_scale_kernel[lambda: ((grid, 1, 1), (W_out, 1, 1))](
            y, ln_w, ln_b, z,
            N, C_out, D_out, H_out, W_out,
            self._eps, self.scaling_factor,
        )

        return z
