import torch
import torch.nn as nn
import avelang
import avelang.language as al


# ---------------------------------------------------------------------------
# Kernel 1: Conv2d — 2-D grid (N, C_out).  Threads use strided spatial
# access (al.range with literal step).  3x3 kernel fully unrolled with
# literal offsets to sidestep index / i32 type conflicts.
# ---------------------------------------------------------------------------
@avelang.jit
def conv2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    H: al.i32,
    W: al.i32,
    C_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    n = al.block_id(0)
    c_out = al.block_id(1)
    tid = al.thread_id(0)

    inp = al.make_tensor(input_ptr, al.bf16,
        al.make_layout((N, C_in, H, W), (C_in * H * W, H * W, W, 1)))
    wgt = al.make_tensor(weight_ptr, al.bf16,
        al.make_layout((C_out, C_in, 3, 3), (C_in * 9, 9, 3, 1)))
    out = al.make_tensor(output_ptr, al.bf16,
        al.make_layout((N, C_out, H_out, W_out),
                       (C_out * H_out * W_out, H_out * W_out, W_out, 1)))
    bias_t = al.make_tensor(bias_ptr, al.bf16,
        al.make_layout((C_out,), (1,)))

    bias_val = al.convert(bias_t[c_out], al.f32)

    total_spatial = H_out * W_out
    for spat in al.range(tid, total_spatial, 256):
        oh = spat // W_out
        ow = spat % W_out

        acc = al.convert(0.0, al.f32)
        for ic in al.range(C_in):
            # 3x3 kernel fully unrolled — literal offsets only
            # Row 0
            acc = acc + al.convert(inp[n, ic, oh + 0, ow + 0], al.f32) * al.convert(wgt[c_out, ic, 0, 0], al.f32)
            acc = acc + al.convert(inp[n, ic, oh + 0, ow + 1], al.f32) * al.convert(wgt[c_out, ic, 0, 1], al.f32)
            acc = acc + al.convert(inp[n, ic, oh + 0, ow + 2], al.f32) * al.convert(wgt[c_out, ic, 0, 2], al.f32)
            # Row 1
            acc = acc + al.convert(inp[n, ic, oh + 1, ow + 0], al.f32) * al.convert(wgt[c_out, ic, 1, 0], al.f32)
            acc = acc + al.convert(inp[n, ic, oh + 1, ow + 1], al.f32) * al.convert(wgt[c_out, ic, 1, 1], al.f32)
            acc = acc + al.convert(inp[n, ic, oh + 1, ow + 2], al.f32) * al.convert(wgt[c_out, ic, 1, 2], al.f32)
            # Row 2
            acc = acc + al.convert(inp[n, ic, oh + 2, ow + 0], al.f32) * al.convert(wgt[c_out, ic, 2, 0], al.f32)
            acc = acc + al.convert(inp[n, ic, oh + 2, ow + 1], al.f32) * al.convert(wgt[c_out, ic, 2, 1], al.f32)
            acc = acc + al.convert(inp[n, ic, oh + 2, ow + 2], al.f32) * al.convert(wgt[c_out, ic, 2, 2], al.f32)

        acc = acc + bias_val
        out[n, c_out, oh, ow] = al.convert(acc, al.bf16)


# ---------------------------------------------------------------------------
# Kernel 2: BatchNorm2d eval mode + scale  (strided spatial)
# ---------------------------------------------------------------------------
@avelang.jit
def batchnorm_eval_scale_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    running_mean_ptr: al.Pointer(al.bf16),
    running_var_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    scalars_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
):
    n = al.block_id(0)
    c = al.block_id(1)
    tid = al.thread_id(0)

    inp = al.make_tensor(input_ptr, al.bf16,
        al.make_layout((N, C, H, W), (C * H * W, H * W, W, 1)))
    out = al.make_tensor(output_ptr, al.bf16,
        al.make_layout((N, C, H, W), (C * H * W, H * W, W, 1)))

    mean_t = al.make_tensor(running_mean_ptr, al.bf16,
        al.make_layout((C,), (1,)))
    var_t = al.make_tensor(running_var_ptr, al.bf16,
        al.make_layout((C,), (1,)))
    gamma_t = al.make_tensor(gamma_ptr, al.bf16,
        al.make_layout((C,), (1,)))
    beta_t = al.make_tensor(beta_ptr, al.bf16,
        al.make_layout((C,), (1,)))
    scalars_t = al.make_tensor(scalars_ptr, al.bf16,
        al.make_layout((2,), (1,)))

    ch_mean = al.convert(mean_t[c], al.f32)
    ch_var = al.convert(var_t[c], al.f32)
    ch_gamma = al.convert(gamma_t[c], al.f32)
    ch_beta = al.convert(beta_t[c], al.f32)
    eps = al.convert(scalars_t[0], al.f32)
    scaling_factor = al.convert(scalars_t[1], al.f32)

    inv_std = al.convert(1.0, al.f32) / al.sqrt(ch_var + eps)

    total_spatial = H * W
    for spat in al.range(tid, total_spatial, 256):
        h = spat // W
        w = spat % W
        val = al.convert(inp[n, c, h, w], al.f32)
        normed = (val - ch_mean) * inv_std
        result = (ch_gamma * normed + ch_beta) * scaling_factor
        out[n, c, h, w] = al.convert(result, al.bf16)


# ---------------------------------------------------------------------------
# ModelNew: host wrapper
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = float(scaling_factor)
        self._in_channels = int(in_channels)
        self._out_channels = int(out_channels)
        self._kernel_size = int(kernel_size)
        self.register_buffer(
            '_scalars',
            torch.tensor([self.bn.eps, float(scaling_factor)], dtype=torch.bfloat16))

    def forward(self, x):
        N = int(x.shape[0])
        C_in = int(x.shape[1])
        H = int(x.shape[2])
        W = int(x.shape[3])
        C_out = self._out_channels
        K = self._kernel_size
        H_out = H - K + 1
        W_out = W - K + 1

        device = x.device
        in_dtype = x.dtype

        x_bf16 = x.contiguous().to(torch.bfloat16)
        w_bf16 = self.conv.weight.data.to(torch.bfloat16).contiguous()
        if self.conv.bias is not None:
            b_bf16 = self.conv.bias.data.to(torch.bfloat16).contiguous()
        else:
            b_bf16 = torch.zeros(C_out, device=device, dtype=torch.bfloat16)

        rm_bf16 = self.bn.running_mean.data.to(torch.bfloat16).contiguous()
        rv_bf16 = self.bn.running_var.data.to(torch.bfloat16).contiguous()
        bn_w_bf16 = self.bn.weight.data.to(torch.bfloat16).contiguous()
        if self.bn.bias is not None:
            bn_b_bf16 = self.bn.bias.data.to(torch.bfloat16).contiguous()
        else:
            bn_b_bf16 = torch.zeros(C_out, device=device, dtype=torch.bfloat16)

        conv_out = torch.empty(
            N, C_out, H_out, W_out, device=device, dtype=torch.bfloat16)

        BLOCK_SIZE = 256
        conv2d_kernel[lambda: ((N, C_out, 1), (BLOCK_SIZE, 1, 1))](
            x_bf16, w_bf16, b_bf16, conv_out,
            N, C_in, H, W, C_out, H_out, W_out)

        final_out = torch.empty_like(conv_out)
        batchnorm_eval_scale_kernel[lambda: ((N, C_out, 1), (BLOCK_SIZE, 1, 1))](
            conv_out, final_out,
            rm_bf16, rv_bf16, bn_w_bf16, bn_b_bf16,
            self._scalars,
            N, C_out, H_out, W_out)

        return final_out.to(in_dtype)
