import torch
import torch.nn as nn
import avelang
import avelang.language as al


# ============================================================
# Conv2d 3x3 kernel with bias (no padding)
# ============================================================
@avelang.jit
def conv2d_3x3_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    K = al.convert(3, al.i32)
    one_i32 = al.convert(1, al.i32)

    in_layout = al.make_layout(
        (N, C_in, H_in, W_in),
        (C_in * H_in * W_in, H_in * W_in, W_in, one_i32),
    )
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    wt_layout = al.make_layout(
        (C_out, C_in, K, K),
        (C_in * K * K, K * K, K, one_i32),
    )
    weight_t = al.make_tensor(weight_ptr, al.bf16, wt_layout)

    bias_layout = al.make_layout((C_out,), (one_i32,))
    bias_t = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    out_layout = al.make_layout(
        (N, C_out, H_out, W_out),
        (C_out * H_out * W_out, H_out * W_out, W_out, one_i32),
    )
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    bdim = al.block_dim(0)
    idx = tid + bid * bdim

    total = N * C_out * H_out * W_out

    if idx < total:
        n = idx // (C_out * H_out * W_out)
        rem = idx % (C_out * H_out * W_out)
        c_out = rem // (H_out * W_out)
        rem2 = rem % (H_out * W_out)
        h = rem2 // W_out
        w = rem2 % W_out

        acc = al.convert(0.0, al.f64)

        for kh in al.range(K):
            h_in = h + kh
            for kw in al.range(K):
                w_in = w + kw
                for c_in in al.range(C_in):
                    v = input_t[n, c_in, h_in, w_in]
                    wt = weight_t[c_out, c_in, kh, kw]
                    acc = acc + al.convert(v, al.f64) * al.convert(wt, al.f64)

        acc_f32 = al.convert(acc, al.f32)
        acc_f32 = acc_f32 + al.convert(bias_t[c_out], al.f32)
        output_t[n, c_out, h, w] = al.convert(acc_f32, al.bf16)


# ============================================================
# Mish activation kernel: x * tanh(softplus(x))
# ============================================================
@avelang.jit
def mish_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
):
    one_i32 = al.convert(1, al.i32)

    in_layout = al.make_layout(
        (N, C, H, W),
        (C * H * W, H * W, W, one_i32),
    )
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    out_layout = al.make_layout(
        (N, C, H, W),
        (C * H * W, H * W, W, one_i32),
    )
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    bdim = al.block_dim(0)
    idx = tid + bid * bdim

    total = N * C * H * W

    if idx < total:
        n = idx // (C * H * W)
        rem = idx % (C * H * W)
        c = rem // (H * W)
        rem2 = rem % (H * W)
        h = rem2 // W
        w = rem2 % W

        x_val = al.convert(input_t[n, c, h, w], al.f32)

        one_f32 = al.convert(1.0, al.f32)
        exp_x = al.exp(x_val)
        sp = al.log(one_f32 + exp_x)

        tanh_sp = al.tanh(sp)
        result = x_val * tanh_sp
        output_t[n, c, h, w] = al.convert(result, al.bf16)


# ============================================================
# BN eval-mode kernel (uses running mean/variance)
# ============================================================
@avelang.jit
def bn_eval_kernel(
    input_ptr: al.Pointer(al.bf16),
    running_mean_ptr: al.Pointer(al.f32),
    running_var_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
):
    one_i32 = al.convert(1, al.i32)

    in_layout = al.make_layout(
        (N, C, H, W),
        (C * H * W, H * W, W, one_i32),
    )
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    mean_layout = al.make_layout((C,), (one_i32,))
    mean_t = al.make_tensor(running_mean_ptr, al.f32, mean_layout)

    var_layout = al.make_layout((C,), (one_i32,))
    var_t = al.make_tensor(running_var_ptr, al.f32, var_layout)

    gamma_layout = al.make_layout((C,), (one_i32,))
    gamma_t = al.make_tensor(gamma_ptr, al.bf16, gamma_layout)

    beta_layout = al.make_layout((C,), (one_i32,))
    beta_t = al.make_tensor(beta_ptr, al.bf16, beta_layout)

    out_layout = al.make_layout(
        (N, C, H, W),
        (C * H * W, H * W, W, one_i32),
    )
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    tid = al.thread_id(0)
    bid = al.block_id(0)
    bdim = al.block_dim(0)
    idx = tid + bid * bdim

    total = N * C * H * W

    if idx < total:
        n = idx // (C * H * W)
        rem = idx % (C * H * W)
        c = rem // (H * W)
        rem2 = rem % (H * W)
        h = rem2 // W
        w = rem2 % W

        x_val = al.convert(input_t[n, c, h, w], al.f32)
        mean_val = mean_t[c]
        var_val = var_t[c]
        gamma_val = al.convert(gamma_t[c], al.f32)
        beta_val = al.convert(beta_t[c], al.f32)

        eps_f32 = al.convert(1e-5, al.f32)
        one_f32 = al.convert(1.0, al.f32)
        inv_std = one_f32 / al.sqrt(var_val + eps_f32)
        normalized = (x_val - mean_val) * inv_std
        result = gamma_val * normalized + beta_val

        output_t[n, c, h, w] = al.convert(result, al.bf16)


# ============================================================
# Host wrapper
# ============================================================
def avelang_model(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
) -> torch.Tensor:
    N, C_in, H_in, W_in = x.shape
    C_out, _, K, _ = conv_weight.shape

    H_conv = H_in - K + 1
    W_conv = W_in - K + 1

    assert x.is_cuda, "Input must be on GPU"

    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = conv_weight.to(torch.bfloat16).contiguous()
    b_bf16 = conv_bias.to(torch.bfloat16).contiguous()

    conv_out = torch.empty(N, C_out, H_conv, W_conv, dtype=torch.bfloat16, device=x.device)

    BLOCK_SIZE = 256
    total_conv = N * C_out * H_conv * W_conv
    grid_conv = (total_conv + BLOCK_SIZE - 1) // BLOCK_SIZE
    conv2d_3x3_kernel[lambda: ((grid_conv, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, w_bf16, b_bf16, conv_out, N, C_in, C_out, H_in, W_in, H_conv, W_conv,
    )

    mish_out = torch.empty(N, C_out, H_conv, W_conv, dtype=torch.bfloat16, device=x.device)
    total_mish = N * C_out * H_conv * W_conv
    grid_mish = (total_mish + BLOCK_SIZE - 1) // BLOCK_SIZE
    mish_kernel[lambda: ((grid_mish, 1, 1), (BLOCK_SIZE, 1, 1))](
        conv_out, mish_out, N, C_out, H_conv, W_conv,
    )

    bn_out = torch.empty(N, C_out, H_conv, W_conv, dtype=torch.bfloat16, device=x.device)
    bn_w_bf16 = bn_weight.to(torch.bfloat16).contiguous()
    bn_b_bf16 = bn_bias.to(torch.bfloat16).contiguous()
    rm_f32 = bn_running_mean.to(torch.float32).contiguous()
    rv_f32 = bn_running_var.to(torch.float32).contiguous()
    total_bn = N * C_out * H_conv * W_conv
    grid_bn = (total_bn + BLOCK_SIZE - 1) // BLOCK_SIZE
    bn_eval_kernel[lambda: ((grid_bn, 1, 1), (BLOCK_SIZE, 1, 1))](
        mish_out, rm_f32, rv_f32, bn_w_bf16, bn_b_bf16, bn_out,
        N, C_out, H_conv, W_conv,
    )

    return bn_out


# ============================================================
# ModelNew class
# ============================================================
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        conv_w = self.conv.weight.data
        conv_b = self.conv.bias.data
        bn_w = self.bn.weight.data
        bn_b = self.bn.bias.data
        bn_rm = self.bn.running_mean.data
        bn_rv = self.bn.running_var.data

        return avelang_model(
            x, conv_w, conv_b, bn_w, bn_b, bn_rm, bn_rv,
        )


batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
