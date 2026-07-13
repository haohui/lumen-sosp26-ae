import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def fused_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    sum_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K: al.i32,
    SP: al.i32,
):
    n = al.block_id(0)
    co = al.block_id(1)
    sp_tid = al.block_id(2) * al.block_dim(0) + al.thread_id(0)

    if sp_tid < SP:
        wh = H_out * W_out
        d = sp_tid // wh
        rem = sp_tid % wh
        h = rem // W_out
        w = rem % W_out

        in_t = al.make_tensor(
            input_ptr, al.bf16,
            al.make_layout(
                (B, C_in, D_in, H_in, W_in),
                (C_in * D_in * H_in * W_in, D_in * H_in * W_in, H_in * W_in, W_in, 1),
            ),
        )
        w_t = al.make_tensor(
            weight_ptr, al.bf16,
            al.make_layout(
                (C_out, C_in, K, K, K),
                (C_in * K * K * K, K * K * K, K * K, K, 1),
            ),
        )
        b_t = al.make_tensor(bias_ptr, al.bf16, al.make_layout((C_out,), (1,)))
        s_t = al.make_tensor(sum_ptr, al.bf16, al.make_layout((C_out,), (1,)))
        o_t = al.make_tensor(
            output_ptr, al.bf16,
            al.make_layout(
                (B, C_out, D_out, H_out, W_out),
                (C_out * D_out * H_out * W_out, D_out * H_out * W_out, H_out * W_out, W_out, 1),
            ),
        )

        # --- 3D Convolution ---
        acc = al.convert(0.0, al.f32)
        for ci in al.range(C_in):
            for kd in al.range(K):
                for kh in al.range(K):
                    for kw in al.range(K):
                        iv = al.convert(in_t[n, ci, d + kd, h + kh, w + kw], al.f32)
                        wv = al.convert(w_t[co, ci, kd, kh, kw], al.f32)
                        acc = acc + iv * wv
        acc = acc + al.convert(b_t[co], al.f32)

        # --- LeakyReLU ---
        zero = al.convert(0.0, al.f32)
        slope = al.convert(0.2, al.f32)
        if acc >= zero:
            pass
        else:
            acc = slope * acc

        # --- Add sum_tensor ---
        acc = acc + al.convert(s_t[co], al.f32)

        # --- Clamp to [-1, 1] ---
        one = al.convert(1.0, al.f32)
        neg_one = al.convert(-1.0, al.f32)
        if acc > one:
            acc = one
        if acc < neg_one:
            acc = neg_one

        # --- GELU via tanh approximation ---
        sqrt2pi = al.convert(0.7978845608028654, al.f32)
        c_044715 = al.convert(0.044715, al.f32)
        half = al.convert(0.5, al.f32)
        one_f = al.convert(1.0, al.f32)

        x3 = acc * acc * acc
        inner = sqrt2pi * (acc + c_044715 * x3)
        gelu_val = half * acc * (one_f + al.tanh(inner))

        o_t[n, co, d, h, w] = al.convert(gelu_val, al.bf16)


def _launch_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    sum_tensor: torch.Tensor,
) -> torch.Tensor:
    B, C_in, D_in, H_in, W_in = x.shape
    C_out = weight.shape[0]
    K = weight.shape[2]
    D_out = D_in - K + 1
    H_out = H_in - K + 1
    W_out = W_in - K + 1
    sp = D_out * H_out * W_out

    out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)

    BLOCK = 256
    sp_blocks = (sp + BLOCK - 1) // BLOCK
    fused_kernel[lambda: ((B, C_out, sp_blocks), (BLOCK, 1, 1))](
        x, weight, bias, sum_tensor, out,
        B, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, sp,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        s = self.sum_tensor.contiguous()

        return _launch_fused(x, w, b, s)
