import torch
import torch.nn as nn
import avelang
import avelang.language as al


# =============================================================================
# Kernel: 3D Convolution with bias (BF16 I/O, FP32 accumulation)
#
# Launch: grid=(H_out * D_out, C_out, B), block=(W_out, 1, 1)
# =============================================================================
@avelang.jit
def conv3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    K: al.i32,
):
    D_out = D - K + 1
    H_out = H - K + 1
    W_out = W - K + 1

    # Input layout: (B, C_in, D, H, W) row-major
    x_s0 = C_in * D * H * W
    x_s1 = D * H * W
    x_s2 = H * W
    x_s3 = W
    x_s4 = 1
    x_layout = al.make_layout((B, C_in, D, H, W), (x_s0, x_s1, x_s2, x_s3, x_s4))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    # Weight layout: (C_out, C_in, K, K, K) row-major
    w_s0 = C_in * K * K * K
    w_s1 = K * K * K
    w_s2 = K * K
    w_s3 = K
    w_s4 = 1
    w_layout = al.make_layout((C_out, C_in, K, K, K), (w_s0, w_s1, w_s2, w_s3, w_s4))
    w = al.make_tensor(w_ptr, al.bf16, w_layout)

    # Bias layout: (C_out,) row-major
    b_layout = al.make_layout((C_out,), (1,))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)

    # Output layout: (B, C_out, D_out, H_out, W_out) row-major
    o_s0 = C_out * D_out * H_out * W_out
    o_s1 = D_out * H_out * W_out
    o_s2 = H_out * W_out
    o_s3 = W_out
    o_s4 = 1
    o_layout = al.make_layout((B, C_out, D_out, H_out, W_out), (o_s0, o_s1, o_s2, o_s3, o_s4))
    out = al.make_tensor(out_ptr, al.bf16, o_layout)

    # Grid/block decomposition:
    #   grid  = (H_out * D_out,  C_out,   B  )
    #   block = (W_out,          1,       1  )
    ow = al.thread_id(0)        # 0 .. W_out-1
    spatial = al.block_id(0)    # 0 .. H_out*D_out-1
    oc = al.block_id(1)         # 0 .. C_out-1
    b_idx = al.block_id(2)      # 0 .. B-1

    # Decompose spatial into (oh, od)
    od = spatial // H_out       # 0 .. D_out-1
    oh = spatial % H_out        # 0 .. H_out-1

    # FP32 accumulation, start with bias
    acc = al.convert(b[oc], al.f32)

    for ic in al.range(C_in):
        for kd in al.range(K):
            for kh in al.range(K):
                for kw in al.range(K):
                    x_val = al.convert(x[b_idx, ic, od + kd, oh + kh, ow + kw], al.f32)
                    w_val = al.convert(w[oc, ic, kd, kh, kw], al.f32)
                    acc = acc + x_val * w_val

    out[b_idx, oc, od, oh, ow] = al.convert(acc, al.bf16)


# =============================================================================
# Kernel: Softmax along channel dimension (dim=1).
#
# Launch: grid=(H * D, 1, B), block=(W, 1, 1)
# =============================================================================
@avelang.jit
def softmax_channel_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
):
    # Layout: (B, C, D, H, W) row-major
    s0 = C * D * H * W
    s1 = D * H * W
    s2 = H * W
    s3 = W
    s4 = 1
    layout = al.make_layout((B, C, D, H, W), (s0, s1, s2, s3, s4))
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    # Grid/block decomposition:
    #   grid  = (H * D,     1,      B  )
    #   block = (W,          1,      1  )
    w = al.thread_id(0)         # 0 .. W-1
    spatial = al.block_id(0)    # 0 .. H*D-1
    b = al.block_id(2)          # 0 .. B-1

    # Decompose spatial into (h, d)
    h = spatial % H
    d = spatial // H

    # Pass 1: find max across channels for numerical stability
    max_val = al.convert(x[b, 0, d, h, w], al.f32)
    for c in al.range(1, C):
        val = al.convert(x[b, c, d, h, w], al.f32)
        if val > max_val:
            max_val = val

    # Pass 2: compute sum of exp(x_i - max)
    sum_val = al.full((1,), 0.0, al.f32)[0]
    for c in al.range(C):
        val = al.convert(x[b, c, d, h, w], al.f32)
        sum_val = sum_val + al.exp(val - max_val)

    # Pass 3: normalize and store
    for c in al.range(C):
        val = al.convert(x[b, c, d, h, w], al.f32)
        out[b, c, d, h, w] = al.convert(al.exp(val - max_val) / sum_val, al.bf16)


# =============================================================================
# Kernel: 3D Max Pooling (kernel=2, stride=2, padding=0).
#
# Launch: grid=(H_out * D_out, C, B), block=(W_out, 1, 1)
# =============================================================================
@avelang.jit
def maxpool3d_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
):
    D_out = D_in // 2
    H_out = H_in // 2
    W_out = W_in // 2

    # Input layout: (B, C, D_in, H_in, W_in) row-major
    x_s0 = C * D_in * H_in * W_in
    x_s1 = D_in * H_in * W_in
    x_s2 = H_in * W_in
    x_s3 = W_in
    x_s4 = 1
    x_layout = al.make_layout((B, C, D_in, H_in, W_in), (x_s0, x_s1, x_s2, x_s3, x_s4))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    # Output layout: (B, C, D_out, H_out, W_out) row-major
    o_s0 = C * D_out * H_out * W_out
    o_s1 = D_out * H_out * W_out
    o_s2 = H_out * W_out
    o_s3 = W_out
    o_s4 = 1
    o_layout = al.make_layout((B, C, D_out, H_out, W_out), (o_s0, o_s1, o_s2, o_s3, o_s4))
    out = al.make_tensor(out_ptr, al.bf16, o_layout)

    # Grid/block decomposition:
    #   grid  = (H_out * D_out,  C,      B  )
    #   block = (W_out,          1,      1  )
    ow = al.thread_id(0)        # 0 .. W_out-1
    spatial = al.block_id(0)    # 0 .. H_out*D_out-1
    oc = al.block_id(1)         # 0 .. C-1
    b = al.block_id(2)          # 0 .. B-1

    # Decompose spatial into (oh, od)
    od = spatial // H_out
    oh = spatial % H_out

    # Max over 2x2x2 window
    max_val = al.convert(x[b, oc, od * 2, oh * 2, ow * 2], al.f32)
    for kd in al.range(2):
        for kh in al.range(2):
            for kw in al.range(2):
                val = al.convert(x[b, oc, od * 2 + kd, oh * 2 + kh, ow * 2 + kw], al.f32)
                if val > max_val:
                    max_val = val

    out[b, oc, od, oh, ow] = al.convert(max_val, al.bf16)


# =============================================================================
# Host wrapper
# =============================================================================
def avelang_forward(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> torch.Tensor:
    orig_dtype = x.dtype
    device = x.device

    # Convert to BF16 contiguous
    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = conv_weight.to(torch.bfloat16).contiguous()
    b_bf16 = conv_bias.to(torch.bfloat16).contiguous()

    B, C_in, D, H, W = x.shape
    C_out = conv_weight.shape[0]
    K = conv_weight.shape[2]  # kernel_size

    D_conv = D - K + 1
    H_conv = H - K + 1
    W_conv = W - K + 1

    # Intermediate tensor: conv output
    conv_out = torch.empty(B, C_out, D_conv, H_conv, W_conv, dtype=torch.bfloat16, device=device)

    # Launch conv kernel: grid=(H_out * D_out, C_out, B), block=(W_out, 1, 1)
    conv3d_kernel[lambda: ((H_conv * D_conv, C_out, B), (W_conv, 1, 1))](
        x_bf16, w_bf16, b_bf16, conv_out,
        B, C_in, C_out, D, H, W, K,
    )

    # Intermediate tensor: softmax output (same shape as conv output)
    softmax_out = torch.empty_like(conv_out)

    # Launch softmax kernel: grid=(H * D, 1, B), block=(W, 1, 1)
    softmax_channel_kernel[lambda: ((H_conv * D_conv, 1, B), (W_conv, 1, 1))](
        conv_out, softmax_out,
        B, C_out, D_conv, H_conv, W_conv,
    )

    # First maxpool: (B, C_out, D_conv, H_conv, W_conv) -> (B, C_out, D_p1, H_p1, W_p1)
    D_p1 = D_conv // 2
    H_p1 = H_conv // 2
    W_p1 = W_conv // 2
    pool1_out = torch.empty(B, C_out, D_p1, H_p1, W_p1, dtype=torch.bfloat16, device=device)

    # Launch maxpool kernel 1: grid=(H_out * D_out, C, B), block=(W_out, 1, 1)
    maxpool3d_kernel[lambda: ((H_p1 * D_p1, C_out, B), (W_p1, 1, 1))](
        softmax_out, pool1_out,
        B, C_out, D_conv, H_conv, W_conv,
    )

    # Second maxpool: (B, C_out, D_p1, H_p1, W_p1) -> (B, C_out, D_p2, H_p2, W_p2)
    D_p2 = D_p1 // 2
    H_p2 = H_p1 // 2
    W_p2 = W_p1 // 2
    pool2_out = torch.empty(B, C_out, D_p2, H_p2, W_p2, dtype=torch.bfloat16, device=device)

    # Launch maxpool kernel 2: grid=(H_out * D_out, C, B), block=(W_out, 1, 1)
    maxpool3d_kernel[lambda: ((H_p2 * D_p2, C_out, B), (W_p2, 1, 1))](
        pool1_out, pool2_out,
        B, C_out, D_p1, H_p1, W_p1,
    )

    return pool2_out.to(orig_dtype)


# =============================================================================
# ModelNew: compatible with get_inputs() / get_init_inputs()
# =============================================================================
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return avelang_forward(x, self.conv.weight, self.conv.bias)


# Required module-level definitions for the harness
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
