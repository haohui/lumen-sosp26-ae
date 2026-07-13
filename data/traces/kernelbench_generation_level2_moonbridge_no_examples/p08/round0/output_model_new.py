import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ===========================================================================
# Kernel 1: Fused Conv3D + Divide + MaxPool3D
#
# Computes: max_pool(conv3d(input, weight) / 2.0)
#
# Grid:  (N, C_out * D_pool, 1)
# Block: (128, 1, 1)
# ===========================================================================
@avelang.jit
def fused_conv_pool_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    conv_bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32, C_in: al.i32, C_out: al.i32,
    D_in: al.i32, H_in: al.i32, W_in: al.i32,
    KD: al.i32, KH: al.i32, KW: al.i32,
    D_pool: al.i32, H_pool: al.i32, W_pool: al.i32,
):
    n = al.block_id(0)
    cd = al.block_id(1)
    c = cd // D_pool
    dp = cd % D_pool

    tid = al.thread_id(0)
    NT = al.block_dim(0)

    # Input layout (NCDHW strides)
    in_s_C = D_in * H_in * W_in
    in_layout = al.make_layout(
        (N, C_in, D_in, H_in, W_in),
        (C_in * D_in * H_in * W_in, in_s_C, H_in * W_in, W_in, 1),
    )
    in_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    # Conv bias
    cb_layout = al.make_layout((C_out,), (1,))
    cb_t = al.make_tensor(conv_bias_ptr, al.bf16, cb_layout)

    # Weight layout (C_out, C_in, KD, KH, KW)
    w_s_CI = KD * KH * KW
    w_layout = al.make_layout(
        (C_out, C_in, KD, KH, KW),
        (C_in * KD * KH * KW, w_s_CI, KH * KW, KH, 1),
    )
    w_t = al.make_tensor(weight_ptr, al.bf16, w_layout)

    # Output layout
    out_s_CO = D_pool * H_pool * W_pool
    out_layout = al.make_layout(
        (N, C_out, D_pool, H_pool, W_pool),
        (C_out * D_pool * H_pool * W_pool, out_s_CO, H_pool * W_pool, W_pool, 1),
    )
    out_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    total_spatial = H_pool * W_pool
    for idx in al.range(tid, total_spatial, NT):
        hp = idx // W_pool
        wp = idx % W_pool

        max_val = al.convert(-1.0e30, al.f32)

        for pd in al.range(2):
            dc = al.convert(2, al.i32) * dp + pd
            for ph in al.range(2):
                hc = al.convert(2, al.i32) * hp + ph
                for pw in al.range(2):
                    wc = al.convert(2, al.i32) * wp + pw

                    conv_val = al.convert(cb_t[c], al.f32)
                    for ci in al.range(C_in):
                        for kd in al.range(KD):
                            d_idx = dc + kd
                            for kh in al.range(KH):
                                h_idx = hc + kh
                                for kw in al.range(KW):
                                    w_idx = wc + kw
                                    in_val = al.convert(
                                        in_t[n, ci, d_idx, h_idx, w_idx], al.f32
                                    )
                                    w_val = al.convert(
                                        w_t[c, ci, kd, kh, kw], al.f32
                                    )
                                    conv_val = conv_val + in_val * w_val

                    conv_val = conv_val / al.convert(2.0, al.f32)
                    if conv_val > max_val:
                        max_val = conv_val

        out_t[n, c, dp, hp, wp] = al.convert(max_val, al.bf16)


# ===========================================================================
# Kernel 2: Global Average Pool + Extra Bias + Sum over Channels
#
# Computes: sum_c( avg_pool(input)[c] + extra_bias[c] )
#
# Grid:  (N, 1, 1)
# Block: (256, 1, 1)
# ===========================================================================
@avelang.jit
def reduce_kernel(
    pool_ptr: al.Pointer(al.bf16),
    extra_bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32, C_out: al.i32,
    D_pool: al.i32, H_pool: al.i32, W_pool: al.i32,
):
    n = al.block_id(0)
    tid = al.thread_id(0)
    NT = al.block_dim(0)

    # Input layout
    in_s_CO = D_pool * H_pool * W_pool
    pool_layout = al.make_layout(
        (N, C_out, D_pool, H_pool, W_pool),
        (C_out * D_pool * H_pool * W_pool, in_s_CO, H_pool * W_pool, W_pool, 1),
    )
    pool_t = al.make_tensor(pool_ptr, al.bf16, pool_layout)

    # Extra bias layout (C_out,)
    eb_layout = al.make_layout((C_out,), (1,))
    eb_t = al.make_tensor(extra_bias_ptr, al.bf16, eb_layout)

    S = D_pool * H_pool * W_pool
    S_f32 = al.convert(S, al.f32)
    partial = al.convert(0.0, al.f32)

    for ch in al.range(tid, C_out, NT):
        ch_sum = al.convert(0.0, al.f32)
        for dp in al.range(D_pool):
            for hp in al.range(H_pool):
                for wp in al.range(W_pool):
                    ch_sum = ch_sum + al.convert(
                        pool_t[n, ch, dp, hp, wp], al.f32
                    )
        partial = partial + ch_sum / S_f32
        partial = partial + al.convert(eb_t[ch], al.f32)

    # Block reduction via shared memory
    smem = al.make_shared((256,), al.f32)
    smem[tid] = partial
    al.syncthreads()

    stride = al.convert(128, al.i32)
    for _ in al.range(8):
        if tid < stride:
            smem[tid] = smem[tid] + smem[tid + stride]
        al.syncthreads()
        stride = stride // 2

    if tid == 0:
        out_layout = al.make_layout((N,), (1,))
        out_t = al.make_tensor(output_ptr, al.bf16, out_layout)
        out_t[n] = al.convert(smem[al.convert(0, al.i32)], al.bf16)


# ===========================================================================
# Host wrapper
# ===========================================================================
def _make_contiguous_bf16(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is contiguous and in bf16 on the current GPU device."""
    if not t.is_contiguous():
        t = t.contiguous()
    return t.to(dtype=torch.bfloat16)


class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, divisor,
        pool_size, bias_shape, sum_dim,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kD, self.kH, self.kW = kernel_size
        self.pD, self.pH, self.pW = pool_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        C_in = x.shape[1]
        D_in = x.shape[2]
        H_in = x.shape[3]
        W_in = x.shape[4]

        kD, kH, kW = self.kD, self.kH, self.kW

        D_pool = (D_in - kD + 1) // self.pD
        H_pool = (H_in - kH + 1) // self.pH
        W_pool = (W_in - kW + 1) // self.pW

        C_out = self.out_channels

        # Extract weights and biases, ensure contiguous bf16
        x_bf16 = _make_contiguous_bf16(x)
        w = _make_contiguous_bf16(self.conv.weight)
        cb = _make_contiguous_bf16(self.conv.bias)
        eb = _make_contiguous_bf16(self.bias.view(-1))

        # Intermediate buffer: (N, C_out, D_pool, H_pool, W_pool)
        pooled = torch.empty(
            N, C_out, D_pool, H_pool, W_pool,
            dtype=torch.bfloat16, device=x.device,
        )

        # Launch kernel 1: fused conv + divide + max pool
        grid1 = (N, C_out * D_pool, 1)
        block1 = (128, 1, 1)
        fused_conv_pool_kernel[lambda: (grid1, block1)](
            x_bf16, w, cb, pooled,
            N, C_in, C_out,
            D_in, H_in, W_in,
            kD, kH, kW,
            D_pool, H_pool, W_pool,
        )

        # Output buffer: (N,)
        out_flat = torch.empty(N, dtype=torch.bfloat16, device=x.device)

        # Launch kernel 2: global avg pool + extra bias + sum over channels
        grid2 = (N, 1, 1)
        block2 = (256, 1, 1)
        reduce_kernel[lambda: (grid2, block2)](
            pooled, eb, out_flat,
            N, C_out, D_pool, H_pool, W_pool,
        )

        # Reshape to match reference output shape: (N, 1, 1, 1)
        return out_flat.view(N, 1, 1, 1)
