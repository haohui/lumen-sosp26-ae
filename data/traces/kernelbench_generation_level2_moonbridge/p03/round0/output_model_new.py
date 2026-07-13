import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 64
POOL_BLOCK_SIZE: al.constexpr = 32


@avelang.jit
def add_layernorm_kernel(
    x_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    sum_weight: al.f32,
    eps: al.f32,
    total_elements: al.i32,
    W: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    base = bid * W

    if base < total_elements:
        layout_1d = al.make_layout((total_elements,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_1d)
        out = al.make_tensor(out_ptr, al.bf16, layout_1d)

        layout_w = al.make_layout((W,), (1,))
        gamma = al.make_tensor(gamma_ptr, al.bf16, layout_w)
        beta_t = al.make_tensor(beta_ptr, al.bf16, layout_w)

        smem_val = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_red = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_stats = al.make_shared((2,), al.f32)

        # Phase 1: Load input, add sum_weight in bf16, store f32 values in shared memory
        if tid < W:
            val = al.convert(x[base + tid], al.f32) + sum_weight
            val_bf16 = al.convert(val, al.bf16)
            val_f32 = al.convert(val_bf16, al.f32)
            smem_val[tid] = val_f32
            smem_red[tid] = val_f32
        else:
            smem_val[tid] = al.convert(0.0, al.f32)
            smem_red[tid] = al.convert(0.0, al.f32)

        al.syncthreads()

        # Reduction 1: tree reduce to compute sum -> mean
        if tid < 32:
            smem_red[tid] = smem_red[tid] + smem_red[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_red[tid] = smem_red[tid] + smem_red[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_red[tid] = smem_red[tid] + smem_red[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_red[tid] = smem_red[tid] + smem_red[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_red[tid] = smem_red[tid] + smem_red[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_red[tid] = smem_red[tid] + smem_red[tid + 1]

        # Thread 0 computes mean, stores in smem_stats[0]
        if tid == 0:
            W_f32 = al.convert(W, al.f32)
            mean = smem_red[0] / W_f32
            smem_stats[0] = mean

        al.syncthreads()

        mean = smem_stats[0]

        # Phase 2: compute (x - mean)^2 from stored values, reduce to get variance
        if tid < W:
            diff = smem_val[tid] - mean
            smem_red[tid] = diff * diff
        else:
            smem_red[tid] = al.convert(0.0, al.f32)

        al.syncthreads()

        # Reduction 2: tree reduce squared diffs -> variance
        if tid < 32:
            smem_red[tid] = smem_red[tid] + smem_red[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_red[tid] = smem_red[tid] + smem_red[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_red[tid] = smem_red[tid] + smem_red[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_red[tid] = smem_red[tid] + smem_red[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_red[tid] = smem_red[tid] + smem_red[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_red[tid] = smem_red[tid] + smem_red[tid + 1]

        # Thread 0 computes rstd, stores in smem_stats[1]
        if tid == 0:
            W_f32 = al.convert(W, al.f32)
            var = smem_red[0] / W_f32
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)
            smem_stats[1] = rstd

        al.syncthreads()

        rstd = smem_stats[1]
        mean = smem_stats[0]

        # Phase 3: apply normalization and write output
        if tid < W:
            val = smem_val[tid]
            g = al.convert(gamma[tid], al.f32)
            b = al.convert(beta_t[tid], al.f32)
            result = (val - mean) * rstd * g + b
            out[base + tid] = al.convert(result, al.bf16)


@avelang.jit
def avgpool_gelu_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_in: al.i32,
    total_out: al.i32,
    C: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    stride_in_dh: al.i32,
    stride_in_hw: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    # Decode block id into (b, c, d_o, h_o)
    b = bid // (C * D_out * H_out)
    rem = bid - b * (C * D_out * H_out)
    c = rem // (D_out * H_out)
    rem = rem - c * (D_out * H_out)
    d_o = rem // H_out
    h_o = rem - d_o * H_out
    w_o = tid

    if w_o < W_out:
        layout_in = al.make_layout((total_in,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_in)

        layout_out = al.make_layout((total_out,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        in_base = (b * C + c) * D_in * H_in * W_in

        d0 = 2 * d_o
        d1 = d0 + 1
        h0 = 2 * h_o
        h1 = h0 + 1
        w0 = 2 * w_o
        w1 = w0 + 1

        pool_sum = al.convert(0.0, al.f32)

        # Pool 2x2x2: 8 input elements
        pool_sum = pool_sum + al.convert(x[in_base + d0 * stride_in_dh + h0 * stride_in_hw + w0], al.f32)
        pool_sum = pool_sum + al.convert(x[in_base + d0 * stride_in_dh + h0 * stride_in_hw + w1], al.f32)
        pool_sum = pool_sum + al.convert(x[in_base + d0 * stride_in_dh + h1 * stride_in_hw + w0], al.f32)
        pool_sum = pool_sum + al.convert(x[in_base + d0 * stride_in_dh + h1 * stride_in_hw + w1], al.f32)
        pool_sum = pool_sum + al.convert(x[in_base + d1 * stride_in_dh + h0 * stride_in_hw + w0], al.f32)
        pool_sum = pool_sum + al.convert(x[in_base + d1 * stride_in_dh + h0 * stride_in_hw + w1], al.f32)
        pool_sum = pool_sum + al.convert(x[in_base + d1 * stride_in_dh + h1 * stride_in_hw + w0], al.f32)
        pool_sum = pool_sum + al.convert(x[in_base + d1 * stride_in_dh + h1 * stride_in_hw + w1], al.f32)

        pooled = pool_sum / al.convert(8.0, al.f32)

        # GELU activation using tanh approximation:
        # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_div_pi = al.convert(0.7978845608, al.f32)
        coeff = al.convert(0.044715, al.f32)
        half = al.convert(0.5, al.f32)
        one = al.convert(1.0, al.f32)

        x3 = pooled * pooled * pooled
        tanh_arg = sqrt_2_div_pi * (pooled + coeff * x3)
        gelu_result = half * pooled * (one + al.tanh(tanh_arg))

        out_idx = ((b * C + c) * D_out + d_o) * H_out * W_out + h_o * W_out + w_o
        out[out_idx] = al.convert(gelu_result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.norm_shape = norm_shape

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self._sum_weight_val = float(sum_weight)
        self.ln_weight = nn.Parameter(torch.ones(norm_shape))
        self.ln_bias = nn.Parameter(torch.zeros(norm_shape))

    def forward(self, x):
        # Stage 1: ConvTranspose3d (PyTorch)
        x = self.conv_transpose(x)

        # Stage 2: Add sum_weight + LayerNorm (AveLang kernel)
        x = self._fused_add_layernorm(x)

        # Stage 3: AvgPool3d + GELU (AveLang kernel)
        x = self._fused_avgpool_gelu(x)

        return x

    def _fused_add_layernorm(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        total_elements = B * C * D * H * W

        x_bf16 = x.contiguous().to(torch.bfloat16)
        gamma_bf16 = self.ln_weight.contiguous().to(torch.bfloat16)
        beta_bf16 = self.ln_bias.contiguous().to(torch.bfloat16)

        out = torch.empty_like(x_bf16)

        num_groups = B * C * D * H
        eps = 1e-5

        add_layernorm_kernel[lambda: ((num_groups, 1, 1), (BLOCK_SIZE, 1, 1))](
            x_bf16, gamma_bf16, beta_bf16, out,
            self._sum_weight_val, eps,
            total_elements, W
        )

        return out.to(x.dtype)

    def _fused_avgpool_gelu(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D_in, H_in, W_in = x.shape
        kD, kH, kW = self.pool_kernel_size

        D_out = D_in // kD
        H_out = H_in // kH
        W_out = W_in // kW

        total_in = B * C * D_in * H_in * W_in
        total_out = B * C * D_out * H_out * W_out

        x_bf16 = x.contiguous().to(torch.bfloat16)
        out = torch.empty((B, C, D_out, H_out, W_out), dtype=torch.bfloat16, device=x.device)

        num_groups = B * C * D_out * H_out

        avgpool_gelu_kernel[lambda: ((num_groups, 1, 1), (POOL_BLOCK_SIZE, 1, 1))](
            x_bf16, out,
            total_in, total_out,
            C, D_out, H_out, W_out,
            D_in, H_in, W_in,
            H_in * W_in,
            W_in,
        )

        return out.to(x.dtype)
