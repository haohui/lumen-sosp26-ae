import torch
import torch.nn as nn
import struct
import avelang
import avelang.language as al

BLOCK_SIZE = 256


@avelang.jit
def conv3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    kD: al.i32,
    kH: al.i32,
    kW: al.i32,
    has_bias: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    total_input = N * C_in * D_in * H_in * W_in
    total_weight = C_out * C_in * kD * kH * kW
    total_output = N * C_out * D_out * H_out * W_out

    inp_layout = al.make_layout((total_input,), (1,))
    inp = al.make_tensor(input_ptr, al.bf16, inp_layout)

    wgt_layout = al.make_layout((total_weight,), (1,))
    wgt = al.make_tensor(weight_ptr, al.bf16, wgt_layout)

    out_layout = al.make_layout((total_output,), (1,))
    out = al.make_tensor(output_ptr, al.bf16, out_layout)

    bias_layout = al.make_layout((C_out,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    tid = al.block_id(0) * BLOCK_SIZE + al.thread_id(0)

    if tid < total_output:
        spatial_size = D_out * H_out * W_out
        ch_spatial = C_out * spatial_size

        n = tid // ch_spatial
        rem = tid - n * ch_spatial
        c_out = rem // spatial_size
        rem = rem - c_out * spatial_size

        d_out = rem // (H_out * W_out)
        rem = rem - d_out * (H_out * W_out)
        h_out = rem // W_out
        w_out = rem - h_out * W_out

        acc = al.convert(0.0, al.f32)

        c_stride_in = D_in * H_in * W_in
        c_stride_wgt = C_in * kD * kH * kW
        ic_stride_wgt = kD * kH * kW
        kd_stride_wgt = kH * kW

        base_in = n * C_in * c_stride_in
        base_wgt = c_out * c_stride_wgt

        for c_in in al.range(C_in):
            for kd in al.range(kD):
                for kh in al.range(kH):
                    for kw in al.range(kW):
                        d = d_out + kd
                        h = h_out + kh
                        w = w_out + kw

                        inp_idx = base_in + c_in * c_stride_in + d * (H_in * W_in) + h * W_in + w
                        wgt_idx = base_wgt + c_in * ic_stride_wgt + kd * kd_stride_wgt + kh * kW + kw

                        inp_val = al.convert(inp[inp_idx], al.f32)
                        w_val = al.convert(wgt[wgt_idx], al.f32)
                        acc = acc + inp_val * w_val

        if has_bias:
            b_val = al.convert(bias[c_out], al.f32)
            acc = acc + b_val

        out[tid] = al.convert(acc, al.bf16)


@avelang.jit
def groupnorm_min_clamp_kernel(
    input_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    G: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    eps_bits: al.i32,
    min_val_bits: al.i32,
    max_val_bits: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    total_pairs = N * G
    pair_idx = al.block_id(0)

    if pair_idx < total_pairs:
        n = pair_idx // G
        g = pair_idx - n * G

        C_per_group = C // G
        c_start = g * C_per_group
        spatial_per_ch = D * H * W
        total_per_group = C_per_group * spatial_per_ch

        total_elements = N * C * D * H * W

        x_layout = al.make_layout((total_elements,), (1,))
        x = al.make_tensor(input_ptr, al.bf16, x_layout)

        y_layout = al.make_layout((total_elements,), (1,))
        y = al.make_tensor(output_ptr, al.bf16, y_layout)

        gamma_layout = al.make_layout((C,), (1,))
        gamma = al.make_tensor(gamma_ptr, al.bf16, gamma_layout)

        beta_layout = al.make_layout((C,), (1,))
        beta = al.make_tensor(beta_ptr, al.bf16, beta_layout)

        tid = al.thread_id(0)

        # Shared memory for partial sums in FP32
        sdata = al.make_shared((BLOCK_SIZE,), al.f32)

        base_n = n * C * D * H * W

        # --- PASS 1: Compute mean ---
        partial_sum = al.convert(0.0, al.f32)

        for idx in al.range(tid, total_per_group, BLOCK_SIZE):
            c_local = idx // spatial_per_ch
            spatial = idx - c_local * spatial_per_ch

            d = spatial // (H * W)
            rem = spatial - d * (H * W)
            h = rem // W
            w = rem - h * W

            c_global = c_start + c_local
            flat_idx = base_n + c_global * spatial_per_ch + spatial

            val = al.convert(x[flat_idx], al.f32)
            partial_sum = partial_sum + val

        sdata[tid] = partial_sum
        al.syncthreads()

        # Thread 0 does sequential reduction
        if tid == 0:
            total_sum = al.convert(0.0, al.f32)
            for i in al.range(BLOCK_SIZE):
                total_sum = total_sum + sdata[i]
            mean = total_sum / al.convert(total_per_group, al.f32)
            sdata[0] = mean

        al.syncthreads()
        mean = sdata[0]

        # --- PASS 2: Compute variance ---
        partial_sq = al.convert(0.0, al.f32)

        for idx in al.range(tid, total_per_group, BLOCK_SIZE):
            c_local = idx // spatial_per_ch
            spatial = idx - c_local * spatial_per_ch

            d = spatial // (H * W)
            rem = spatial - d * (H * W)
            h = rem // W
            w = rem - h * W

            c_global = c_start + c_local
            flat_idx = base_n + c_global * spatial_per_ch + spatial

            val = al.convert(x[flat_idx], al.f32)
            diff = val - mean
            partial_sq = partial_sq + diff * diff

        sdata[tid] = partial_sq
        al.syncthreads()

        if tid == 0:
            total_sq = al.convert(0.0, al.f32)
            for i in al.range(BLOCK_SIZE):
                total_sq = total_sq + sdata[i]
            var = total_sq / al.convert(total_per_group, al.f32)
            sdata[1] = var

        al.syncthreads()
        var = sdata[1]

        eps = al.bitcast(eps_bits, al.f32)
        min_val = al.bitcast(min_val_bits, al.f32)
        max_val = al.bitcast(max_val_bits, al.f32)
        inv_std = al.convert(1.0, al.f32) / al.sqrt(var + eps)

        # --- PASS 3: Normalize, scale, min, clamp, and write output ---
        for idx in al.range(tid, total_per_group, BLOCK_SIZE):
            c_local = idx // spatial_per_ch
            spatial = idx - c_local * spatial_per_ch

            d = spatial // (H * W)
            rem = spatial - d * (H * W)
            h = rem // W
            w = rem - h * W

            c_global = c_start + c_local
            flat_idx = base_n + c_global * spatial_per_ch + spatial

            val = al.convert(x[flat_idx], al.f32)
            norm_val = (val - mean) * inv_std

            g_val = al.convert(gamma[c_global], al.f32)
            b_val = al.convert(beta[c_global], al.f32)
            result = norm_val * g_val + b_val

            # min(result, min_val): element-wise minimum
            if result > min_val:
                result = min_val

            # clamp(result, min_val, max_val)
            if result < min_val:
                result = min_val
            if result > max_val:
                result = max_val

            y[flat_idx] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.min_value = min_value
        self.max_value = max_value
        self.groups = groups

    def forward(self, x):
        N, C_in, D_in, H_in, W_in = x.shape
        C_out = self.conv.out_channels
        kD = self.conv.kernel_size[0]
        kH = self.conv.kernel_size[1]
        kW = self.conv.kernel_size[2]

        D_out = D_in - kD + 1
        H_out = H_in - kH + 1
        W_out = W_in - kW + 1

        input_dtype = x.dtype
        device = x.device

        # Ensure contiguous BF16 input
        x = x.contiguous().to(torch.bfloat16)

        # Extract conv weights (BF16, contiguous)
        weight = self.conv.weight.data.contiguous().to(torch.bfloat16)

        has_bias = 1 if self.conv.bias is not None else 0
        if has_bias:
            bias = self.conv.bias.data.contiguous().to(torch.bfloat16)
        else:
            bias = torch.zeros(C_out, dtype=torch.bfloat16, device=device)

        # Allocate conv output
        conv_out = torch.empty(N, C_out, D_out, H_out, W_out, dtype=torch.bfloat16, device=device)

        # Launch conv3d kernel
        total_outputs = N * C_out * D_out * H_out * W_out
        grid_x = (total_outputs + BLOCK_SIZE - 1) // BLOCK_SIZE

        conv3d_kernel[lambda: ((grid_x, 1, 1), (BLOCK_SIZE, 1, 1))](
            x, weight, bias, conv_out,
            N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
            kD, kH, kW, has_bias, BLOCK_SIZE,
        )

        # Extract GroupNorm params (BF16, contiguous)
        gamma = self.norm.weight.data.contiguous().to(torch.bfloat16)
        beta = self.norm.bias.data.contiguous().to(torch.bfloat16)
        eps = float(self.norm.eps)

        # Allocate GroupNorm output
        gn_out = torch.empty_like(conv_out)

        # Launch groupnorm kernel
        grid_gn = N * self.groups

        def f32_to_bits(v):
            return struct.unpack('<i', struct.pack('<f', float(v)))[0]

        eps_bits = f32_to_bits(self.norm.eps)
        min_val_bits = f32_to_bits(self.min_value)
        max_val_bits = f32_to_bits(self.max_value)

        groupnorm_min_clamp_kernel[lambda: ((grid_gn, 1, 1), (BLOCK_SIZE, 1, 1))](
            conv_out, gamma, beta, gn_out,
            N, C_out, self.groups, D_out, H_out, W_out,
            eps_bits, min_val_bits, max_val_bits, BLOCK_SIZE,
        )

        # Convert back to original dtype for dropout
        result = gn_out.to(input_dtype)
        result = self.dropout(result)

        return result
