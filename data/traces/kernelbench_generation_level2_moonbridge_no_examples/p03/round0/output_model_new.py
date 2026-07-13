import struct
import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ============================================================
# Kernel 1: LayerNorm (training mode, over last dim)
# ============================================================
@avelang.jit
def layernorm_kernel(
    inp_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    W: al.i32,
    N_groups: al.i32,
    total_in: al.i32,
    eps_bits: al.i32,
):
    eps = al.bitcast(eps_bits, al.f32)

    group_idx = al.block_id(0)
    tid = al.thread_id(0)

    in_layout = al.make_layout((total_in,), (1,))
    w_layout = al.make_layout((W,), (1,))
    in_flat = al.make_tensor(inp_ptr, al.bf16, in_layout)
    gamma_flat = al.make_tensor(gamma_ptr, al.bf16, w_layout)
    beta_flat = al.make_tensor(beta_ptr, al.bf16, w_layout)
    out_flat = al.make_tensor(out_ptr, al.bf16, in_layout)

    if group_idx < N_groups and tid < W:
        base = group_idx * W
        val_f32 = al.convert(in_flat[base + tid], al.f32)

        shared_data = al.make_shared((64,), al.f32)
        shared_data[tid] = val_f32
        al.syncthreads()

        if tid < 32:
            shared_data[tid] = shared_data[tid] + shared_data[tid + 32]
        al.syncthreads()
        if tid < 16:
            shared_data[tid] = shared_data[tid] + shared_data[tid + 16]
        al.syncthreads()
        if tid < 8:
            shared_data[tid] = shared_data[tid] + shared_data[tid + 8]
        al.syncthreads()
        if tid < 4:
            shared_data[tid] = shared_data[tid] + shared_data[tid + 4]
        al.syncthreads()
        if tid < 2:
            shared_data[tid] = shared_data[tid] + shared_data[tid + 2]
        al.syncthreads()
        if tid < 1:
            shared_data[tid] = shared_data[tid] + shared_data[tid + 1]
        al.syncthreads()

        mean = shared_data[0] / al.convert(W, al.f32)

        diff = val_f32 - mean
        shared_data[tid] = diff * diff
        al.syncthreads()

        if tid < 32:
            shared_data[tid] = shared_data[tid] + shared_data[tid + 32]
        al.syncthreads()
        if tid < 16:
            shared_data[tid] = shared_data[tid] + shared_data[tid + 16]
        al.syncthreads()
        if tid < 8:
            shared_data[tid] = shared_data[tid] + shared_data[tid + 8]
        al.syncthreads()
        if tid < 4:
            shared_data[tid] = shared_data[tid] + shared_data[tid + 4]
        al.syncthreads()
        if tid < 2:
            shared_data[tid] = shared_data[tid] + shared_data[tid + 2]
        al.syncthreads()
        if tid < 1:
            shared_data[tid] = shared_data[tid] + shared_data[tid + 1]
        al.syncthreads()

        var = shared_data[0] / al.convert(W, al.f32)
        inv_std = al.convert(1.0, al.f32) / al.sqrt(var + eps)
        norm = diff * inv_std

        gamma_val = al.convert(gamma_flat[tid], al.f32)
        beta_val = al.convert(beta_flat[tid], al.f32)
        result = norm * gamma_val + beta_val

        out_flat[base + tid] = al.convert(result, al.bf16)


# ============================================================
# Kernel 2: AvgPool3d + GELU
# ============================================================
@avelang.jit
def avgpool_gelu_kernel(
    inp_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_out: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    pool_d: al.i32,
    pool_h: al.i32,
    pool_w: al.i32,
    total_per_batch: al.i32,
    total_elements: al.i32,
):
    in_total = B * C_out * D_in * H_in * W_in
    out_total = B * C_out * D_out * H_out * W_out

    in_flat = al.make_tensor(inp_ptr, al.bf16, al.make_layout((in_total,), (1,)))
    out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((out_total,), (1,)))

    in_n_stride = C_out * D_in * H_in * W_in
    in_c_stride = D_in * H_in * W_in
    in_d_stride = H_in * W_in
    in_h_stride = W_in

    out_n_stride = C_out * D_out * H_out * W_out
    out_c_stride = D_out * H_out * W_out
    out_d_stride = H_out * W_out
    out_h_stride = W_out

    d_out_h_out_w_out = D_out * H_out * W_out
    h_out_w_out = H_out * W_out

    tid_global = al.block_id(0) * al.block_dim(0) + al.thread_id(0)

    if tid_global < total_elements:
        batch = tid_global // total_per_batch
        local_idx = tid_global - batch * total_per_batch

        c = local_idx // d_out_h_out_w_out
        spatial = local_idx - c * d_out_h_out_w_out
        dp = spatial // h_out_w_out
        rem = spatial - dp * h_out_w_out
        hp = rem // W_out
        wp = rem - hp * W_out

        if c < C_out:
            in_nc_base = batch * in_n_stride + c * in_c_stride
            out_idx = batch * out_n_stride + c * out_c_stride + dp * out_d_stride + hp * out_h_stride + wp

            pool_sum = al.convert(0.0, al.f32)

            d_start = dp * pool_d
            h_start = hp * pool_h
            w_start = wp * pool_w

            for dd in al.range(pool_d):
                d_idx = d_start + dd
                in_d_base = in_nc_base + d_idx * in_d_stride
                for dh in al.range(pool_h):
                    h_idx = h_start + dh
                    in_h_base = in_d_base + h_idx * in_h_stride
                    for dw in al.range(pool_w):
                        w_idx = w_start + dw
                        in_idx = in_h_base + w_idx
                        val = al.convert(in_flat[in_idx], al.f32)
                        pool_sum = pool_sum + val

            pool_count = al.convert(pool_d * pool_h * pool_w, al.f32)
            avg = pool_sum / pool_count

            sqrt_2_over_pi = al.convert(0.7978845608, al.f32)
            coeff = al.convert(0.044715, al.f32)
            half = al.convert(0.5, al.f32)
            one = al.convert(1.0, al.f32)

            x3 = avg * avg * avg
            inner = sqrt_2_over_pi * (avg + coeff * x3)
            tanh_val = al.tanh(inner)
            gelu_val = half * avg * (one + tanh_val)

            out_flat[out_idx] = al.convert(gelu_val, al.bf16)


# ============================================================
# Host wrapper / ModelNew
# ============================================================
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        norm = nn.LayerNorm(norm_shape)

        self.register_buffer("norm_gamma", norm.weight.data.clone())
        self.register_buffer("norm_beta", norm.bias.data.clone())

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.sum_weight_val = float(sum_weight)
        self.eps_bits = struct.unpack('<I', struct.pack('<f', 1e-5))[0]
        self.pool_kernel_size = pool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W_norm = norm_shape[0]

    def forward(self, x):
        x = x.contiguous()
        B = x.shape[0]

        # ConvTranspose3d + sum_weight via PyTorch (exact match to reference)
        x_bf16 = x.to(torch.bfloat16)
        conv_out = self.conv_transpose(x_bf16)
        conv_out = conv_out + self.sum_weight_val
        conv_out = conv_out.contiguous()

        D_out = conv_out.shape[2]
        H_out = conv_out.shape[3]
        W_out = conv_out.shape[4]
        C_out = conv_out.shape[1]

        gamma = self.norm_gamma.to(torch.bfloat16).contiguous()
        beta = self.norm_beta.to(torch.bfloat16).contiguous()

        # -------- Kernel 1: LayerNorm --------
        W_norm = self.W_norm
        N_groups = B * C_out * D_out * H_out
        total_in_norm = N_groups * W_norm

        norm_out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)

        layernorm_kernel[lambda: ((N_groups, 1, 1), (64, 1, 1))](
            conv_out.data_ptr(),
            gamma.data_ptr(),
            beta.data_ptr(),
            norm_out.data_ptr(),
            W_norm,
            N_groups,
            total_in_norm,
            self.eps_bits,
        )

        # -------- Kernel 2: AvgPool3d + GELU --------
        pool_d, pool_h, pool_w = self.pool_kernel_size
        D_pool = D_out // pool_d
        H_pool = H_out // pool_h
        W_pool = W_out // pool_w

        total_per_batch_pool = C_out * D_pool * H_pool * W_pool
        total_elements_pool = B * total_per_batch_pool
        BLOCK_SIZE = 256
        num_blocks_pool = (total_elements_pool + BLOCK_SIZE - 1) // BLOCK_SIZE

        final_out = torch.empty(B, C_out, D_pool, H_pool, W_pool, dtype=torch.bfloat16, device=x.device)

        avgpool_gelu_kernel[lambda: ((num_blocks_pool, 1, 1), (BLOCK_SIZE, 1, 1))](
            norm_out.data_ptr(),
            final_out.data_ptr(),
            B, C_out, D_out, H_out, W_out,
            D_pool, H_pool, W_pool,
            pool_d, pool_h, pool_w,
            total_per_batch_pool,
            total_elements_pool,
        )

        return final_out
