import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ============================================================
# Kernel 1: Tiled GEMM with bias (BF16 inputs, FP32 accumulation)
# ============================================================

@avelang.jit
def gemm_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tid = al.thread_id(0)

    row = block_m * 128 + tid
    col_start = block_n * 128

    a_layout = al.make_layout((M, K), (K, 1))
    a = al.make_tensor(a_ptr, al.bf16, a_layout)
    b_layout = al.make_layout((N, K), (K, 1))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)
    c_layout = al.make_layout((M, N), (N, 1))
    c = al.make_tensor(c_ptr, al.bf16, c_layout)
    bias_layout = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)

    if row < M:
        acc = al.make_local((128,), al.f32)
        for n in al.range(0, 128):
            acc[n] = al.convert(0.0, al.f32)

        for k_idx in al.range(0, K):
            a_val = al.convert(a[row, k_idx], al.f32)
            for n in al.range(0, 128):
                col = col_start + n
                if col < N:
                    b_val = al.convert(b[col, k_idx], al.f32)
                    acc[n] = acc[n] + a_val * b_val

        for n in al.range(0, 128):
            col = col_start + n
            if col < N:
                bias_val = al.convert(bias[col], al.f32)
                c[row, col] = al.convert(acc[n] + bias_val, al.bf16)


# ============================================================
# Kernel 2: BatchNorm statistics (mean and variance, FP32 output)
# ============================================================

@avelang.jit
def bn_stats_kernel(
    x_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    M: al.i32,
    N: al.i32,
):
    col = al.block_id(0)
    tid = al.thread_id(0)

    if col >= N:
        return

    x_layout = al.make_layout((M, N), (N, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    s_sum = al.make_shared((256,), al.f32)
    s_sq = al.make_shared((256,), al.f32)

    my_sum = al.convert(0.0, al.f32)
    my_sq = al.convert(0.0, al.f32)
    for i in al.range(tid, M, 256):
        val = al.convert(x[i, col], al.f32)
        my_sum = my_sum + val
        my_sq = my_sq + val * val

    s_sum[tid] = my_sum
    s_sq[tid] = my_sq
    al.syncthreads()

    if tid < 128:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 128]
        s_sq[tid] = s_sq[tid] + s_sq[tid + 128]
    al.syncthreads()
    if tid < 64:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 64]
        s_sq[tid] = s_sq[tid] + s_sq[tid + 64]
    al.syncthreads()
    if tid < 32:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 32]
        s_sq[tid] = s_sq[tid] + s_sq[tid + 32]
    al.syncthreads()
    if tid < 16:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 16]
        s_sq[tid] = s_sq[tid] + s_sq[tid + 16]
    al.syncthreads()
    if tid < 8:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 8]
        s_sq[tid] = s_sq[tid] + s_sq[tid + 8]
    al.syncthreads()
    if tid < 4:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 4]
        s_sq[tid] = s_sq[tid] + s_sq[tid + 4]
    al.syncthreads()
    if tid < 2:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 2]
        s_sq[tid] = s_sq[tid] + s_sq[tid + 2]
    al.syncthreads()
    if tid == 0:
        total_sum = s_sum[0] + s_sum[1]
        total_sq = s_sq[0] + s_sq[1]
        m_f32 = al.convert(M, al.f32)
        mean_val = total_sum / m_f32
        var_val = total_sq / m_f32 - mean_val * mean_val

        mean_layout = al.make_layout((N,), (1,))
        var_layout = al.make_layout((N,), (1,))
        mean = al.make_tensor(mean_ptr, al.f32, mean_layout)
        var = al.make_tensor(var_ptr, al.f32, var_layout)
        mean[col] = mean_val
        var[col] = var_val


# ============================================================
# Kernel 3: BatchNorm + GELU + ReLU (BF16 in/out, FP32 mean/var)
# ============================================================

@avelang.jit
def bn_gelu_relu_kernel(
    x_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.bf16),
    mean_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    block_m = al.block_id(0)
    block_n = al.block_id(1)
    tid = al.thread_id(0)

    row = block_m * 128 + tid
    col_start = block_n * 128

    x_layout = al.make_layout((M, N), (N, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    y_layout = al.make_layout((M, N), (N, 1))
    y = al.make_tensor(y_ptr, al.bf16, y_layout)
    mean_layout = al.make_layout((N,), (1,))
    mean = al.make_tensor(mean_ptr, al.f32, mean_layout)
    var_layout = al.make_layout((N,), (1,))
    var = al.make_tensor(var_ptr, al.f32, var_layout)
    gamma_layout = al.make_layout((N,), (1,))
    gamma = al.make_tensor(gamma_ptr, al.bf16, gamma_layout)
    beta_layout = al.make_layout((N,), (1,))
    beta = al.make_tensor(beta_ptr, al.bf16, beta_layout)

    if row < M:
        for n in al.range(0, 128):
            col = col_start + n
            if col < N:
                x_val = al.convert(x[row, col], al.f32)
                m = mean[col]
                v = var[col]

                # BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
                inv_std = al.convert(1.0, al.f32) / al.sqrt(v + al.convert(1e-5, al.f32))
                normed = (x_val - m) * inv_std
                g = al.convert(gamma[col], al.f32)
                b = al.convert(beta[col], al.f32)
                bn_out = normed * g + b

                # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                x3 = bn_out * bn_out * bn_out
                inner = al.convert(0.7978845608, al.f32) * (bn_out + al.convert(0.044715, al.f32) * x3)
                gelu_out = al.convert(0.5, al.f32) * bn_out * (al.convert(1.0, al.f32) + al.tanh(inner))

                # ReLU via (x + |x|) / 2
                relu_out = (gelu_out + al.abs(gelu_out)) * al.convert(0.5, al.f32)

                y[row, col] = al.convert(relu_out, al.bf16)


# ============================================================
# ModelNew
# ============================================================

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        M = x.shape[0]
        N = self.out_features
        K = self.in_features

        x = x.contiguous()
        weight = self.gemm.weight.data.contiguous()
        bias = self.gemm.bias.data.contiguous()
        bn_weight = self.batch_norm.weight.data.contiguous()
        bn_bias = self.batch_norm.bias.data.contiguous()

        # Intermediate GEMM output: (M, N)
        gemm_out = torch.empty(M, N, dtype=x.dtype, device=x.device)

        grid_m = (M + 127) // 128
        grid_n = (N + 127) // 128

        gemm_kernel[lambda: ((grid_m, grid_n, 1), (128, 1, 1))](
            x, weight, gemm_out, bias,
            M, N, K,
        )

        # BatchNorm statistics: training vs eval
        if self.training:
            mean = torch.empty(N, dtype=torch.float32, device=x.device)
            var = torch.empty(N, dtype=torch.float32, device=x.device)
            bn_stats_kernel[lambda: ((N, 1, 1), (256, 1, 1))](
                gemm_out, mean, var,
                M, N,
            )
        else:
            mean = self.batch_norm.running_mean.data.to(torch.float32).contiguous()
            var = self.batch_norm.running_var.data.to(torch.float32).contiguous()

        # BatchNorm + GELU + ReLU
        out = torch.empty(M, N, dtype=x.dtype, device=x.device)

        bn_gelu_relu_kernel[lambda: ((grid_m, grid_n, 1), (128, 1, 1))](
            gemm_out, out, mean, var,
            bn_weight, bn_bias,
            M, N,
        )

        return out
