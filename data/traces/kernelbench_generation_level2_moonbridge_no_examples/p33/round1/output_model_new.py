import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def gemm_scale_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    scale_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    pid_m = al.block_id(1)
    pid_n = al.block_id(0)
    tid_m = al.thread_id(1)
    tid_n = al.thread_id(0)

    # Global tensor views
    layout_a = al.make_layout((M, K), (K, 1))
    a = al.make_tensor(a_ptr, al.bf16, layout_a)
    layout_b = al.make_layout((N, K), (K, 1))
    b = al.make_tensor(b_ptr, al.bf16, layout_b)
    layout_bias = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.f32, layout_bias)
    layout_scale = al.make_layout((N,), (1,))
    scale = al.make_tensor(scale_ptr, al.bf16, layout_scale)
    layout_c = al.make_layout((M, N), (N, 1))
    c = al.make_tensor(c_ptr, al.bf16, layout_c)

    # Shared memory tiles: BM=64, BN=64, BK=64
    a_tile = al.make_shared((64, 64), al.bf16)
    b_tile = al.make_shared((64, 64), al.bf16)

    # Initialize accumulator (4x4 FP32 per thread)
    acc = al.full((4, 4), 0.0, al.f32)

    # Loop over K dimension in tiles of 64
    for k_block in al.range(0, K, 64):
        # Cooperative load A tile: each thread loads 4x4 = 16 elements
        for r in al.range(4):
            m_row = pid_m * 64 + tid_m * 4 + r
            for c4 in al.range(4):
                k_col = k_block + tid_n * 4 + c4
                if m_row < M and k_col < K:
                    a_tile[tid_m * 4 + r, tid_n * 4 + c4] = a[m_row, k_col]
                else:
                    a_tile[tid_m * 4 + r, tid_n * 4 + c4] = al.convert(0.0, al.bf16)

        # Cooperative load B tile: each thread loads 4x4 = 16 elements
        for r in al.range(4):
            n_row = pid_n * 64 + tid_m * 4 + r
            for c4 in al.range(4):
                k_col = k_block + tid_n * 4 + c4
                if n_row < N and k_col < K:
                    b_tile[tid_m * 4 + r, tid_n * 4 + c4] = b[n_row, k_col]
                else:
                    b_tile[tid_m * 4 + r, tid_n * 4 + c4] = al.convert(0.0, al.bf16)

        al.syncthreads()

        # Inner product
        for kk in al.range(64):
            for r in al.range(4):
                a_val = al.convert(a_tile[tid_m * 4 + r, kk], al.f32)
                for c4 in al.range(4):
                    b_val = al.convert(b_tile[tid_n * 4 + c4, kk], al.f32)
                    acc[r, c4] = acc[r, c4] + a_val * b_val

        al.syncthreads()

    # Epilogue: add bias, multiply by scale, store to output
    for r in al.range(4):
        m_out = pid_m * 64 + tid_m * 4 + r
        for c4 in al.range(4):
            n_out = pid_n * 64 + tid_n * 4 + c4
            if m_out < M and n_out < N:
                val = acc[r, c4] + bias[n_out]
                val = val * al.convert(scale[n_out], al.f32)
                c[m_out, n_out] = al.convert(val, al.bf16)


@avelang.jit
def batchnorm_eval_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    running_mean_ptr: al.Pointer(al.f32),
    running_var_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    N: al.i32,
    C: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    layout_x = al.make_layout((N, C), (C, 1))
    x = al.make_tensor(x_ptr, al.bf16, layout_x)
    layout_out = al.make_layout((N, C), (C, 1))
    out = al.make_tensor(out_ptr, al.bf16, layout_out)
    layout_1d = al.make_layout((C,), (1,))
    running_mean = al.make_tensor(running_mean_ptr, al.f32, layout_1d)
    running_var = al.make_tensor(running_var_ptr, al.f32, layout_1d)
    gamma = al.make_tensor(gamma_ptr, al.f32, layout_1d)
    beta = al.make_tensor(beta_ptr, al.f32, layout_1d)

    eps_f32 = al.convert(1e-5, al.f32)

    block_size = 256
    elems_per_block = N * C
    start = bid * block_size + tid

    if start < elems_per_block:
        i = start // C
        j = start % C

        x_val = al.convert(x[i, j], al.f32)
        mean_val = running_mean[j]
        var_val = running_var[j]

        inv_std = al.convert(1.0, al.f32) / al.sqrt(var_val + eps_f32)
        gamma_val = gamma[j]
        beta_val = beta[j]

        norm_val = (x_val - mean_val) * inv_std
        out_val = norm_val * gamma_val + beta_val
        out[i, j] = al.convert(out_val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        M, K = x.shape
        N = self.gemm.out_features

        x_bf16 = x.to(torch.bfloat16).contiguous()
        weight_bf16 = self.gemm.weight.data.to(torch.bfloat16).contiguous()
        scale_bf16 = self.scale.data.to(torch.bfloat16).contiguous()
        bias_f32 = self.gemm.bias.data.to(torch.float32).contiguous()

        gemm_out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

        grid_n = (N + 63) // 64
        grid_m = (M + 63) // 64

        gemm_scale_kernel[lambda: ((grid_n, grid_m, 1), (16, 16, 1))](
            x_bf16,
            weight_bf16,
            bias_f32,
            scale_bf16,
            gemm_out,
            M,
            N,
            K,
        )

        bn_out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        running_mean_f32 = self.bn.running_mean.data.to(torch.float32).contiguous()
        running_var_f32 = self.bn.running_var.data.to(torch.float32).contiguous()
        gamma_f32 = self.bn.weight.data.to(torch.float32).contiguous()
        beta_f32 = self.bn.bias.data.to(torch.float32).contiguous()

        total_elems = M * N
        num_blocks = (total_elems + 255) // 256

        batchnorm_eval_kernel[lambda: ((num_blocks, 1, 1), (256, 1, 1))](
            gemm_out,
            bn_out,
            running_mean_f32,
            running_var_f32,
            gamma_f32,
            beta_f32,
            M,
            N,
        )

        return bn_out


batch_size = 1024
in_features = 8192
out_features = 8192
scale_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, scale_shape]
