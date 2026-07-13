import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Tile sizes for the matmul kernel
BM = 16
BN = 16
BK = 16


@avelang.jit
def matmul_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    Bias_ptr: al.Pointer(al.f32),
    C_ptr: al.Pointer(al.f32),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    pid_m = al.block_id(0)
    pid_n = al.block_id(1)
    tid_m = al.thread_id(0)
    tid_n = al.thread_id(1)

    row = pid_m * BM + tid_m
    col = pid_n * BN + tid_n

    layout_a = al.make_layout((M, K), (K, 1))
    A = al.make_tensor(A_ptr, al.bf16, layout_a)
    layout_b = al.make_layout((N, K), (K, 1))
    B = al.make_tensor(B_ptr, al.bf16, layout_b)

    acc = al.convert(0.0, al.f32)

    As = al.make_shared((BM, BK), al.bf16)
    Bs = al.make_shared((BK, BN), al.bf16)

    num_k_blocks = K // BK
    for k_block in al.range(num_k_blocks):
        k_start = k_block * BK

        if row < M:
            if (k_start + tid_n) < K:
                As[tid_m, tid_n] = A[row, k_start + tid_n]
        if col < N:
            if (k_start + tid_m) < K:
                Bs[tid_m, tid_n] = B[col, k_start + tid_m]

        al.syncthreads()

        if row < M:
            if col < N:
                for kk in al.range(BK):
                    a_val = al.convert(As[tid_m, kk], al.f32)
                    b_val = al.convert(Bs[kk, tid_n], al.f32)
                    acc = acc + a_val * b_val

        al.syncthreads()

    if row < M:
        if col < N:
            layout_bias = al.make_layout((N,), (1,))
            Bias = al.make_tensor(Bias_ptr, al.f32, layout_bias)
            layout_c = al.make_layout((M, N), (N, 1))
            C = al.make_tensor(C_ptr, al.f32, layout_c)
            C[row, col] = acc + Bias[col]


@avelang.jit
def fused_reduce_kernel(
    Mat_ptr: al.Pointer(al.f32),
    Out_ptr: al.Pointer(al.bf16),
    scale: al.constexpr,
    M: al.i32,
    N: al.i32,
    pool_size: al.i32,
):
    row = al.block_id(0)

    if row < M:
        layout_mat = al.make_layout((M, N), (N, 1))
        Mat = al.make_tensor(Mat_ptr, al.f32, layout_mat)

        num_pools = N // pool_size
        max_val = al.convert(-3.402823e38, al.f32)

        sqrt_2_over_pi = al.convert(0.7978845608028654, al.f32)
        gelu_coeff = al.convert(0.044715, al.f32)
        half = al.convert(0.5, al.f32)
        one = al.convert(1.0, al.f32)
        pool_size_f = al.convert(pool_size, al.f32)
        scale_f = al.convert(scale, al.f32)

        for p in al.range(num_pools):
            pool_start = p * pool_size
            pool_sum = al.convert(0.0, al.f32)
            for k in al.range(pool_size):
                pool_sum = pool_sum + Mat[row, pool_start + k]

            avg = pool_sum / pool_size_f

            x3 = avg * avg * avg
            inner = sqrt_2_over_pi * (avg + gelu_coeff * x3)
            tanh_val = al.tanh(inner)
            gelu_val = half * avg * (one + tanh_val)

            scaled = gelu_val * scale_f

            if scaled > max_val:
                max_val = scaled

        layout_out = al.make_layout((M,), (1,))
        Out = al.make_tensor(Out_ptr, al.bf16, layout_out)
        Out[row] = al.convert(max_val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5.0 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / (fan_in ** 0.5) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size = x.shape[0]
        M = batch_size
        N = self.out_features
        K = self.in_features
        pool_size = self.pool_kernel_size

        # Cast input and weight to BF16 for the matmul
        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = self.weight.to(torch.bfloat16).contiguous()
        bias_f32 = self.bias.to(torch.float32).contiguous()

        # Intermediate matmul output: (M, N) in FP32
        mat_out = torch.empty(M, N, dtype=torch.float32, device=x.device)

        # Launch matmul kernel
        grid_m = (M + BM - 1) // BM
        grid_n = (N + BN - 1) // BN
        matmul_kernel[lambda: ((grid_m, grid_n, 1), (BM, BN, 1))](
            x_bf16, w_bf16, bias_f32, mat_out,
            M, N, K,
        )

        # Output: (M,) in BF16 to match reference dtype
        output = torch.empty(M, dtype=torch.bfloat16, device=x.device)

        # Launch fused reduce kernel with scale as constexpr
        fused_reduce_kernel[lambda: ((M, 1, 1), (1, 1, 1))](
            mat_out, output,
            float(self.scale_factor),
            M, N, pool_size,
        )

        return output
