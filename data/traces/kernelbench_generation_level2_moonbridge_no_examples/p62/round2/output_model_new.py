import torch
import torch.nn as nn
import avelang
import avelang.language as al

BM = al.constexpr(64)
BN = al.constexpr(64)
BK = al.constexpr(32)

BM_VAL = 64
BN_VAL = 64
BK_VAL = 32


@avelang.jit
def matmul_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    K: al.i32,
    N: al.i32,
):
    pid_m = al.block_id(0)
    pid_n = al.block_id(1)
    tid = al.thread_id(0)

    a_layout = al.make_layout((M, K), (K, 1))
    a = al.make_tensor(a_ptr, al.bf16, a_layout)
    b_layout = al.make_layout((K, N), (N, 1))
    b = al.make_tensor(b_ptr, al.bf16, b_layout)
    bias_layout = al.make_layout((N,), (1,))
    bias = al.make_tensor(bias_ptr, al.bf16, bias_layout)
    out_layout = al.make_layout((M, N), (N, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    start_m = pid_m * BM
    start_n = pid_n * BN

    thread_row = tid // 16
    thread_col = tid % 16

    acc = al.full((4, 4), 0.0, al.f32)

    a_tile = al.make_shared((64, 32), al.bf16)
    b_tile = al.make_shared((32, 64), al.bf16)

    for kb in al.range(256):
        k_block = kb * BK

        for i in al.range(8):
            flat_idx = i * 256 + tid
            r = flat_idx // 32
            c = flat_idx % 32
            a_tile[r, c] = a[start_m + r, k_block + c]

        for i in al.range(8):
            flat_idx = i * 256 + tid
            r = flat_idx // 64
            c = flat_idx % 64
            b_tile[r, c] = b[k_block + r, start_n + c]

        al.syncthreads()

        for ki in al.range(32):
            for local_m in al.range(4):
                a_val = al.convert(a_tile[thread_row * 4 + local_m, ki], al.f32)
                for local_n in al.range(4):
                    b_val = al.convert(b_tile[ki, thread_col * 4 + local_n], al.f32)
                    acc[local_m, local_n] = acc[local_m, local_n] + a_val * b_val

        al.syncthreads()

    for local_m in al.range(4):
        global_r = start_m + thread_row * 4 + local_m
        for local_n in al.range(4):
            global_c = start_n + thread_col * 4 + local_n
            val = acc[local_m, local_n] + al.convert(bias[global_c], al.f32)
            out[global_r, global_c] = al.convert(val, al.bf16)


@avelang.jit
def groupnorm_leakyrelu_add_kernel(
    input_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    G: al.i32,
):
    pid = al.block_id(0)
    tid = al.thread_id(0)

    in_layout = al.make_layout((N, C), (C, 1))
    inp = al.make_tensor(input_ptr, al.bf16, in_layout)
    g_layout = al.make_layout((C,), (1,))
    gamma = al.make_tensor(gamma_ptr, al.bf16, g_layout)
    b_layout = al.make_layout((C,), (1,))
    beta_vec = al.make_tensor(b_ptr, al.bf16, b_layout)
    o_layout = al.make_layout((N, C), (C, 1))
    out = al.make_tensor(output_ptr, al.bf16, o_layout)

    eps_val = al.convert(0.00001, al.f32)
    neg_slope_val = al.convert(0.01, al.f32)
    zero_val = al.convert(0.0, al.f32)
    one_val = al.convert(1.0, al.f32)
    count_val = al.convert(16.0, al.f32)

    for g_offset in al.range(2):
        g = tid * 2 + g_offset
        if g < G:
            base_c = g * 16

            sum_val = al.convert(0.0, al.f32)
            for c in al.range(16):
                sum_val = sum_val + al.convert(inp[pid, base_c + c], al.f32)
            mean = sum_val / count_val

            sum_sq = al.convert(0.0, al.f32)
            for c in al.range(16):
                diff = al.convert(inp[pid, base_c + c], al.f32) - mean
                sum_sq = sum_sq + diff * diff
            var = sum_sq / count_val

            inv_std = one_val / al.sqrt(var + eps_val)

            for c in al.range(16):
                val = al.convert(inp[pid, base_c + c], al.f32)
                g_val = al.convert(gamma[base_c + c], al.f32)
                b_val = al.convert(beta_vec[base_c + c], al.f32)

                gn_out = (val - mean) * inv_std * g_val + b_val

                if gn_out >= zero_val:
                    out[pid, base_c + c] = al.convert(gn_out + gn_out, al.bf16)
                else:
                    neg_val = neg_slope_val * gn_out
                    out[pid, base_c + c] = al.convert(neg_val + neg_val, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc_weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.fc_bias = nn.Parameter(torch.empty(hidden_size))
        self.gn_weight = nn.Parameter(torch.empty(hidden_size))
        self.gn_bias = nn.Parameter(torch.empty(hidden_size))
        self.eps = eps
        self.negative_slope = negative_slope
        self.num_groups = num_groups

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc_weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc_weight)
        bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.fc_bias, -bound, bound)
        nn.init.ones_(self.gn_weight)
        nn.init.zeros_(self.gn_bias)

    def forward(self, x):
        orig_dtype = x.dtype
        device = x.device

        x_bf16 = x.to(torch.bfloat16).contiguous()

        w_t = self.fc_weight.T.contiguous().to(torch.bfloat16)
        bias_bf16 = self.fc_bias.to(torch.bfloat16).contiguous()

        M = x_bf16.shape[0]
        K = x_bf16.shape[1]
        N = w_t.shape[1]

        matmul_out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        grid_mm = ((M + BM_VAL - 1) // BM_VAL, (N + BN_VAL - 1) // BN_VAL, 1)
        block = (256, 1, 1)
        matmul_kernel[lambda: (grid_mm, block)](
            x_bf16, w_t, bias_bf16, matmul_out,
            M, K, N,
        )

        gamma_bf16 = self.gn_weight.to(torch.bfloat16).contiguous()
        beta_bf16 = self.gn_bias.to(torch.bfloat16).contiguous()

        final_out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        grid_gn = (M, 1, 1)
        groupnorm_leakyrelu_add_kernel[lambda: (grid_gn, block)](
            matmul_out, gamma_bf16, beta_bf16, final_out,
            M, N, self.num_groups,
        )

        return final_out.to(orig_dtype)
