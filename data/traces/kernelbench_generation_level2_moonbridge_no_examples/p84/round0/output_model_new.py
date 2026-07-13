import torch
import torch.nn
import avelang
import avelang.language as al


# ---------------------------------------------------------------------------
# Kernel 1: Tiled GEMM with bias (BF16→FP32 output)
# ---------------------------------------------------------------------------
@avelang.jit
def gemm_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.f32),
    c_ptr: al.Pointer(al.f32),
    bias_ptr: al.Pointer(al.f32),
    M: al.i32,
    N: al.i32,
    K: al.i32,
):
    pid_m = al.block_id(0)
    pid_n = al.block_id(1)
    tid = al.thread_id(0)
    tid_m = tid % 16
    tid_n = tid // 16

    a = al.make_tensor(a_ptr, al.bf16, al.make_layout((M, K), (K, 1)))
    b = al.make_tensor(b_ptr, al.f32, al.make_layout((K, N), (1, K)))
    c = al.make_tensor(c_ptr, al.f32, al.make_layout((M, N), (N, 1)))
    bias = al.make_tensor(bias_ptr, al.f32, al.make_layout((N,), (1,)))

    a_sh = al.make_shared((128, 32), al.bf16)
    b_sh = al.make_shared((32, 64), al.f32)

    acc = al.make_local((8, 8), al.f32)
    for mi in al.range(8):
        for ni in al.range(8):
            acc[mi, ni] = al.convert(0.0, al.f32)

    offs_m = pid_m * 128
    offs_n = pid_n * 64

    for k_block in al.range(0, K, 32):
        for load_i in al.range(0, 4096, 128):
            load_idx = tid + load_i
            load_m = load_idx // 32
            load_k = load_idx % 32
            a_sh[load_m, load_k] = a[offs_m + load_m, k_block + load_k]

        for load_i in al.range(0, 2048, 128):
            load_idx = tid + load_i
            load_k = load_idx // 64
            load_n = load_idx % 64
            b_sh[load_k, load_n] = b[k_block + load_k, offs_n + load_n]

        al.syncthreads()

        for mi in al.range(8):
            m_local = tid_m + mi * 16
            for ni in al.range(8):
                n_local = tid_n + ni * 8
                inner = al.convert(0.0, al.f32)
                for k_loc in al.range(32):
                    a_val = al.convert(a_sh[m_local, k_loc], al.f32)
                    b_val = b_sh[k_loc, n_local]
                    inner = inner + a_val * b_val
                acc[mi, ni] = acc[mi, ni] + inner

        al.syncthreads()

    for mi in al.range(8):
        m_global = offs_m + tid_m + mi * 16
        for ni in al.range(8):
            n_global = offs_n + tid_n + ni * 8
            out_val = acc[mi, ni] + bias[n_global]
            c[m_global, n_global] = out_val



# ---------------------------------------------------------------------------
# Kernel 2: BatchNorm statistics (FP32 in, FP32 out)
# ---------------------------------------------------------------------------
@avelang.jit
def bn_stats_kernel(
    x_ptr: al.Pointer(al.f32),
    mean_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    M: al.i32,
    N: al.i32,
):
    pid = al.block_id(0)
    tid = al.thread_id(0)

    x = al.make_tensor(x_ptr, al.f32, al.make_layout((M, N), (N, 1)))
    mean = al.make_tensor(mean_ptr, al.f32, al.make_layout((N,), (1,)))
    var = al.make_tensor(var_ptr, al.f32, al.make_layout((N,), (1,)))

    if pid < N:
        local_sum = al.convert(0.0, al.f32)
        for i in al.range(tid, M, 256):
            local_sum = local_sum + x[i, pid]

        sh = al.make_shared((256,), al.f32)
        sh[tid] = local_sum
        al.syncthreads()

        if tid < 128:
            sh[tid] = sh[tid] + sh[tid + 128]
        al.syncthreads()
        if tid < 64:
            sh[tid] = sh[tid] + sh[tid + 64]
        al.syncthreads()
        if tid < 32:
            sh[tid] = sh[tid] + sh[tid + 32]
        al.syncthreads()
        if tid < 16:
            sh[tid] = sh[tid] + sh[tid + 16]
        al.syncthreads()
        if tid < 8:
            sh[tid] = sh[tid] + sh[tid + 8]
        al.syncthreads()
        if tid < 4:
            sh[tid] = sh[tid] + sh[tid + 4]
        al.syncthreads()
        if tid < 2:
            sh[tid] = sh[tid] + sh[tid + 2]
        al.syncthreads()
        if tid < 1:
            sh[tid] = sh[tid] + sh[tid + 1]
        al.syncthreads()

        col_mean = sh[0] / al.convert(M, al.f32)
        if tid == 0:
            mean[pid] = col_mean

        local_var_sum = al.convert(0.0, al.f32)
        for i in al.range(tid, M, 256):
            diff = x[i, pid] - col_mean
            local_var_sum = local_var_sum + diff * diff

        sh[tid] = local_var_sum
        al.syncthreads()

        if tid < 128:
            sh[tid] = sh[tid] + sh[tid + 128]
        al.syncthreads()
        if tid < 64:
            sh[tid] = sh[tid] + sh[tid + 64]
        al.syncthreads()
        if tid < 32:
            sh[tid] = sh[tid] + sh[tid + 32]
        al.syncthreads()
        if tid < 16:
            sh[tid] = sh[tid] + sh[tid + 16]
        al.syncthreads()
        if tid < 8:
            sh[tid] = sh[tid] + sh[tid + 8]
        al.syncthreads()
        if tid < 4:
            sh[tid] = sh[tid] + sh[tid + 4]
        al.syncthreads()
        if tid < 2:
            sh[tid] = sh[tid] + sh[tid + 2]
        al.syncthreads()
        if tid < 1:
            sh[tid] = sh[tid] + sh[tid + 1]
        al.syncthreads()

        col_var = sh[0] / al.convert(M, al.f32)
        if tid == 0:
            var[pid] = col_var


# ---------------------------------------------------------------------------
# Kernel 3: BN apply + scale (FP32 in, FP32 out)
# ---------------------------------------------------------------------------
@avelang.jit
def bn_apply_scale_kernel(
    x_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    mean_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    scale_eps_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    pid_m = al.block_id(0)
    pid_n = al.block_id(1)
    tid = al.thread_id(0)
    offs_m = pid_m * 32
    offs_n = pid_n * 64

    x = al.make_tensor(x_ptr, al.f32, al.make_layout((M, N), (N, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((M, N), (N, 1)))
    mean = al.make_tensor(mean_ptr, al.f32, al.make_layout((N,), (1,)))
    var = al.make_tensor(var_ptr, al.f32, al.make_layout((N,), (1,)))
    gamma = al.make_tensor(gamma_ptr, al.bf16, al.make_layout((N,), (1,)))
    beta = al.make_tensor(beta_ptr, al.bf16, al.make_layout((N,), (1,)))
    se = al.make_tensor(scale_eps_ptr, al.bf16, al.make_layout((2,), (1,)))

    scale_val = al.convert(se[0], al.f32)
    eps_val = al.convert(se[1], al.f32)

    for i in al.range(0, 2048, 256):
        idx = tid + i
        loc_m = idx // 64
        loc_n = idx % 64
        mm = offs_m + loc_m
        nn = offs_n + loc_n
        if mm < M:
            if nn < N:
                x_val = x[mm, nn]
                m_val = mean[nn]
                v_val = var[nn]
                sq_arg = v_val + eps_val
                denom = al.sqrt(sq_arg)
                g_val = al.convert(gamma[nn], al.f32)
                b_val = al.convert(beta[nn], al.f32)
                shifted = x_val - m_val
                normed = shifted / denom * g_val + b_val
                scaled = normed * scale_val
                out[mm, nn] = scaled


# ---------------------------------------------------------------------------
# Kernel 4: Row-wise softmax (FP32 in, BF16 out)
# ---------------------------------------------------------------------------
@avelang.jit
def softmax_kernel(
    x_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    pid = al.block_id(0)
    tid = al.thread_id(0)

    x = al.make_tensor(x_ptr, al.f32, al.make_layout((M, N), (N, 1)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((M, N), (N, 1)))

    if pid < M:
        row = pid

        local_max = al.convert(-1.0, al.f32)
        for j in al.range(tid, N, 256):
            val = x[row, j]
            if val > local_max:
                local_max = val

        sh = al.make_shared((256,), al.f32)
        sh[tid] = local_max
        al.syncthreads()

        if tid < 128:
            if sh[tid + 128] > sh[tid]:
                sh[tid] = sh[tid + 128]
        al.syncthreads()
        if tid < 64:
            if sh[tid + 64] > sh[tid]:
                sh[tid] = sh[tid + 64]
        al.syncthreads()
        if tid < 32:
            if sh[tid + 32] > sh[tid]:
                sh[tid] = sh[tid + 32]
        al.syncthreads()
        if tid < 16:
            if sh[tid + 16] > sh[tid]:
                sh[tid] = sh[tid + 16]
        al.syncthreads()
        if tid < 8:
            if sh[tid + 8] > sh[tid]:
                sh[tid] = sh[tid + 8]
        al.syncthreads()
        if tid < 4:
            if sh[tid + 4] > sh[tid]:
                sh[tid] = sh[tid + 4]
        al.syncthreads()
        if tid < 2:
            if sh[tid + 2] > sh[tid]:
                sh[tid] = sh[tid + 2]
        al.syncthreads()
        if tid < 1:
            if sh[tid + 1] > sh[tid]:
                sh[tid] = sh[tid + 1]
        al.syncthreads()

        row_max = sh[0]

        local_sum = al.convert(0.0, al.f32)
        for j in al.range(tid, N, 256):
            val = x[row, j]
            local_sum = local_sum + al.exp(val - row_max)

        sh[tid] = local_sum
        al.syncthreads()

        if tid < 128:
            sh[tid] = sh[tid] + sh[tid + 128]
        al.syncthreads()
        if tid < 64:
            sh[tid] = sh[tid] + sh[tid + 64]
        al.syncthreads()
        if tid < 32:
            sh[tid] = sh[tid] + sh[tid + 32]
        al.syncthreads()
        if tid < 16:
            sh[tid] = sh[tid] + sh[tid + 16]
        al.syncthreads()
        if tid < 8:
            sh[tid] = sh[tid] + sh[tid + 8]
        al.syncthreads()
        if tid < 4:
            sh[tid] = sh[tid] + sh[tid + 4]
        al.syncthreads()
        if tid < 2:
            sh[tid] = sh[tid] + sh[tid + 2]
        al.syncthreads()
        if tid < 1:
            sh[tid] = sh[tid] + sh[tid + 1]
        al.syncthreads()

        row_sum = sh[0]
        inv_sum = al.convert(1.0, al.f32) / row_sum

        for j in al.range(tid, N, 256):
            val = x[row, j]
            out[row, j] = al.convert(al.exp(val - row_max) * inv_sum, al.bf16)


# ---------------------------------------------------------------------------
# ModelNew
# ---------------------------------------------------------------------------
class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = torch.nn.Linear(in_features, out_features)
        self.bn = torch.nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bn_eps = bn_eps
        self.scale = torch.nn.Parameter(torch.ones(scale_shape))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        M = x.shape[0]
        K = self.in_features
        N = self.out_features

        x_bf16 = x.to(torch.bfloat16).contiguous()
        w = self.gemm.weight.contiguous()
        bias_f32 = self.gemm.bias.contiguous()

        gemm_out = torch.empty(M, N, dtype=torch.float32, device=x.device)
        grid_m = (M + 127) // 128
        grid_n = (N + 63) // 64
        gemm_kernel[lambda: ((grid_m, grid_n, 1), (128, 1, 1))](
            x_bf16, w, gemm_out, bias_f32, M, N, K,
        )

        mean = torch.empty(N, dtype=torch.float32, device=x.device)
        var = torch.empty(N, dtype=torch.float32, device=x.device)
        bn_stats_kernel[lambda: ((N, 1, 1), (256, 1, 1))](
            gemm_out, mean, var, M, N,
        )

        bn_out = torch.empty(M, N, dtype=torch.float32, device=x.device)
        gamma = self.bn.weight.contiguous()
        beta = self.bn.bias.contiguous()
        scale_f = float(self.scale.item())
        scale_eps = torch.tensor([scale_f, self.bn_eps], dtype=torch.bfloat16, device=x.device)
        grid_m_bn = (M + 31) // 32
        grid_n_bn = (N + 63) // 64
        bn_apply_scale_kernel[lambda: ((grid_m_bn, grid_n_bn, 1), (256, 1, 1))](
            gemm_out, bn_out, mean, var, gamma, beta, scale_eps, M, N,
        )

        softmax_out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        softmax_kernel[lambda: ((M, 1, 1), (256, 1, 1))](
            bn_out, softmax_out, M, N,
        )

        return softmax_out
