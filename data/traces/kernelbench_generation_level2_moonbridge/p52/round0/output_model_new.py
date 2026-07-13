import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Problem dimensions
BATCH_SIZE = 64
IN_CHANNELS = 64
OUT_CHANNELS = 128
H, W = 128, 128
KERNEL_SIZE = 3
H_OUT = H - KERNEL_SIZE + 1  # 126
W_OUT = W - KERNEL_SIZE + 1  # 126
K_TOTAL = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE  # 576
M_TOTAL = BATCH_SIZE * H_OUT * W_OUT  # 1016064
EPS = 1e-5

# Kernel launch constants
BLOCK = 256
IM2COL_BLOCK = 256
K_ELEMS_PER_THREAD = (K_TOTAL + IM2COL_BLOCK - 1) // IM2COL_BLOCK  # 3


@avelang.jit
def im2col_kernel(
    input_ptr: al.Pointer(al.f32),
    im2col_ptr: al.Pointer(al.f32),
    N: al.i32,
    C_in: al.i32,
    H: al.i32,
    W: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    K_size: al.i32,
    K_total: al.i32,
    M: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    m = bid
    if m < M:
        n = m // (H_out * W_out)
        rem = m - n * H_out * W_out
        h_out = rem // W_out
        w_out = rem - h_out * W_out

        in_layout = al.make_layout((N, C_in, H, W), (C_in * H * W, H * W, W, 1))
        inp = al.make_tensor(input_ptr, al.f32, in_layout)

        out_layout = al.make_layout((M, K_total), (K_total, 1))
        out = al.make_tensor(im2col_ptr, al.f32, out_layout)

        k_size_sq = K_size * K_size
        for step in al.range(K_ELEMS_PER_THREAD):
            k = tid + step * IM2COL_BLOCK
            if k < K_total:
                c_in = k // k_size_sq
                rem_k = k - c_in * k_size_sq
                kh = rem_k // K_size
                kw = rem_k - kh * K_size
                out[m, k] = inp[n, c_in, h_out + kh, w_out + kw]


@avelang.jit
def gemm_fused_mish_kernel(
    a_ptr: al.Pointer(al.f32),
    b_ptr: al.Pointer(al.f32),
    bias_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    m: al.i32,
    n: al.i32,
    k: al.i32,
):
    tid = al.thread_id(0)
    gid = al.block_id(0) * BLOCK + tid
    total_out = m * n

    a_mem = al.make_tensor(a_ptr, al.f32, al.make_layout((m * k,), (1,)))
    b_mem = al.make_tensor(b_ptr, al.f32, al.make_layout((n * k,), (1,)))
    g_bias = al.make_tensor(bias_ptr, al.f32, al.make_layout((n,), (1,)))
    g_out = al.make_tensor(out_ptr, al.f32, al.make_layout((m * n,), (1,)))

    if gid < total_out:
        row = gid // n
        col = gid - row * n

        val = al.convert(0.0, al.f32)
        for kk in al.range(k):
            val = val + a_mem[row * k + kk] * b_mem[col * k + kk]

        val = val + g_bias[col]

        # Mish activation: val * tanh(softplus(val))
        one_f = al.convert(1.0, al.f32)
        two_f = al.convert(2.0, al.f32)
        sp = al.log(one_f + al.exp(val))
        exp2sp = al.exp(two_f * sp)
        th = (exp2sp - one_f) / (exp2sp + one_f)

        g_out[gid] = val * th


@avelang.jit
def bn_eval_kernel(
    x_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    running_mean_ptr: al.Pointer(al.f32),
    running_var_ptr: al.Pointer(al.f32),
    weight_ptr: al.Pointer(al.f32),
    bias_ptr: al.Pointer(al.f32),
    num_elements: al.i32,
    num_channels: al.i32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    gid = al.block_id(0) * BLOCK + tid

    x_mem = al.make_tensor(x_ptr, al.f32, al.make_layout((num_elements * num_channels,), (1,)))
    out_mem = al.make_tensor(out_ptr, al.f32, al.make_layout((num_elements * num_channels,), (1,)))
    rm_mem = al.make_tensor(running_mean_ptr, al.f32, al.make_layout((num_channels,), (1,)))
    rv_mem = al.make_tensor(running_var_ptr, al.f32, al.make_layout((num_channels,), (1,)))
    wt_mem = al.make_tensor(weight_ptr, al.f32, al.make_layout((num_channels,), (1,)))
    bt_mem = al.make_tensor(bias_ptr, al.f32, al.make_layout((num_channels,), (1,)))

    if gid < num_elements * num_channels:
        channel = gid % num_channels
        x_val = x_mem[gid]
        mean_val = rm_mem[channel]
        var_val = rv_mem[channel]
        w_val = wt_mem[channel]
        b_val = bt_mem[channel]

        rstd = al.convert(1.0, al.f32) / al.sqrt(var_val + eps)
        normalized = (x_val - mean_val) * rstd
        out_mem[gid] = normalized * w_val + b_val


def _ensure_cuda_contiguous(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if t.is_cuda and t.dtype == dtype and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=dtype)
    return t.contiguous().cuda().to(dtype=dtype)


def avelang_conv_bn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_f32 = _ensure_cuda_contiguous(x, torch.float32)
    conv_weight_f32 = _ensure_cuda_contiguous(conv_weight, torch.float32)
    conv_bias_f32 = _ensure_cuda_contiguous(conv_bias, torch.float32)
    bn_weight_f32 = _ensure_cuda_contiguous(bn_weight, torch.float32)
    bn_bias_f32 = _ensure_cuda_contiguous(bn_bias, torch.float32)
    bn_rm_f32 = _ensure_cuda_contiguous(bn_running_mean, torch.float32)
    bn_rv_f32 = _ensure_cuda_contiguous(bn_running_var, torch.float32)

    N, C_in, H_in, W_in = x_f32.shape
    C_out = conv_weight_f32.shape[0]
    H_out = H_in - KERNEL_SIZE + 1
    W_out = W_in - KERNEL_SIZE + 1
    M = N * H_out * W_out
    K = C_in * KERNEL_SIZE * KERNEL_SIZE

    weight_2d = conv_weight_f32.reshape(C_out, K).contiguous()

    # Step 1: im2col in FP32
    im2col_buf = torch.empty((M, K), dtype=torch.float32, device=x_f32.device)
    im2col_grid = (M, 1, 1)
    im2col_kernel[lambda: (im2col_grid, (IM2COL_BLOCK, 1, 1))](
        x_f32, im2col_buf,
        N, C_in, H_in, W_in, H_out, W_out,
        KERNEL_SIZE, K, M,
    )

    # Step 2: GEMM with fused Mish activation, FP32 output
    total_out = M * C_out
    conv_flat = torch.empty(total_out, dtype=torch.float32, device=x_f32.device)
    gemm_grid = ((total_out + BLOCK - 1) // BLOCK, 1, 1)
    gemm_fused_mish_kernel[lambda: (gemm_grid, (BLOCK, 1, 1))](
        im2col_buf, weight_2d, conv_bias_f32, conv_flat,
        M, C_out, K,
    )

    del im2col_buf

    # Step 3: BatchNorm in eval mode
    bn_flat = torch.empty(total_out, dtype=torch.float32, device=x_f32.device)
    bn_grid = ((total_out + BLOCK - 1) // BLOCK, 1, 1)
    bn_eval_kernel[lambda: (bn_grid, (BLOCK, 1, 1))](
        conv_flat, bn_flat,
        bn_rm_f32, bn_rv_f32, bn_weight_f32, bn_bias_f32,
        M, C_out, EPS,
    )

    # Reshape to NCHW and convert to BF16
    result_4d = bn_flat.reshape(M, C_out).reshape(N, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()
    return result_4d.to(dtype=torch.bfloat16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        return avelang_conv_bn(
            x,
            self.conv.weight.data,
            self.conv.bias.data,
            self.bn.weight.data,
            self.bn.bias.data,
            self.bn.running_mean.data,
            self.bn.running_var.data,
        )


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_CHANNELS, H, W)]


def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]
