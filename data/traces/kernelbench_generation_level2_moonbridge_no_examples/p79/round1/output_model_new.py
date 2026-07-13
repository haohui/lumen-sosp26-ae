import torch
import torch.nn as nn
import avelang
import avelang.language as al


# ============================================================
# Kernel 1: InstanceNorm mean
# ============================================================
@avelang.jit
def instancenorm_mean_kernel(
    input_ptr: al.Pointer(al.f32),
    mean_ptr: al.Pointer(al.f32),
    N: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    spatial_size: al.i32,
    in_CDHW: al.i32,
    in_DHW: al.i32,
    in_HW: al.i32,
    mean_stride: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    n = al.block_id(1)
    c = al.block_id(0)
    tid = al.thread_id(0)

    in_size = N * in_CDHW
    in_layout = al.make_layout((in_size,), (1,))
    in_t = al.make_tensor(input_ptr, al.f32, in_layout)

    partial_sum = al.convert(0.0, al.f32)
    for i in al.range(tid, spatial_size, BLOCK_SIZE):
        d = i // in_HW
        rem = i - d * in_HW
        h = rem // W
        w = rem - h * W
        in_idx = n * in_CDHW + c * in_DHW + d * in_HW + h * W + w
        partial_sum = partial_sum + in_t[in_idx]

    sum_smem = al.make_shared((BLOCK_SIZE,), al.f32)
    sum_smem[tid] = partial_sum
    al.syncthreads()

    if tid == 0:
        total_sum = sum_smem[0]
        for i in al.range(1, BLOCK_SIZE):
            total_sum = total_sum + sum_smem[i]
        spatial_f = al.convert(spatial_size, al.f32)
        mean_val = total_sum / spatial_f
        sum_smem[0] = mean_val

    al.syncthreads()
    mean_val = sum_smem[0]

    mean_layout = al.make_layout((N * mean_stride,), (1,))
    mean_t = al.make_tensor(mean_ptr, al.f32, mean_layout)
    mean_t[n * mean_stride + c] = mean_val


# ============================================================
# Kernel 2: InstanceNorm normalize + Clamp + Second Multiply
# ============================================================
@avelang.jit
def instancenorm_norm_clamp_mul_kernel(
    input_ptr: al.Pointer(al.f32),
    mean_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    multiplier_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.f32),
    eps: al.constexpr,
    clamp_min: al.constexpr,
    clamp_max: al.constexpr,
    N: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    spatial_size: al.i32,
    in_CDHW: al.i32,
    in_DHW: al.i32,
    in_HW: al.i32,
    out_CDHW: al.i32,
    out_DHW: al.i32,
    out_HW: al.i32,
    mean_stride: al.i32,
    BLOCK_SIZE: al.constexpr,
):
    n = al.block_id(1)
    c = al.block_id(0)
    tid = al.thread_id(0)

    eps_val = al.convert(eps, al.f32)
    cmin_val = al.convert(clamp_min, al.f32)
    cmax_val = al.convert(clamp_max, al.f32)

    mean_layout = al.make_layout((N * mean_stride,), (1,))
    mean_t = al.make_tensor(mean_ptr, al.f32, mean_layout)
    mean_val = mean_t[n * mean_stride + c]

    in_size = N * in_CDHW
    in_layout = al.make_layout((in_size,), (1,))
    in_t = al.make_tensor(input_ptr, al.f32, in_layout)

    partial_var = al.convert(0.0, al.f32)
    for i in al.range(tid, spatial_size, BLOCK_SIZE):
        d = i // in_HW
        rem = i - d * in_HW
        h = rem // W
        w = rem - h * W
        in_idx = n * in_CDHW + c * in_DHW + d * in_HW + h * W + w
        diff = in_t[in_idx] - mean_val
        partial_var = partial_var + diff * diff

    var_smem = al.make_shared((BLOCK_SIZE,), al.f32)
    var_smem[tid] = partial_var
    al.syncthreads()

    if tid == 0:
        total_var = var_smem[0]
        for i in al.range(1, BLOCK_SIZE):
            total_var = total_var + var_smem[i]
        spatial_f = al.convert(spatial_size, al.f32)
        var_val = total_var / spatial_f
        if var_val < 0.0:
            var_val = 0.0
        var_smem[0] = var_val

    al.syncthreads()
    var_val = var_smem[0]

    gamma_layout = al.make_layout((C,), (1,))
    gamma_t = al.make_tensor(gamma_ptr, al.f32, gamma_layout)
    beta_layout = al.make_layout((C,), (1,))
    beta_t = al.make_tensor(beta_ptr, al.f32, beta_layout)
    mult_layout = al.make_layout((C,), (1,))
    mult_t = al.make_tensor(multiplier_ptr, al.bf16, mult_layout)

    gamma_val = gamma_t[c]
    beta_val = beta_t[c]
    mult_val = al.convert(mult_t[c], al.f32)
    inv_std = al.convert(1.0, al.f32) / al.sqrt(var_val + eps_val)

    out_size = N * out_CDHW
    out_layout = al.make_layout((out_size,), (1,))
    out_t = al.make_tensor(output_ptr, al.f32, out_layout)

    for i in al.range(tid, spatial_size, BLOCK_SIZE):
        d = i // out_HW
        rem = i - d * out_HW
        h = rem // W
        w = rem - h * W
        in_idx = n * in_CDHW + c * in_DHW + d * in_HW + h * W + w
        val = in_t[in_idx]
        norm = (val - mean_val) * inv_std * gamma_val + beta_val
        if norm < cmin_val:
            norm = cmin_val
        if norm > cmax_val:
            norm = cmax_val
        result = norm * mult_val
        out_idx = n * out_CDHW + c * out_DHW + d * out_HW + h * W + w
        out_t[out_idx] = result


# ============================================================
# Kernel 3: Max reduction over channel dimension
# ============================================================
@avelang.jit
def max_reduce_kernel(
    input_ptr: al.Pointer(al.f32),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    total_elements: al.i32,
    in_CDHW: al.i32,
    in_DHW: al.i32,
    in_HW: al.i32,
    out_DHW: al.i32,
    out_HW: al.i32,
):
    idx = al.block_id(0) * al.block_dim(0) + al.thread_id(0)
    if idx >= total_elements:
        return

    n = idx // out_DHW
    rem = idx - n * out_DHW
    d = rem // out_HW
    rem = rem - d * out_HW
    h = rem // W
    w = rem - h * W

    in_size = N * in_CDHW
    in_layout = al.make_layout((in_size,), (1,))
    in_t = al.make_tensor(input_ptr, al.f32, in_layout)

    in_idx = n * in_CDHW + 0 * in_DHW + d * in_HW + h * W + w
    max_val = in_t[in_idx]

    for c in al.range(1, C):
        in_idx = n * in_CDHW + c * in_DHW + d * in_HW + h * W + w
        val = in_t[in_idx]
        if val > max_val:
            max_val = val

    out_size = N * out_DHW
    out_layout = al.make_layout((out_size,), (1,))
    out_t = al.make_tensor(output_ptr, al.bf16, out_layout)
    out_idx = n * out_DHW + d * out_HW + h * W + w
    out_t[out_idx] = al.convert(max_val, al.bf16)


# ============================================================
# Host wrapper
# ============================================================
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        assert x.is_cuda, "Input must be on CUDA/HIP device"
        x = x.contiguous()
        N, _, D, H, W = x.shape
        C_out = self.conv.out_channels
        K = self.conv.kernel_size[0]
        D_out = D - K + 1
        H_out = H - K + 1
        W_out = W - K + 1

        # Convolution + first multiply via PyTorch (exact match with reference)
        x = self.conv(x)
        x = x * self.multiplier
        x_f32 = x.to(torch.float32).contiguous()

        # Parameters for AveLang kernels
        multiplier = self.multiplier.data.to(torch.bfloat16).contiguous()
        gamma = self.instance_norm.weight
        gamma = gamma.data.to(torch.float32).contiguous() if gamma is not None else torch.ones(C_out, dtype=torch.float32, device=x.device)
        beta = self.instance_norm.bias
        beta = beta.data.to(torch.float32).contiguous() if beta is not None else torch.zeros(C_out, dtype=torch.float32, device=x.device)
        eps = float(self.instance_norm.eps)
        clamp_min = float(self.clamp_min)
        clamp_max = float(self.clamp_max)

        out_CoutDoutHWout = C_out * D_out * H_out * W_out
        out_DoutHWout = D_out * H_out * W_out
        out_HWout = H_out * W_out
        spatial_size = D_out * H_out * W_out
        total_max = N * D_out * H_out * W_out
        BLOCK_SIZE = 256

        mean_buf = torch.empty(N, C_out, dtype=torch.float32, device=x.device)
        norm_out = torch.empty(N, C_out, D_out, H_out, W_out, dtype=torch.float32, device=x.device)
        output = torch.empty(N, D_out, H_out, W_out, dtype=x.dtype, device=x.device)

        grid_norm = (C_out, N, 1)
        instancenorm_mean_kernel[lambda: (grid_norm, (BLOCK_SIZE, 1, 1))](
            x_f32, mean_buf,
            N, C_out, D_out, H_out, W_out,
            spatial_size,
            out_CoutDoutHWout, out_DoutHWout, out_HWout,
            C_out,
            BLOCK_SIZE,
        )

        instancenorm_norm_clamp_mul_kernel[lambda: (grid_norm, (BLOCK_SIZE, 1, 1))](
            x_f32, mean_buf, gamma, beta, multiplier, norm_out,
            eps, clamp_min, clamp_max,
            N, C_out, D_out, H_out, W_out,
            spatial_size,
            out_CoutDoutHWout, out_DoutHWout, out_HWout,
            out_CoutDoutHWout, out_DoutHWout, out_HWout,
            C_out,
            BLOCK_SIZE,
        )

        grid_max = ((total_max + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
        max_reduce_kernel[lambda: (grid_max, (BLOCK_SIZE, 1, 1))](
            norm_out, output,
            N, C_out, D_out, H_out, W_out,
            total_max,
            out_CoutDoutHWout, out_DoutHWout, out_HWout,
            D_out * H_out * W_out, H_out * W_out,
        )

        return output
