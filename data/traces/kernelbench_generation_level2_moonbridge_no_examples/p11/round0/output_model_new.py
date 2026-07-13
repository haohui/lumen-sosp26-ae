import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ============================================================================
# Kernel 1: BatchNorm eval + Tanh fused — BF16 in/out
# ============================================================================
@avelang.jit
def batch_norm_tanh_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    running_mean_ptr: al.Pointer(al.bf16),
    running_var_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
):
    n = al.block_id(0)
    flat_idx = al.block_id(1) * al.block_dim(0) + al.thread_id(0)
    total = C * H * W

    in_layout = al.make_layout((N * total,), (1,))
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)
    out_layout = al.make_layout((N * total,), (1,))
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    mean_layout = al.make_layout((C,), (1,))
    running_mean_t = al.make_tensor(running_mean_ptr, al.bf16, mean_layout)
    running_var_t = al.make_tensor(running_var_ptr, al.bf16, mean_layout)

    gamma_layout = al.make_layout((C,), (1,))
    gamma_t = al.make_tensor(gamma_ptr, al.bf16, gamma_layout)
    beta_layout = al.make_layout((C,), (1,))
    beta_t = al.make_tensor(beta_ptr, al.bf16, beta_layout)

    if flat_idx < total:
        c = flat_idx // (H * W)
        rem = flat_idx - c * H * W
        h = rem // W
        w = rem - h * W

        in_idx = n * total + c * H * W + h * W + w
        val = al.convert(input_t[in_idx], al.f32)

        mean_val = al.convert(running_mean_t[c], al.f32)
        var_val = al.convert(running_var_t[c], al.f32)
        gamma_val = al.convert(gamma_t[c], al.f32)
        beta_val = al.convert(beta_t[c], al.f32)

        norm_val = (val - mean_val) / al.sqrt(var_val + al.convert(0.00001, al.f32))
        out_val = gamma_val * norm_val + beta_val
        two_x = out_val + out_val
        exp_val = al.exp(two_x)
        out_val = (exp_val - al.convert(1.0, al.f32)) / (exp_val + al.convert(1.0, al.f32))

        out_idx = n * total + c * H * W + h * W + w
        output_t[out_idx] = al.convert(out_val, al.bf16)


# ============================================================================
# Kernel 2: MaxPool2d — BF16 in/out
# ============================================================================
@avelang.jit
def max_pool2d_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    IH: al.i32,
    IW: al.i32,
    OH: al.i32,
    OW: al.i32,
):
    n = al.block_id(0)
    flat_idx = al.block_id(1) * al.block_dim(0) + al.thread_id(0)
    total_out = C * OH * OW

    in_layout = al.make_layout((N * C * IH * IW,), (1,))
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)
    out_layout = al.make_layout((N * C * OH * OW,), (1,))
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    if flat_idx < total_out:
        c = flat_idx // (OH * OW)
        rem = flat_idx - c * OH * OW
        oh = rem // OW
        ow = rem - oh * OW

        ih0 = oh * 2
        iw0 = ow * 2

        max_val = al.convert(-1.0e9, al.f32)

        for dh in al.range(2):
            ih = ih0 + dh
            for dw in al.range(2):
                iw = iw0 + dw
                in_idx = n * C * IH * IW + c * IH * IW + ih * IW + iw
                val = al.convert(input_t[in_idx], al.f32)
                if val > max_val:
                    max_val = val

        out_idx = n * C * OH * OW + c * OH * OW + oh * OW + ow
        output_t[out_idx] = al.convert(max_val, al.bf16)


# ============================================================================
# Kernel 3: GroupNorm sum — BF16 in, FP32 accumulation
# ============================================================================
@avelang.jit
def group_norm_sum_kernel(
    input_ptr: al.Pointer(al.bf16),
    sum_ptr: al.Pointer(al.f32),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    num_groups: al.i32,
):
    flat_block = al.block_id(0)
    n = flat_block // num_groups
    g = flat_block - n * num_groups
    tid = al.thread_id(0)
    num_threads = al.block_dim(0)

    s_sum = al.make_shared((256,), al.f32)

    in_layout = al.make_layout((N * C * H * W,), (1,))
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    C_PER_GROUP = C // num_groups
    c_start = g * C_PER_GROUP
    total_per_group = C_PER_GROUP * H * W
    c_stride = C * H * W

    thread_sum = al.convert(0.0, al.f32)

    for idx in al.range(tid, total_per_group, num_threads):
        c_offset = idx // (H * W)
        rem = idx - c_offset * H * W
        h = rem // W
        w = rem - h * W
        c = c_start + c_offset
        in_idx = n * c_stride + c * H * W + h * W + w
        val = al.convert(input_t[in_idx], al.f32)
        thread_sum = thread_sum + val

    s_sum[tid] = thread_sum
    al.syncthreads()

    if tid < 128:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 128]
    al.syncthreads()
    if tid < 64:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 64]
    al.syncthreads()
    if tid < 32:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 32]
    al.syncthreads()
    if tid < 16:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 16]
    al.syncthreads()
    if tid < 8:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 8]
    al.syncthreads()
    if tid < 4:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 4]
    al.syncthreads()
    if tid < 2:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 2]
    al.syncthreads()
    if tid < 1:
        s_sum[tid] = s_sum[tid] + s_sum[tid + 1]

    if tid == 0:
        sum_layout = al.make_layout((N * num_groups,), (1,))
        sum_t = al.make_tensor(sum_ptr, al.f32, sum_layout)
        sum_t[flat_block] = s_sum[0]


# ============================================================================
# Kernel 4: GroupNorm var — BF16 in, uses precomputed mean
# ============================================================================
@avelang.jit
def group_norm_var_kernel(
    input_ptr: al.Pointer(al.bf16),
    sum_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    num_groups: al.i32,
):
    flat_block = al.block_id(0)
    n = flat_block // num_groups
    g = flat_block - n * num_groups
    tid = al.thread_id(0)
    num_threads = al.block_dim(0)

    s_var = al.make_shared((256,), al.f32)

    in_layout = al.make_layout((N * C * H * W,), (1,))
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    sum_layout = al.make_layout((N * num_groups,), (1,))
    sum_t = al.make_tensor(sum_ptr, al.f32, sum_layout)

    C_PER_GROUP = C // num_groups
    c_start = g * C_PER_GROUP
    total_per_group = C_PER_GROUP * H * W
    c_stride = C * H * W
    count = al.convert(C_PER_GROUP * H * W, al.f32)

    mean = sum_t[flat_block] / count

    thread_var = al.convert(0.0, al.f32)

    for idx in al.range(tid, total_per_group, num_threads):
        c_offset = idx // (H * W)
        rem = idx - c_offset * H * W
        h = rem // W
        w = rem - h * W
        c = c_start + c_offset
        in_idx = n * c_stride + c * H * W + h * W + w
        val = al.convert(input_t[in_idx], al.f32)
        diff = val - mean
        thread_var = thread_var + diff * diff

    s_var[tid] = thread_var
    al.syncthreads()

    if tid < 128:
        s_var[tid] = s_var[tid] + s_var[tid + 128]
    al.syncthreads()
    if tid < 64:
        s_var[tid] = s_var[tid] + s_var[tid + 64]
    al.syncthreads()
    if tid < 32:
        s_var[tid] = s_var[tid] + s_var[tid + 32]
    al.syncthreads()
    if tid < 16:
        s_var[tid] = s_var[tid] + s_var[tid + 16]
    al.syncthreads()
    if tid < 8:
        s_var[tid] = s_var[tid] + s_var[tid + 8]
    al.syncthreads()
    if tid < 4:
        s_var[tid] = s_var[tid] + s_var[tid + 4]
    al.syncthreads()
    if tid < 2:
        s_var[tid] = s_var[tid] + s_var[tid + 2]
    al.syncthreads()
    if tid < 1:
        s_var[tid] = s_var[tid] + s_var[tid + 1]

    if tid == 0:
        var_layout = al.make_layout((N * num_groups,), (1,))
        var_t = al.make_tensor(var_ptr, al.f32, var_layout)
        var_t[flat_block] = s_var[0]


# ============================================================================
# Kernel 5: GroupNorm apply — BF16 in/out
# ============================================================================
@avelang.jit
def group_norm_apply_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    sum_ptr: al.Pointer(al.f32),
    var_ptr: al.Pointer(al.f32),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    num_groups: al.i32,
):
    n = al.block_id(0)
    flat_idx = al.block_id(1) * al.block_dim(0) + al.thread_id(0)
    total = C * H * W

    in_layout = al.make_layout((N * total,), (1,))
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)
    out_layout = al.make_layout((N * total,), (1,))
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    sum_layout = al.make_layout((N * num_groups,), (1,))
    sum_t = al.make_tensor(sum_ptr, al.f32, sum_layout)
    var_t = al.make_tensor(var_ptr, al.f32, sum_layout)

    gamma_layout = al.make_layout((C,), (1,))
    gamma_t = al.make_tensor(gamma_ptr, al.bf16, gamma_layout)
    beta_layout = al.make_layout((C,), (1,))
    beta_t = al.make_tensor(beta_ptr, al.bf16, beta_layout)

    C_PER_GROUP = C // num_groups
    count = al.convert(C_PER_GROUP * H * W, al.f32)

    if flat_idx < total:
        c = flat_idx // (H * W)
        rem = flat_idx - c * H * W
        h = rem // W
        w = rem - h * W

        g = c // C_PER_GROUP
        sum_idx = n * num_groups + g

        in_idx = n * total + c * H * W + h * W + w
        val = al.convert(input_t[in_idx], al.f32)

        mean = sum_t[sum_idx] / count
        variance = var_t[sum_idx] / count

        gamma_val = al.convert(gamma_t[c], al.f32)
        beta_val = al.convert(beta_t[c], al.f32)

        norm_val = (val - mean) / al.sqrt(variance + al.convert(0.00001, al.f32))
        out_val = gamma_val * norm_val + beta_val

        out_idx = n * total + c * H * W + h * W + w
        output_t[out_idx] = al.convert(out_val, al.bf16)


# ============================================================================
# Host model: ModelNew
# ============================================================================
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._num_groups = num_groups

    def forward(self, x):
        N = x.shape[0]
        OC = self._out_channels
        IH = x.shape[2]
        IW = x.shape[3]
        NUM_GROUPS = self._num_groups

        OH = (IH - 1) * self._stride - 2 * self._padding + self._kernel_size
        OW = (IW - 1) * self._stride - 2 * self._padding + self._kernel_size
        OH_pool = OH // 2
        OW_pool = OW // 2

        x = x.contiguous()

        # Step 1: ConvTranspose2d — use PyTorch for bit-exact match
        conv_out = self.conv_transpose(x)

        # Step 2: BatchNorm (eval) + Tanh — AveLang kernels
        bn_weight = self.batch_norm.weight.data.contiguous()
        bn_bias = self.batch_norm.bias.data.contiguous()
        bn_running_mean = self.batch_norm.running_mean.data.contiguous()
        bn_running_var = self.batch_norm.running_var.data.contiguous()

        bn_out = torch.empty(N, OC, OH, OW, dtype=torch.bfloat16, device=x.device)
        block = (256, 1, 1)
        grid_bn = (N, (OC * OH * OW + 255) // 256, 1)
        batch_norm_tanh_kernel[lambda: (grid_bn, block)](
            conv_out.data_ptr(), bn_out.data_ptr(),
            bn_running_mean.data_ptr(), bn_running_var.data_ptr(),
            bn_weight.data_ptr(), bn_bias.data_ptr(),
            N, OC, OH, OW,
        )

        # Step 3: MaxPool2d — AveLang kernel
        pool_out = torch.empty(N, OC, OH_pool, OW_pool, dtype=torch.bfloat16, device=x.device)
        grid_pool = (N, (OC * OH_pool * OW_pool + 255) // 256, 1)
        max_pool2d_kernel[lambda: (grid_pool, block)](
            bn_out.data_ptr(), pool_out.data_ptr(),
            N, OC, OH, OW, OH_pool, OW_pool,
        )

        # Step 4: GroupNorm — two-pass AveLang kernels
        gn_weight = self.group_norm.weight.data.contiguous()
        gn_bias = self.group_norm.bias.data.contiguous()
        gn_sum = torch.empty(N * NUM_GROUPS, dtype=torch.float32, device=x.device)
        gn_var = torch.empty(N * NUM_GROUPS, dtype=torch.float32, device=x.device)

        grid_gn_reduce = (N * NUM_GROUPS, 1, 1)
        group_norm_sum_kernel[lambda: (grid_gn_reduce, block)](
            pool_out.data_ptr(), gn_sum.data_ptr(),
            N, OC, OH_pool, OW_pool, NUM_GROUPS,
        )
        group_norm_var_kernel[lambda: (grid_gn_reduce, block)](
            pool_out.data_ptr(), gn_sum.data_ptr(), gn_var.data_ptr(),
            N, OC, OH_pool, OW_pool, NUM_GROUPS,
        )

        final_out = torch.empty(N, OC, OH_pool, OW_pool, dtype=torch.bfloat16, device=x.device)
        grid_gn_apply = (N, (OC * OH_pool * OW_pool + 255) // 256, 1)
        group_norm_apply_kernel[lambda: (grid_gn_apply, block)](
            pool_out.data_ptr(), final_out.data_ptr(),
            gn_sum.data_ptr(), gn_var.data_ptr(),
            gn_weight.data_ptr(), gn_bias.data_ptr(),
            N, OC, OH_pool, OW_pool, NUM_GROUPS,
        )

        return final_out
