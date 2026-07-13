import torch
import torch.nn as nn
import avelang
import avelang.language as al

@avelang.jit
def swish_bias_kernel(
    x_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
):
    layout_x = al.make_layout((M, N), (N, 1))
    layout_bias = al.make_layout((N,), (1,))
    layout_y = al.make_layout((M, N), (N, 1))

    x = al.make_tensor(x_ptr, al.bf16, layout_x)
    b = al.make_tensor(bias_ptr, al.bf16, layout_bias)
    y = al.make_tensor(y_ptr, al.bf16, layout_y)

    col = al.block_id(0) * 256 + al.thread_id(0)
    row = al.block_id(1)

    if row < M and col < N:
        val = al.convert(x[row, col], al.f32)
        neg_val = al.convert(0.0, al.f32) - val
        exp_neg = al.exp(neg_val)
        one = al.convert(1.0, al.f32)
        sigmoid = one / (one + exp_neg)
        swish = val * sigmoid
        out_val = swish + al.convert(b[col], al.f32)
        y[row, col] = al.convert(out_val, al.bf16)


@avelang.jit
def groupnorm_kernel(
    x_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    y_ptr: al.Pointer(al.bf16),
    M: al.i32,
    N: al.i32,
    num_groups: al.i32,
    channels_per_group: al.i32,
):
    layout_x = al.make_layout((M, N), (N, 1))
    layout_g = al.make_layout((N,), (1,))
    layout_b = al.make_layout((N,), (1,))
    layout_y = al.make_layout((M, N), (N, 1))

    x = al.make_tensor(x_ptr, al.bf16, layout_x)
    gamma = al.make_tensor(gamma_ptr, al.bf16, layout_g)
    beta_t = al.make_tensor(beta_ptr, al.bf16, layout_b)
    y = al.make_tensor(y_ptr, al.bf16, layout_y)

    row = al.block_id(0)
    tid = al.thread_id(0)

    sdata = al.make_shared((64,), al.f32)
    count_f = al.convert(channels_per_group, al.f32)
    eps = al.convert(1e-5, al.f32)

    for g in al.range(0, num_groups, 1):
        channel_start = g * channels_per_group
        channel_idx = channel_start + tid

        val = al.convert(x[row, channel_idx], al.f32)

        sdata[tid] = val
        al.syncthreads()
        if tid < 32:
            sdata[tid] = sdata[tid] + sdata[tid + 32]
        al.syncthreads()
        if tid < 16:
            sdata[tid] = sdata[tid] + sdata[tid + 16]
        al.syncthreads()
        if tid < 8:
            sdata[tid] = sdata[tid] + sdata[tid + 8]
        al.syncthreads()
        if tid < 4:
            sdata[tid] = sdata[tid] + sdata[tid + 4]
        al.syncthreads()
        if tid < 2:
            sdata[tid] = sdata[tid] + sdata[tid + 2]
        al.syncthreads()
        if tid < 1:
            sdata[tid] = sdata[tid] + sdata[tid + 1]
        al.syncthreads()

        mean = sdata[0] / count_f

        diff = val - mean
        sdata[tid] = diff * diff
        al.syncthreads()
        if tid < 32:
            sdata[tid] = sdata[tid] + sdata[tid + 32]
        al.syncthreads()
        if tid < 16:
            sdata[tid] = sdata[tid] + sdata[tid + 16]
        al.syncthreads()
        if tid < 8:
            sdata[tid] = sdata[tid] + sdata[tid + 8]
        al.syncthreads()
        if tid < 4:
            sdata[tid] = sdata[tid] + sdata[tid + 4]
        al.syncthreads()
        if tid < 2:
            sdata[tid] = sdata[tid] + sdata[tid + 2]
        al.syncthreads()
        if tid < 1:
            sdata[tid] = sdata[tid] + sdata[tid + 1]
        al.syncthreads()

        var = sdata[0] / count_f

        inv_std = al.convert(1.0, al.f32) / al.sqrt(var + eps)
        norm_val = diff * inv_std
        gamma_val = al.convert(gamma[channel_idx], al.f32)
        beta_val = al.convert(beta_t[channel_idx], al.f32)
        result = norm_val * gamma_val + beta_val

        y[row, channel_idx] = al.convert(result, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        # x is already BF16 from the eval's model.to(dtype=precision)
        x = self.matmul(x)
        x = torch.sigmoid(x) * x
        x = x + self.bias

        # GroupNorm in AveLang
        M, N = x.shape
        num_groups = self.group_norm.num_groups
        channels_per_group = N // num_groups
        out = torch.empty_like(x)

        groupnorm_kernel[lambda: ((M, 1, 1), (64, 1, 1))](
            x, self.group_norm.weight, self.group_norm.bias,
            out, M, N, num_groups, channels_per_group,
        )
        return out
