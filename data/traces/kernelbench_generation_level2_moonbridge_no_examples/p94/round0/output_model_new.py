import torch
import torch.nn as nn
import avelang
import avelang.language as al

THREADS = 256


# ══════════════════════════════════════════════════════════════════════════════
# Kernel 1 ─ element-wise Hardtanh + Mish
# ══════════════════════════════════════════════════════════════════════════════


@avelang.jit
def hardtanh_mish_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
):
    tid = al.block_id(0) * al.block_dim(0) + al.thread_id(0)

    if tid < numel:
        x_layout = al.make_layout((numel,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, x_layout)
        out_layout = al.make_layout((numel,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, out_layout)

        val = al.convert(x[tid], al.f32)

        neg_one = al.convert(-1.0, al.f32)
        pos_one = al.convert(1.0, al.f32)
        if val < neg_one:
            val = neg_one
        if val > pos_one:
            val = pos_one

        one = al.convert(1.0, al.f32)
        sp = al.log(one + al.exp(val))
        mish_val = val * al.tanh(sp)

        out[tid] = al.convert(mish_val, al.bf16)


# ══════════════════════════════════════════════════════════════════════════════
# Kernel 2 ─ GroupNorm (training mode, per-sample per-group)
# ══════════════════════════════════════════════════════════════════════════════


@avelang.jit
def groupnorm_kernel(
    x_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    sample: al.i32,
    batch_size: al.i32,
    num_channels: al.i32,
    num_groups: al.i32,
):
    group = al.block_id(0)
    tid = al.thread_id(0)
    channels_per_group = num_channels // num_groups

    x_layout = al.make_layout((batch_size, num_channels), (num_channels, 1))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)
    w_layout = al.make_layout((num_channels,), (1,))
    weight = al.make_tensor(weight_ptr, al.bf16, w_layout)
    bias = al.make_tensor(bias_ptr, al.bf16, w_layout)
    out_layout = al.make_layout((batch_size, num_channels), (num_channels, 1))
    out = al.make_tensor(out_ptr, al.bf16, out_layout)

    ch = group * channels_per_group + tid
    val = al.convert(x[sample, ch], al.f32)

    sh_mem = al.make_shared((32,), al.f64)

    # mean
    sh_mem[tid] = al.convert(0.0, al.f64)
    al.syncthreads()
    sh_mem[tid] = al.convert(val, al.f64)
    al.syncthreads()
    if tid < 16:
        sh_mem[tid] = sh_mem[tid] + sh_mem[tid + 16]
    al.syncthreads()
    if tid < 8:
        sh_mem[tid] = sh_mem[tid] + sh_mem[tid + 8]
    al.syncthreads()
    if tid < 4:
        sh_mem[tid] = sh_mem[tid] + sh_mem[tid + 4]
    al.syncthreads()
    if tid < 2:
        sh_mem[tid] = sh_mem[tid] + sh_mem[tid + 2]
    al.syncthreads()
    if tid < 1:
        sh_mem[tid] = sh_mem[tid] + sh_mem[tid + 1]
    al.syncthreads()

    mean = al.convert(
        sh_mem[0] / al.convert(channels_per_group, al.f64), al.f32
    )

    # variance
    val_f64 = al.convert(val, al.f64)
    mean_f64 = al.convert(mean, al.f64)
    diff = val_f64 - mean_f64
    sh_mem[tid] = al.convert(0.0, al.f64)
    al.syncthreads()
    sh_mem[tid] = diff * diff
    al.syncthreads()
    if tid < 16:
        sh_mem[tid] = sh_mem[tid] + sh_mem[tid + 16]
    al.syncthreads()
    if tid < 8:
        sh_mem[tid] = sh_mem[tid] + sh_mem[tid + 8]
    al.syncthreads()
    if tid < 4:
        sh_mem[tid] = sh_mem[tid] + sh_mem[tid + 4]
    al.syncthreads()
    if tid < 2:
        sh_mem[tid] = sh_mem[tid] + sh_mem[tid + 2]
    al.syncthreads()
    if tid < 1:
        sh_mem[tid] = sh_mem[tid] + sh_mem[tid + 1]
    al.syncthreads()

    var = al.convert(
        sh_mem[0] / al.convert(channels_per_group, al.f64), al.f32
    )
    denom = var + al.convert(1e-5, al.f32)
    inv_std = al.convert(1.0, al.f32) / al.sqrt(denom)

    # normalize + affine
    norm_val = (val - mean) * inv_std
    gamma = al.convert(weight[ch], al.f32)
    beta = al.convert(bias[ch], al.f32)
    out[sample, ch] = al.convert(norm_val * gamma + beta, al.bf16)


# ══════════════════════════════════════════════════════════════════════════════
# Host wrappers
# ══════════════════════════════════════════════════════════════════════════════


def _launch_hardtanh_mish(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    numel = x.numel()
    out = torch.empty_like(x)

    grid = ((numel + THREADS - 1) // THREADS, 1, 1)
    block = (THREADS, 1, 1)

    hardtanh_mish_kernel[lambda: (grid, block)](x, out, numel)
    return out


def _launch_groupnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, num_channels = x.shape
    channels_per_group = num_channels // num_groups

    out = torch.empty_like(x)

    grid = (num_groups, 1, 1)
    block = (channels_per_group, 1, 1)

    for s in range(batch_size):
        groupnorm_kernel[lambda: (grid, block)](
            x, weight, bias, out,
            s, batch_size, num_channels, num_groups,
        )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ModelNew
# ══════════════════════════════════════════════════════════════════════════════


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(
            num_groups=num_groups, num_channels=out_features
        )
        self._num_groups = num_groups

    def forward(self, x):
        x_bf16 = x.to(torch.bfloat16)
        w_bf16 = self.gemm.weight.data.to(torch.bfloat16)
        gn_w = self.groupnorm.weight.data.to(torch.bfloat16)
        gn_b = self.groupnorm.bias.data.to(torch.bfloat16)

        # Match reference: gemm(x) + bias (same rounding order)
        x1 = torch.nn.functional.linear(x_bf16, w_bf16, self.gemm.bias.data.to(torch.bfloat16))
        x1 = x1 + self.bias.data.to(torch.bfloat16)
        # AveLang hardtanh + mish
        x2 = _launch_hardtanh_mish(x1)
        # AveLang GroupNorm
        x3 = _launch_groupnorm(x2, gn_w, gn_b, self._num_groups)

        return x3
