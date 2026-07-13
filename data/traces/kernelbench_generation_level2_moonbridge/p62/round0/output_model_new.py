import torch
import torch.nn as nn
import avelang
import avelang.language as al

from avelang_kernels.amdgpu_gemm import gemm_pipeline_transposed_b

# ── GroupNorm constants ────────────────────────────────────────────────
GN_BLOCK_SIZE = 256
GN_CHANNELS_PER_GROUP = 16
GN_GROUPS_PER_ITER = GN_BLOCK_SIZE // GN_CHANNELS_PER_GROUP  # 16


# ═══════════════════════════════════════════════════════════════════════
# Fused kernel: GroupNorm (training mode) + LeakyReLU + element-wise x*2
# ═══════════════════════════════════════════════════════════════════════

@avelang.jit
def groupnorm_leakyrelu_add_kernel(
    x_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    channels: al.i32,
    num_groups: al.i32,
    eps: al.f32,
    negative_slope: al.f32,
):
    """
    Fused kernel:
      1. GroupNorm (training mode — compute mean/var from input)
      2. LeakyReLU
      3. result * 2  (x + x)
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    if bid < batch_size:
        layout_flat = al.make_layout((batch_size * channels,), (1,))
        x = al.make_tensor(x_ptr, al.bf16, layout_flat)
        w = al.make_tensor(weight_ptr, al.bf16, al.make_layout((channels,), (1,)))
        b = al.make_tensor(bias_ptr, al.bf16, al.make_layout((channels,), (1,)))
        out = al.make_tensor(out_ptr, al.bf16, layout_flat)

        channels_per_group = channels // num_groups
        group_in_iter = tid // channels_per_group
        thread_in_group = tid % channels_per_group

        smem_sum = al.make_shared((GN_BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((GN_BLOCK_SIZE,), al.f32)

        num_iters = num_groups // GN_GROUPS_PER_ITER

        for gi in al.range(num_iters):
            base_group = gi * GN_GROUPS_PER_ITER
            global_group = base_group + group_in_iter
            channel = global_group * channels_per_group + thread_in_group
            element_idx = bid * channels + channel

            val = al.convert(x[element_idx], al.f32)

            smem_sum[tid] = val
            smem_sq[tid] = val * val
            al.syncthreads()

            # Shared-memory reduction tree: 16 → 8 → 4 → 2 → 1 (per group)
            if thread_in_group < 8:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
            al.syncthreads()
            if thread_in_group < 4:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
            al.syncthreads()
            if thread_in_group < 2:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
            al.syncthreads()
            if thread_in_group < 1:
                smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
                smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]
            al.syncthreads()

            # Thread 0 of each group computes mean, var, rstd
            if thread_in_group == 0:
                cpg_f32 = al.convert(channels_per_group, al.f32)
                mean_val = smem_sum[tid] / cpg_f32
                var_val = smem_sq[tid] / cpg_f32 - mean_val * mean_val
                rstd_val = al.convert(1.0, al.f32) / al.sqrt(var_val + eps)
                smem_sum[tid] = mean_val
                smem_sq[tid] = rstd_val
            al.syncthreads()

            local_group_start = group_in_iter * channels_per_group
            mean_val = smem_sum[local_group_start]
            rstd_val = smem_sq[local_group_start]

            w_val = al.convert(w[channel], al.f32)
            b_val = al.convert(b[channel], al.f32)

            normalized = (val - mean_val) * rstd_val
            result = normalized * w_val + b_val

            # LeakyReLU
            zero = al.convert(0.0, al.f32)
            two = al.convert(2.0, al.f32)
            if result < zero:
                result = result * negative_slope
            # x + x → x * 2
            result = result * two

            out[element_idx] = al.convert(result, al.bf16)


# ═══════════════════════════════════════════════════════════════════════
# Host wrapper
# ═══════════════════════════════════════════════════════════════════════

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_groupnorm_leakyrelu_add(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
    negative_slope: float,
) -> torch.Tensor:
    """Fused: GroupNorm → LeakyReLU → *2."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    batch_size, channels = x_bf16.shape
    if channels % num_groups != 0:
        raise ValueError(
            f"channels={channels} must be divisible by num_groups={num_groups}"
        )

    out = torch.empty_like(x_bf16)
    eps_f32 = float(eps)
    neg_slope_f32 = float(negative_slope)

    groupnorm_leakyrelu_add_kernel[lambda: ((batch_size, 1, 1), (GN_BLOCK_SIZE, 1, 1))](
        x_bf16, weight_bf16, bias_bf16, out,
        batch_size, channels, num_groups, eps_f32, neg_slope_f32,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════
# ModelNew
# ═══════════════════════════════════════════════════════════════════════

class ModelNew(nn.Module):
    """
    Optimized BF16 model: Linear → GroupNorm → LeakyReLU → x+x
    Uses AveLang DSL kernel for the GroupNorm+LeakyReLU+x*2 fused path.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope

        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)

    def forward(self, x):
        # Step 1: Linear layer (matmul + bias)
        x = self.fc(x)

        # Step 2-4: GroupNorm + LeakyReLU + x*2 fused in one AveLang kernel
        x = avelang_groupnorm_leakyrelu_add(
            x,
            self.gn.weight.data,
            self.gn.bias.data,
            self.num_groups,
            self.eps,
            self.negative_slope,
        )
        return x
