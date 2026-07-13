import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── GroupNorm constants ─────────────────────────────────────────────────
GN_BLOCK_SIZE: al.constexpr = 64
GN_SHARED_SIZE: al.constexpr = 64


# ══════════════════════════════════════════════════════════════════════════
#  GroupNorm kernel  (reads FP32 intermediate, writes BF16 final)
# ══════════════════════════════════════════════════════════════════════════

@avelang.jit
def groupnorm_kernel(
    x_ptr: al.Pointer(al.f32),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    N: al.i32,
    num_groups: al.i32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    batch_idx = bid // num_groups
    group_idx = bid - batch_idx * num_groups

    if batch_idx < batch_size:
        channels_per_group = N // num_groups
        base_col = group_idx * channels_per_group

        smem_sum = al.make_shared((GN_SHARED_SIZE,), al.f32)
        smem_sq = al.make_shared((GN_SHARED_SIZE,), al.f32)

        layout_in = al.make_layout((batch_size * N,), (1,))
        x = al.make_tensor(x_ptr, al.f32, layout_in)

        layout_w = al.make_layout((N,), (1,))
        w = al.make_tensor(weight_ptr, al.bf16, layout_w)
        b = al.make_tensor(bias_ptr, al.bf16, layout_w)

        layout_out = al.make_layout((batch_size * N,), (1,))
        out = al.make_tensor(out_ptr, al.bf16, layout_out)

        idx = batch_idx * N + base_col + tid
        val = x[idx]
        w_val = al.convert(w[base_col + tid], al.f32)
        b_val = al.convert(b[base_col + tid], al.f32)

        smem_sum[tid] = val
        smem_sq[tid] = val * val
        al.syncthreads()

        # Reduction tree: 64 → 32 → 16 → 8 → 4 → 2 → 1
        if tid < 32:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]

        if tid == 0:
            n_f32 = al.convert(channels_per_group, al.f32)
            mean = smem_sum[0] / n_f32
            var = smem_sq[0] / n_f32 - mean * mean
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)
            smem_sum[0] = mean
            smem_sq[0] = rstd

        al.syncthreads()

        mean = smem_sum[0]
        rstd = smem_sq[0]
        normalized = (val - mean) * rstd
        result = normalized * w_val + b_val
        out[idx] = al.convert(result, al.bf16)


# ══════════════════════════════════════════════════════════════════════════
#  Host wrapper
# ══════════════════════════════════════════════════════════════════════════

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_groupnorm(
    x_fp32: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Takes FP32 input, returns BF16 output."""
    if not (x_fp32.is_cuda and x_fp32.dtype == torch.float32 and x_fp32.is_contiguous()):
        x_fp32 = x_fp32.contiguous().cuda().to(dtype=torch.float32)

    w_bf16 = _prepare_bf16_cuda_contiguous(weight)
    b_bf16 = _prepare_bf16_cuda_contiguous(bias)

    batch_size, N = x_fp32.shape

    out = torch.empty((batch_size, N), device=x_fp32.device, dtype=torch.bfloat16)
    grid = (batch_size * num_groups, 1, 1)
    groupnorm_kernel[lambda: (grid, (GN_BLOCK_SIZE, 1, 1))](
        x_fp32, w_bf16, b_bf16, out, batch_size, N, num_groups, eps
    )
    return out


# ══════════════════════════════════════════════════════════════════════════
#  ModelNew  — matmul/swish/bias in PyTorch (matches ref), GN in AveLang
# ══════════════════════════════════════════════════════════════════════════

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        # Stages 1-3: matmul + swish + bias via PyTorch (exact match with reference)
        x = self.matmul(x)
        x = torch.sigmoid(x) * x
        x = x + self.bias

        # Stage 4: GroupNorm via AveLang kernel
        gn_weight = self.group_norm.weight.data
        gn_bias = self.group_norm.bias.data
        result_bf16 = avelang_groupnorm(
            x.float(), gn_weight, gn_bias, self.num_groups, eps=1e-5
        )

        return result_bf16.to(x.dtype)


# ══════════════════════════════════════════════════════════════════════════
#  Preserve public contract
# ══════════════════════════════════════════════════════════════════════════

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
