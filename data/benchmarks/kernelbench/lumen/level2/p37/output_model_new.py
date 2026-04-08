import torch
import torch.nn as nn
import substrate
import substrate.language as S


BIAS_BLOCK_SIZE = 256
BIAS_NUM_WARPS = 4

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)


@substrate.jit
def add_bias_inplace(
    x_ptr: S.Pointer(S.bf16),
    bias_ptr: S.Pointer(S.bf16),
    total_values: S.u32,
    channel_count: S.u32,
):
    idx = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    if idx >= total_values:
        return

    flat = S.make_tensor(x_ptr, S.bf16, S.make_layout((total_values,), (1,)))
    bias = S.make_tensor(bias_ptr, S.bf16, S.make_layout((channel_count,), (1,)))
    flat[idx] = flat[idx] + bias[idx % channel_count]


def add_bias_(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    total_values = x.numel()
    channel_count = x.shape[-1]
    grid = ((total_values + BIAS_BLOCK_SIZE - 1) // BIAS_BLOCK_SIZE, 1, 1)
    add_bias_inplace[lambda: (grid, (BIAS_BLOCK_SIZE, 1, 1))](
        x,
        bias,
        total_values,
        channel_count,
        num_warps=BIAS_NUM_WARPS,
    )
    return x


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError("CUDA/HIP device is required for Substrate kernels.")

        orig_dtype = x.dtype
        x_bf16 = x if x.dtype == torch.bfloat16 else x.to(dtype=torch.bfloat16)

        out = self.matmul(x_bf16)
        sigmoid_out = torch.sigmoid(out)
        out.mul_(sigmoid_out)

        bias = self.bias
        if bias.dtype != torch.bfloat16 or bias.device != out.device or not bias.is_contiguous():
            bias = bias.to(device=out.device, dtype=torch.bfloat16).contiguous()
        add_bias_(out, bias)

        out = self.group_norm(out)
        if orig_dtype != torch.bfloat16:
            return out.to(orig_dtype)
        return out


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
