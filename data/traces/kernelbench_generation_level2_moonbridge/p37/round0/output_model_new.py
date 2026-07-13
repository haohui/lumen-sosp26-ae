import torch
import torch.nn as nn
import avelang
import avelang.language as al

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

# GroupNorm constant (kept for reference, not used in GN kernel since we use PyTorch GN)
GN_GROUP_SIZE = out_features // num_groups


@avelang.jit
def swish_bias_kernel(
    x_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_elems: al.u32,
    n: al.u32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    BLOCK = al.convert(256, al.u32)
    idx = bid * BLOCK + tid

    if idx < total_elems:
        g_in = al.make_tensor(x_ptr, al.bf16, al.make_layout((total_elems,), (1,)))
        g_bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
        g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((total_elems,), (1,)))

        col = idx - (idx // n) * n
        x_val = al.convert(g_in[idx], al.f32)
        bias_val = al.convert(g_bias[col], al.f32)
        one = al.convert(1.0, al.f32)
        zero = al.convert(0.0, al.f32)
        neg_val = zero - x_val
        exp_neg = al.exp(neg_val)
        sigmoid_val = one / (one + exp_neg)
        swish_val = x_val * sigmoid_val
        result = swish_val + bias_val
        g_out[idx] = al.convert(result, al.bf16)


def avelang_swish_bias(
    x: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    if bias.dtype != torch.bfloat16:
        bias = bias.to(torch.bfloat16)

    m, n = x.shape
    total = m * n
    out = torch.empty_like(x)
    BLOCK = 256
    grid_elems = (total + BLOCK - 1) // BLOCK
    swish_bias_kernel[lambda: ((grid_elems, 1, 1), (BLOCK, 1, 1))](
        x.contiguous(), bias.contiguous(), out, total, n
    )
    return out


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
        # Step 1: GEMM via PyTorch Linear (BF16)
        x = self.matmul(x)

        # Step 2: Swish via AveLang
        x = avelang_swish_bias(x, self.bias.data)

        # Step 3: Bias add and GroupNorm via PyTorch
        x = self.group_norm(x)

        return x


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
