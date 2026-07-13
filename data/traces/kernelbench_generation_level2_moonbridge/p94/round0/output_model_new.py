import torch
import torch.nn as nn
import avelang
import avelang.language as al

BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
BIAS_SHAPE = (OUT_FEATURES,)
NUM_GROUPS = 256

EW_THREADS = 256


@avelang.jit
def bias_hardtanh_mish_kernel(
    x_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    total_elems: al.u32,
    n: al.u32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    gid = bid * EW_THREADS + tid

    if gid < total_elems:
        row = gid // n
        col = gid - row * n

        layout_2d = al.make_layout((total_elems // n, n), (n, 1))
        x = al.make_tensor(x_ptr, al.bf16, layout_2d)
        bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
        out = al.make_tensor(out_ptr, al.bf16, layout_2d)

        # BF16 addition first (matches reference: x + bias in BF16)
        val_bf16 = x[row, col]
        b_bf16 = bias[col]
        added_bf16 = val_bf16 + b_bf16

        # Then convert to FP32 for activation functions
        result = al.convert(added_bf16, al.f32)

        neg_one = al.convert(-1.0, al.f32)
        pos_one = al.convert(1.0, al.f32)
        if result < neg_one:
            result = neg_one
        if result > pos_one:
            result = pos_one

        one_f32 = al.convert(1.0, al.f32)
        sp = al.log(one_f32 + al.exp(result))
        result = result * al.tanh(sp)

        out[row, col] = al.convert(result, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_bias_hardtanh_mish(
    x: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required.")
    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    b_bf16 = _prepare_bf16_cuda_contiguous(bias)
    total = x_bf16.numel()
    n = x_bf16.shape[1]
    out = torch.empty_like(x_bf16)
    num_blocks = (total + EW_THREADS - 1) // EW_THREADS
    bias_hardtanh_mish_kernel[lambda: ((num_blocks, 1, 1), (EW_THREADS, 1, 1))](
        x_bf16, b_bf16, out, total, n
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def forward(self, x):
        gn_weight = self.groupnorm.weight.data
        gn_bias = self.groupnorm.bias.data

        # Step 1: Linear = addmm(b_linear, x, w.T) -> matches reference nn.Linear
        linear_out = torch.addmm(
            self.gemm.bias.data, x, self.gemm.weight.data.T,
        ).to(dtype=torch.bfloat16)

        # Step 2: + extra_bias (BF16) + Hardtanh + Mish (AveLang kernel)
        act_out = avelang_bias_hardtanh_mish(linear_out, self.bias.data)

        # Step 3: GroupNorm (PyTorch, matches reference)
        result = torch.nn.functional.group_norm(
            act_out, num_groups=self.num_groups, weight=gn_weight, bias=gn_bias
        )
        return result.to(x.dtype)


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_FEATURES)]


def get_init_inputs():
    return [IN_FEATURES, OUT_FEATURES, BIAS_SHAPE, NUM_GROUPS]
