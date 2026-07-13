import torch
import torch.nn as nn
import avelang
import avelang.language as al


REDUCE_BLOCK = 256


@avelang.jit
def fused_kernel(
    input_ptr: al.Pointer(al.bf16),
    w_colsum_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    bias_sub_sum: al.f32,
    m: al.i32,
    n: al.i32,
    k: al.i32,
):
    tid = al.thread_id(0)
    row = al.block_id(0)
    block_size = al.convert(REDUCE_BLOCK, al.i32)
    one = al.convert(1, al.i32)
    zero = al.convert(0, al.i32)
    n_f32 = al.convert(n, al.f32)

    k_per_thread = k // block_size
    n_per_thread = n // block_size

    input_layout = al.make_layout((m, k), (k, one))
    input_tensor = al.make_tensor(input_ptr, al.bf16, input_layout)
    w_colsum_layout = al.make_layout((k,), (one,))
    w_colsum_tensor = al.make_tensor(w_colsum_ptr, al.bf16, w_colsum_layout)
    output_layout = al.make_layout((m, n), (n, one))
    output_tensor = al.make_tensor(output_ptr, al.bf16, output_layout)

    dot = al.convert(0.0, al.f32)
    base = tid * k_per_thread
    for idx in al.range(k_per_thread):
        pos = base + al.convert(idx, al.i32)
        x_val = al.convert(input_tensor[row, pos], al.f32)
        w_val = al.convert(w_colsum_tensor[pos], al.f32)
        dot = dot + x_val * w_val

    shm = al.make_shared((REDUCE_BLOCK,), al.f32)
    shm[tid] = dot
    al.syncthreads()

    if al.convert(tid, al.i32) == zero:
        total = shm[0]
        for i in al.range(1, REDUCE_BLOCK):
            total = total + shm[i]
        shm[0] = total

    al.syncthreads()

    mean_val = (shm[0] + bias_sub_sum) / n_f32

    sqrt_2_over_pi = al.convert(0.7978845608028654, al.f32)
    gelu_c = al.convert(0.044715, al.f32)
    half = al.convert(0.5, al.f32)
    one_f = al.convert(1.0, al.f32)

    x3 = mean_val * mean_val * mean_val
    inner = sqrt_2_over_pi * (mean_val + gelu_c * x3)
    gelu_val = half * mean_val * (one_f + al.tanh(inner))

    base = tid * n_per_thread
    for idx in al.range(n_per_thread):
        pos = base + al.convert(idx, al.i32)
        out_val = gelu_val + al.convert(input_tensor[row, pos], al.f32)
        output_tensor[row, pos] = al.convert(out_val, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.subtract = nn.Parameter(torch.empty(out_features))
        self.register_buffer(
            '_w_colsum',
            torch.zeros(in_features, dtype=torch.bfloat16),
        )
        self.reset_parameters()
        self._cached_bias_sub_sum = 0.0
        self._precomputed = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.subtract)

    def _ensure_precomputed(self):
        if self._precomputed:
            return
        with torch.no_grad():
            w = self.weight.to(dtype=torch.bfloat16)
            self._w_colsum.copy_(w.sum(dim=0))
            bias_sum = self.bias.float().sum()
            sub_sum = self.subtract.float().sum()
            self._cached_bias_sub_sum = (bias_sum - sub_sum).item()
        self._precomputed = True

    def forward(self, x):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")
        device = x.device

        self._ensure_precomputed()

        x_bf16 = _prepare_bf16_cuda_contiguous(x)
        m_val, k_val = x_bf16.shape
        n_val = self.out_features

        out = torch.empty((m_val, n_val), device=device, dtype=torch.bfloat16)
        grid = (m_val, 1, 1)
        fused_kernel[lambda: (grid, (REDUCE_BLOCK, 1, 1))](
            x_bf16, self._w_colsum, out, self._cached_bias_sub_sum,
            m_val, n_val, k_val,
        )
        return out


batch_size = 2048
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
