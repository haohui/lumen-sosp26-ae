import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
OC_BLOCK: al.constexpr = 128
H_OUT = 514
W_OUT = 514


@avelang.jit
def spatial_reduce_kernel(
    x_ptr: al.Pointer(al.bf16),
    sum_out_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    in_channels: al.i32,
    height: al.i32,
    width: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    ic_idx = bid % in_channels
    batch_idx = bid // in_channels

    hw_total = height * width
    ch_stride = hw_total
    batch_stride = in_channels * ch_stride
    base = batch_idx * batch_stride + ic_idx * ch_stride

    layout_flat = al.make_layout((batch_size * in_channels * hw_total,), (1,))
    x_flat = al.make_tensor(x_ptr, al.bf16, layout_flat)

    smem = al.make_shared((BLOCK_SIZE,), al.f32)

    local_sum = al.convert(0.0, al.f32)
    for i in al.range(tid, hw_total, BLOCK_SIZE):
        val = al.convert(x_flat[base + i], al.f32)
        local_sum = local_sum + val

    smem[tid] = local_sum
    al.syncthreads()

    if tid < 128:
        smem[tid] = smem[tid] + smem[tid + 128]
    al.syncthreads()
    if tid < 64:
        smem[tid] = smem[tid] + smem[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]

    if tid == 0:
        layout_out = al.make_layout((batch_size, in_channels), (in_channels, 1))
        out = al.make_tensor(sum_out_ptr, al.f32, layout_out)
        out[batch_idx, ic_idx] = smem[0]


@avelang.jit
def fused_matmul_logsumexp_kernel(
    input_sum_ptr: al.Pointer(al.f32),
    weight_sum_T_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    batch_size: al.i32,
    IC: al.constexpr,
    OC: al.constexpr,
    scale: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    layout_sums = al.make_layout((batch_size, IC), (IC, 1))
    sums = al.make_tensor(input_sum_ptr, al.f32, layout_sums)

    layout_weight = al.make_layout((OC, IC), (IC, 1))
    weight = al.make_tensor(weight_sum_T_ptr, al.bf16, layout_weight)

    layout_bias = al.make_layout((OC,), (1,))
    bias = al.make_tensor(bias_ptr, al.f32, layout_bias)

    acc = al.convert(0.0, al.f32)
    for ic in al.range(IC):
        w_val = al.convert(weight[tid, ic], al.f32)
        acc = acc + sums[bid, ic] * w_val

    acc = acc * scale + bias[tid]

    smem = al.make_shared((OC,), al.f32)
    smem[tid] = acc
    al.syncthreads()

    # Find max across channels
    if tid < 64:
        a = smem[tid]
        b = smem[tid + 64]
        smem[tid] = a if a > b else b
    al.syncthreads()
    if tid < 32:
        a = smem[tid]
        b = smem[tid + 32]
        smem[tid] = a if a > b else b
    al.syncthreads()
    if tid < 16:
        a = smem[tid]
        b = smem[tid + 16]
        smem[tid] = a if a > b else b
    al.syncthreads()
    if tid < 8:
        a = smem[tid]
        b = smem[tid + 8]
        smem[tid] = a if a > b else b
    al.syncthreads()
    if tid < 4:
        a = smem[tid]
        b = smem[tid + 4]
        smem[tid] = a if a > b else b
    al.syncthreads()
    if tid < 2:
        a = smem[tid]
        b = smem[tid + 2]
        smem[tid] = a if a > b else b
    al.syncthreads()
    if tid < 1:
        a = smem[tid]
        b = smem[tid + 1]
        smem[tid] = a if a > b else b

    max_val = smem[0]

    # Compute exp(x - max) and sum
    shifted = al.exp(acc - max_val)
    smem[tid] = shifted
    al.syncthreads()

    if tid < 64:
        smem[tid] = smem[tid] + smem[tid + 64]
    al.syncthreads()
    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
    al.syncthreads()
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]

    sum_exp = smem[0]
    result = al.log(sum_exp) + max_val
    result = result * al.convert(10.0, al.f32)

    if tid == 0:
        layout_out = al.make_layout((batch_size, 1), (1, 1))
        out = al.make_tensor(out_ptr, al.f32, layout_out)
        out[bid, 0] = result


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_forward(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    sep_bias: torch.Tensor,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device."
    input_dtype = x.dtype

    batch_size = x.shape[0]
    in_channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    out_channels = conv_weight.shape[1]

    # Precompute weight sum over kernel dims: (IC, OC, KH, KW) -> (IC, OC)
    weight_sum = conv_weight.sum(dim=(2, 3))  # (IC, OC)
    # Transpose: (OC, IC) for matmul
    weight_sum_T = weight_sum.t().contiguous().to(dtype=torch.bfloat16)

    # Combined bias: conv_bias (OC,) + sep_bias (OC, 1, 1)
    total_bias = (conv_bias + sep_bias.squeeze()).contiguous().to(dtype=torch.float32)

    x_bf16 = _to_bf16_contiguous(x)

    # Intermediate: spatial sums (B, IC) in FP32
    input_sums = torch.empty((batch_size, in_channels), dtype=torch.float32, device=x.device)

    # Output: (B, 1) in FP32
    output = torch.empty((batch_size, 1), dtype=torch.float32, device=x.device)

    # Scale factor: 1 / (H_out * W_out) = 1 / (514 * 514)
    scale = 1.0 / (H_OUT * W_OUT)

    num_blocks_spatial = batch_size * in_channels
    spatial_reduce_kernel[lambda: ((num_blocks_spatial, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_bf16, input_sums, batch_size, in_channels, height, width
    )

    fused_matmul_logsumexp_kernel[lambda: ((batch_size, 1, 1), (OC_BLOCK, 1, 1))](
        input_sums, weight_sum_T, total_bias, output, batch_size,
        in_channels, out_channels, scale
    )

    # Convert back to input dtype if needed
    if input_dtype != torch.float32:
        output = output.to(dtype=input_dtype)

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        conv_weight = self.conv_transpose.weight.data
        conv_bias = self.conv_transpose.bias.data
        sep_bias = self.bias.data
        return avelang_forward(x, conv_weight, conv_bias, sep_bias)


batch_size = 16
in_channels = 64
out_channels = 128
height = 512
width = 512
kernel_size = 3
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
