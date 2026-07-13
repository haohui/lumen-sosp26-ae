import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_L: al.constexpr = 128
THREADS: al.constexpr = 256
WARP_SIZE: al.constexpr = 64
NUM_WARPS: al.constexpr = 4
WEIGHT_SIZE: al.constexpr = 6144


@avelang.jit
def _load_weight_to_shm(
    shm_w: al.Tensor((WEIGHT_SIZE,), al.bf16),
    w_memref: al.Tensor((WEIGHT_SIZE,), al.bf16),
    tid: al.i32,
):
    for i in al.range(tid, WEIGHT_SIZE, THREADS):
        shm_w[i] = w_memref[i]


@avelang.jit
def conv_transpose1d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    in_channels: al.i32,
    out_channels: al.i32,
    length: al.i32,
    l_out: al.i32,
    kernel_size: al.i32,
    stride: al.i32,
    padding: al.i32,
    dilation: al.i32,
    has_bias: al.i32,
):
    tid = al.thread_id(0)
    l_out_block = al.block_id(0)
    batch_idx = al.block_id(1)

    # Stage weight into shared memory (linearised)
    w_memref = al.make_tensor(
        w_ptr, al.bf16,
        al.make_layout((WEIGHT_SIZE,), (1,))
    )
    shm_w = al.make_shared((WEIGHT_SIZE,), al.bf16)
    _load_weight_to_shm(shm_w, w_memref, tid)
    al.syncthreads()

    l_start = l_out_block * TILE_L
    l_end = l_start + TILE_L
    if l_end > l_out:
        l_end = l_out

    # Input: (batch_size, in_channels, length)
    x = al.make_tensor(
        x_ptr, al.bf16,
        al.make_layout((batch_size, in_channels, length), (in_channels * length, length, 1))
    )
    # Output: (batch_size, out_channels, l_out)
    out = al.make_tensor(
        out_ptr, al.bf16,
        al.make_layout((batch_size, out_channels, l_out), (out_channels * l_out, l_out, 1))
    )

    bias = al.make_tensor(
        bias_ptr, al.bf16,
        al.make_layout((out_channels,), (1,))
    )

    warp_id = tid // WARP_SIZE
    lane = tid - warp_id * WARP_SIZE
    oc = lane

    zero_i32 = al.convert(0, al.i32)
    one_i32 = al.convert(1, al.i32)
    two_i32 = al.convert(2, al.i32)

    # Pre-load bias for this thread
    bias_val = al.convert(0.0, al.f32)
    if has_bias != zero_i32:
        if oc < out_channels:
            bias_val = al.convert(bias[oc], al.f32)

    # Only process odd output positions (with stride=2, padding=1: odd l_out → valid)
    first = l_start
    mod = l_start - (l_start // two_i32) * two_i32
    if mod == zero_i32:
        first = l_start + one_i32

    step = stride * NUM_WARPS
    for l_my in al.range(first + warp_id * stride, l_end, step):
        if oc < out_channels:
            acc = al.convert(0.0, al.f32)
            for ic in al.range(in_channels):
                w_base = ic * out_channels * kernel_size
                for k in al.range(kernel_size):
                    l_in_idx = l_my + padding - k * dilation
                    if l_in_idx >= zero_i32:
                        l_in = l_in_idx // stride
                        if l_in * stride == l_in_idx:
                            if l_in < length:
                                x_val = al.convert(x[batch_idx, ic, l_in], al.f32)
                                w_idx = w_base + oc * kernel_size + k
                                w_val = al.convert(shm_w[w_idx], al.f32)
                                acc = acc + x_val * w_val

            acc = acc + bias_val
            out[batch_idx, oc, l_my] = al.convert(acc, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_transpose1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)

    batch_size, in_channels, length = x_bf16.shape
    w_in_c, out_channels, kernel_size_w = w_bf16.shape
    if w_in_c != in_channels:
        raise ValueError(f"Weight in_channels mismatch: {in_channels} vs {w_in_c}")

    l_out = (length - 1) * stride - 2 * padding + dilation * (kernel_size_w - 1) + 1

    has_bias = 1 if bias is not None else 0
    if bias is not None:
        bias_bf16 = _prepare_bf16_cuda_contiguous(bias)
    else:
        bias_bf16 = torch.empty(1, device=x.device, dtype=torch.bfloat16)

    out = torch.zeros((batch_size, out_channels, l_out), device=x.device, dtype=torch.bfloat16)

    num_l_out_blocks = (l_out + TILE_L - 1) // TILE_L
    grid = (num_l_out_blocks, batch_size, 1)

    conv_transpose1d_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, bias_bf16, out,
        batch_size, in_channels, out_channels, length, l_out,
        kernel_size_w, stride, padding, dilation, has_bias,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_transpose1d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation,
        )
