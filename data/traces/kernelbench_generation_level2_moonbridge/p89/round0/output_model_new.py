import torch
import torch.nn as nn
import avelang
import avelang.language as al

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0

BLOCK_SIZE = 256


@avelang.jit
def fused_softmax_sub_swish_max_kernel(
    x_ptr: al.Pointer(al.bf16),
    subtract_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    ch_stride = D * H * W
    spatial_stride = C * ch_stride
    layout_in = al.make_layout(
        (B, C, D, H, W),
        (spatial_stride, ch_stride, H * W, W, 1),
    )
    x = al.make_tensor(x_ptr, al.bf16, layout_in)

    layout_sub = al.make_layout((C,), (1,))
    sub = al.make_tensor(subtract_ptr, al.bf16, layout_sub)

    layout_out = al.make_layout((B, D, H, W), (D * H * W, H * W, W, 1))
    out = al.make_tensor(out_ptr, al.bf16, layout_out)

    total_out = B * D * H * W
    idx = bid * BLOCK_SIZE + tid

    if idx < total_out:
        hw = H * W
        dhw = D * hw
        b = idx // dhw
        rem_dhw = idx - b * dhw
        d = rem_dhw // hw
        rem_hw = rem_dhw - d * hw
        h = rem_hw // W
        w = rem_hw - h * W

        zero_f32 = al.convert(0.0, al.f32)
        one_f32 = al.convert(1.0, al.f32)
        neg_inf = al.convert(-1e30, al.f32)

        # Pass 1: find max value across channels for softmax stability
        max_val = neg_inf
        for c in al.range(C):
            val = al.convert(x[b, c, d, h, w], al.f32)
            if val > max_val:
                max_val = val

        # Pass 2: compute sum of exp(val - max_val)
        exp_sum = zero_f32
        for c in al.range(C):
            val = al.convert(x[b, c, d, h, w], al.f32)
            exp_val = al.exp(val - max_val)
            exp_sum = exp_sum + exp_val

        # Pass 3: compute softmax, subtract, swish, and track max
        final_max = neg_inf
        for c in al.range(C):
            val = al.convert(x[b, c, d, h, w], al.f32)
            exp_val = al.exp(val - max_val)
            softmax_val = exp_val / exp_sum

            sub_val = al.convert(sub[c], al.f32)
            shifted = softmax_val - sub_val

            # Swish: x * sigmoid(x) = x / (1 + exp(-x))
            neg_shifted = zero_f32 - shifted
            neg_exp = al.exp(neg_shifted)
            sigmoid_val = one_f32 / (one_f32 + neg_exp)
            swish_val = shifted * sigmoid_val

            if swish_val > final_max:
                final_max = swish_val

        out[b, d, h, w] = al.convert(final_max, al.bf16)


def _prepare_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_fused_softmax_sub_swish_max(
    x: torch.Tensor,
    subtract: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_contiguous(x)
    sub_bf16 = _prepare_bf16_contiguous(subtract)

    B, C, D, H, W = x_bf16.shape
    assert sub_bf16.shape == (C,), f"Expected subtract shape ({C},), got {sub_bf16.shape}"

    total_out = B * D * H * W
    num_blocks = (total_out + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty((B, D, H, W), device=x_bf16.device, dtype=torch.bfloat16)

    grid = (num_blocks, 1, 1)
    fused_softmax_sub_swish_max_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x_bf16, sub_bf16, out, B, C, D, H, W,
    )
    return out


class ModelNew(nn.Module):
    """
    A model that performs a sequence of operations:
        - ConvTranspose3d (PyTorch)
        - MaxPool3d (PyTorch)
        - Softmax + Subtract + Swish + Max (fused AveLang kernel)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.max_pool = nn.MaxPool3d(
            kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding,
        )
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        result_bf16 = avelang_fused_softmax_sub_swish_max(x, self.subtract)
        return result_bf16.to(dtype=x.dtype)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding,
            output_padding, pool_kernel_size, pool_stride, pool_padding]
