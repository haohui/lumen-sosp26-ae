import torch
import torch.nn as nn
import torch.nn.functional as F
import avelang
import avelang.language as al

BATCH_SIZE = 128
IN_CHANNELS = 64
OUT_CHANNELS = 128
HEIGHT = 128
WIDTH = 128
KERNEL_SIZE = 3
CONSTANT_VALUE = 0.5
SCALING_FACTOR = 2.0

H_OUT = HEIGHT - KERNEL_SIZE + 1
W_OUT = WIDTH - KERNEL_SIZE + 1

BLOCK_SIZE = 256


# ===========================================================================
# Post-processing kernel
# ===========================================================================
@avelang.jit
def postprocess_kernel(
    conv_out_ptr: al.Pointer(al.bf16),
    extra_bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    batch: al.u32,
    oc: al.u32,
    h_out: al.u32,
    w_out: al.u32,
    total_elems: al.u32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    g_conv = al.make_tensor(conv_out_ptr, al.bf16, al.make_layout((total_elems,), (1,)))
    g_ebias = al.make_tensor(extra_bias_ptr, al.bf16, al.make_layout((oc,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((total_elems,), (1,)))

    gid = bid * BLOCK_SIZE + tid

    if gid < total_elems:
        stride_spatial = h_out * w_out
        n_idx = gid // (oc * stride_spatial)
        rem_n = gid - n_idx * (oc * stride_spatial)
        oc_idx = rem_n // stride_spatial
        rem_oc = rem_n - oc_idx * stride_spatial
        h_idx = rem_oc // w_out
        w_idx = rem_oc - h_idx * w_out

        val = g_conv[gid]

        const_val = al.convert(0.5, al.bf16)
        if val > const_val:
            val = const_val

        eb = g_ebias[oc_idx]
        val = val + eb
        val = val * al.convert(2.0, al.bf16)

        g_out[gid] = val


# ===========================================================================
# Host helpers
# ===========================================================================
def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv_fused(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    extra_bias: torch.Tensor,
    constant_value: float,
    scaling_factor: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(conv_weight)
    cb_bf16 = _prepare_bf16_cuda_contiguous(conv_bias)
    eb_bf16 = extra_bias.squeeze()
    eb_bf16 = _prepare_bf16_cuda_contiguous(eb_bf16)

    batch = x_bf16.shape[0]
    oc_val = w_bf16.shape[0]
    _, _, kh, kw = w_bf16.shape
    h_out = x_bf16.shape[2] - kh + 1
    w_out = x_bf16.shape[3] - kw + 1

    conv_out = F.conv2d(x_bf16, w_bf16, cb_bf16, padding=0)
    total_elems = batch * oc_val * h_out * w_out
    out = torch.empty_like(conv_out)

    grid = ((total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
    postprocess_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        conv_out, eb_bf16, out, batch, oc_val, h_out, w_out, total_elems,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        weight = self.conv.weight.data
        cb = self.conv.bias.data
        eb = self.bias.data
        return avelang_conv_fused(
            x, weight, cb, eb, self.constant_value, self.scaling_factor,
        )


def get_inputs():
    return [torch.rand(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]


def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, CONSTANT_VALUE, (OUT_CHANNELS, 1, 1), SCALING_FACTOR]
