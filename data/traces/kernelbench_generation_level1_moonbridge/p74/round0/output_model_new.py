import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Problem-specific compile-time constants
IC: al.constexpr = 32
OC: al.constexpr = 64
KS: al.constexpr = 5
DIL: al.constexpr = 3

BLOCK_L: al.constexpr = 256
BLOCK_OC: al.constexpr = 16
OC_GROUPS: al.constexpr = OC // BLOCK_OC  # 4

IN_TILE_L: al.constexpr = BLOCK_L + (KS - 1) * DIL  # 268
IN_TILE_SIZE: al.constexpr = IC * IN_TILE_L  # 8576
# Weight layout in shm: (IC, KS, BLOCK_OC), stored as f32 to avoid
# per-element convert in the innermost loop.
W_TILE_SIZE: al.constexpr = IC * KS * BLOCK_OC  # 2560


@avelang.jit
def conv_transpose1d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    L_in: al.i32,
    L_out: al.i32,
):
    tid = al.thread_id(0)
    block_l = al.block_id(0)
    block_b = al.block_id(1)
    block_oc = al.block_id(2)

    oc_start = block_oc * BLOCK_OC

    if block_b < B:
        x_flat = al.make_tensor(
            x_ptr, al.bf16, al.make_layout((B * IC * L_in,), (1,)),
        )
        w_flat = al.make_tensor(
            w_ptr, al.bf16, al.make_layout((IC * OC * KS,), (1,)),
        )

        shm_input = al.make_shared((IN_TILE_SIZE,), al.bf16)
        # Weights pre-converted to f32 to save convert in the inner loop.
        shm_weight = al.make_shared((W_TILE_SIZE,), al.f32)

        for idx in al.range(tid, W_TILE_SIZE, BLOCK_L):
            ic_idx = idx // (KS * BLOCK_OC)
            rest = idx - ic_idx * (KS * BLOCK_OC)
            k_idx = rest // BLOCK_OC
            loc_oc = rest - k_idx * BLOCK_OC
            w_global = ic_idx * OC * KS + (oc_start + loc_oc) * KS + k_idx
            shm_weight[idx] = al.convert(w_flat[w_global], al.f32)

        in_start = block_l * BLOCK_L - (KS - 1) * DIL
        in_base = block_b * IC * L_in
        for idx in al.range(tid, IN_TILE_SIZE, BLOCK_L):
            ic_idx = idx // IN_TILE_L
            l_idx = idx - ic_idx * IN_TILE_L
            g_pos = in_start + l_idx
            if g_pos >= 0 and g_pos < L_in:
                shm_input[idx] = x_flat[in_base + ic_idx * L_in + g_pos]
            else:
                shm_input[idx] = al.convert(0, al.bf16)

        al.syncthreads()

        out_pos = block_l * BLOCK_L + tid
        if out_pos < L_out:
            out = al.make_tensor(
                out_ptr, al.bf16,
                al.make_layout((B, OC, L_out), (OC * L_out, L_out, 1)),
            )

            acc = al.make_local((BLOCK_OC,), al.f32)
            for loc_oc in al.range(BLOCK_OC):
                acc[loc_oc] = al.convert(0.0, al.f32)

            for k in al.range(KS):
                in_pos = out_pos - k * DIL
                local_in_pos = in_pos - in_start
                for ic in al.range(IC):
                    x_val = al.convert(
                        shm_input[ic * IN_TILE_L + local_in_pos], al.f32,
                    )
                    w_base = ic * KS * BLOCK_OC + k * BLOCK_OC
                    for loc_oc in al.range(BLOCK_OC):
                        acc[loc_oc] = acc[loc_oc] + x_val * shm_weight[w_base + loc_oc]

            for loc_oc in al.range(BLOCK_OC):
                out[block_b, oc_start + loc_oc, out_pos] = al.convert(
                    acc[loc_oc], al.bf16,
                )


def avelang_conv_transpose1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    B: int,
    L_in: int,
    L_out: int,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16

    grid_l = (L_out + BLOCK_L - 1) // BLOCK_L
    grid = (grid_l, B, OC_GROUPS)

    out = torch.empty((B, OC, L_out), dtype=torch.bfloat16, device=x.device)
    conv_transpose1d_kernel[lambda: (grid, (BLOCK_L, 1, 1))](
        x, weight, out, B, L_in, L_out,
    )
    return out


class ModelNew(nn.Module):
    """Optimized ConvTranspose1d using an AveLang DSL GPU kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv1d_transpose.weight.data
        stride_val = self.conv1d_transpose.stride[0]
        padding_val = self.conv1d_transpose.padding[0]
        dilation_val = self.conv1d_transpose.dilation[0]
        output_padding_val = self.conv1d_transpose.output_padding[0]
        has_bias = self.conv1d_transpose.bias is not None

        B, ic_val, L_in = x.shape
        w_ic, w_oc, w_ks = weight.shape

        L_out = (
            (L_in - 1) * stride_val
            - 2 * padding_val
            + dilation_val * (w_ks - 1)
            + output_padding_val
            + 1
        )

        if ic_val != IC or w_oc != OC or w_ks != KS or dilation_val != DIL:
            raise ValueError(
                f"Kernel compiled for IC={IC}, OC={OC}, KS={KS}, DIL={DIL}; "
                f"got IC={ic_val}, OC={w_oc}, KS={w_ks}, DIL={dilation_val}"
            )
        if has_bias:
            raise ValueError("Bias support not implemented in AveLang kernel")
        if stride_val != 1 or padding_val != 0:
            raise ValueError(
                "Only stride=1, padding=0 supported in this kernel"
            )

        orig_dtype = x.dtype
        x_bf16 = x.contiguous().to(torch.bfloat16)
        w_bf16 = weight.contiguous().to(torch.bfloat16)

        result = avelang_conv_transpose1d(x_bf16, w_bf16, B, L_in, L_out)
        return result.to(orig_dtype)


# Test code (mirrors input_model.py contract)
batch_size = 32
in_channels = 32
out_channels = 64
kernel_size = 5
length = 131072
stride = 1
padding = 0
dilation = 3


def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
