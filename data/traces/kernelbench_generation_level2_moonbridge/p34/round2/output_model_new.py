import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
W_OUT: al.constexpr = 64
POSITIONS_PER_BLOCK: al.constexpr = BLOCK_SIZE // W_OUT


@avelang.jit
def layernorm_gelu_kernel(
    input_ptr: al.Pointer(al.bf16),
    ln_weight_ptr: al.Pointer(al.bf16),
    ln_bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    num_rows: al.i32,
    W: al.i32,
    eps: al.f32,
    scaling_factor: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    pos_in_block = tid // W_OUT
    lane = tid - pos_in_block * W_OUT

    global_pos = bid * POSITIONS_PER_BLOCK + pos_in_block

    if global_pos < num_rows:
        row_base = global_pos * W

        in_flat = al.make_tensor(input_ptr, al.bf16, al.make_layout((num_rows * W,), (1,)))
        ln_w_t = al.make_tensor(ln_weight_ptr, al.bf16, al.make_layout((W,), (1,)))
        ln_b_t = al.make_tensor(ln_bias_ptr, al.bf16, al.make_layout((W,), (1,)))

        idx = row_base + lane
        val = al.convert(in_flat[idx], al.f32)

        # --- Reduction across W=64 lanes ---
        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        smem_sum[tid] = val
        smem_sq[tid] = val * val
        al.syncthreads()

        if lane < 32:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
        al.syncthreads()
        if lane < 16:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
        al.syncthreads()
        if lane < 8:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
        al.syncthreads()
        if lane < 4:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
        al.syncthreads()
        if lane < 2:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
        al.syncthreads()
        if lane < 1:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]

        pos_start = pos_in_block * W_OUT
        if lane == 0:
            w_f32 = al.convert(W, al.f32)
            mean = smem_sum[tid] / w_f32
            var = smem_sq[tid] / w_f32 - mean * mean
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)
            smem_sum[tid] = mean
            smem_sq[tid] = rstd
        al.syncthreads()

        mean_val = smem_sum[pos_start]
        rstd_val = smem_sq[pos_start]

        ln_w_val = al.convert(ln_w_t[lane], al.f32)
        ln_b_val = al.convert(ln_b_t[lane], al.f32)

        normalized = (val - mean_val) * rstd_val
        result = normalized * ln_w_val + ln_b_val

        # GELU: exact erf-based
        sqrt_2 = al.convert(1.4142135623730951, al.f32)
        half = al.convert(0.5, al.f32)
        one = al.convert(1.0, al.f32)
        gelu_val = half * result * (one + al.erf(result / sqrt_2))

        out_val = gelu_val * scaling_factor

        out_flat = al.make_tensor(output_ptr, al.bf16, al.make_layout((num_rows * W,), (1,)))
        out_flat[idx] = al.convert(out_val, al.bf16)


def _ensure_bf16_cuda(t: torch.Tensor) -> torch.Tensor:
    if not t.is_cuda:
        t = t.cuda()
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.contiguous()


def avelang_layernorm_gelu(
    x_conv: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float,
    scaling_factor: float,
) -> torch.Tensor:
    x_bf16 = _ensure_bf16_cuda(x_conv)
    ln_w_bf16 = _ensure_bf16_cuda(ln_weight)
    ln_b_bf16 = _ensure_bf16_cuda(ln_bias)

    shape = x_bf16.shape
    N, OC, D_out, H_out, W_out = shape
    num_rows = N * OC * D_out * H_out
    W = W_out

    x_flat = x_bf16.reshape(num_rows, W).contiguous()
    out_flat = torch.empty((num_rows, W), dtype=torch.bfloat16, device=x_bf16.device)

    num_blocks = (num_rows + POSITIONS_PER_BLOCK - 1) // POSITIONS_PER_BLOCK

    layernorm_gelu_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_flat,
        ln_w_bf16,
        ln_b_bf16,
        out_flat,
        num_rows,
        W,
        eps,
        scaling_factor,
    )
    return out_flat.reshape(shape)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        eps=1e-5,
        scaling_factor=1.0,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor
        self.eps = eps

    def forward(self, x):
        x = self.conv_transpose(x)
        x = avelang_layernorm_gelu(
            x,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.eps,
            self.scaling_factor,
        )
        return x


batch_size = 32
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]
