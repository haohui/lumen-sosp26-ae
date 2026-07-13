import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256
KH: al.constexpr = 3
KW: al.constexpr = 3


@avelang.jit
def input_reduce_kernel(
    input_ptr: al.Pointer(al.bf16),
    intermed_ptr: al.Pointer(al.f32),
    B: al.i32,
    IC: al.i32,
    H: al.i32,
    W: al.i32,
):
    """
    Reduce input over spatial dimensions, computing four partials per (b, ic):
      intermed[..., 0] = sum_{h,w} input[b,ic,h,w]                (spatial_sum)
      intermed[..., 1] = sum_{w}   input[b,ic,0,w]                (row0_sum)
      intermed[..., 2] = sum_{h}   input[b,ic,h,0]                (col0_sum)
      intermed[..., 3] =            input[b,ic,0,0]               (corner)

    One block per (b, ic).  Threads stride over spatial positions,
    then a shared-memory tree reduction combines all four partials.
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    b = bid // IC
    ic = bid % IC

    if b >= B:
        return

    in_layout = al.make_layout((B, IC, H, W), (IC * H * W, H * W, W, 1))
    input_t = al.make_tensor(input_ptr, al.bf16, in_layout)

    total_spatial = H * W

    acc_sum = al.convert(0.0, al.f32)
    acc_row0 = al.convert(0.0, al.f32)
    acc_col0 = al.convert(0.0, al.f32)
    acc_corner = al.convert(0.0, al.f32)

    for pos in al.range(tid, total_spatial, BLOCK_SIZE):
        h_in = pos // W
        w_in = pos % W

        val = al.convert(input_t[b, ic, h_in, w_in], al.f32)
        acc_sum = acc_sum + val

        if h_in == 0:
            acc_row0 = acc_row0 + val
        if w_in == 0:
            acc_col0 = acc_col0 + val
        if h_in == 0 and w_in == 0:
            acc_corner = val

    smem = al.make_shared((BLOCK_SIZE * 4,), al.f32)
    smem[tid] = acc_sum
    smem[tid + BLOCK_SIZE] = acc_row0
    smem[tid + 2 * BLOCK_SIZE] = acc_col0
    smem[tid + 3 * BLOCK_SIZE] = acc_corner
    al.syncthreads()

    if tid < 128:
        smem[tid] = smem[tid] + smem[tid + 128]
        smem[tid + BLOCK_SIZE] = smem[tid + BLOCK_SIZE] + smem[tid + BLOCK_SIZE + 128]
        smem[tid + 2 * BLOCK_SIZE] = smem[tid + 2 * BLOCK_SIZE] + smem[tid + 2 * BLOCK_SIZE + 128]
        smem[tid + 3 * BLOCK_SIZE] = smem[tid + 3 * BLOCK_SIZE] + smem[tid + 3 * BLOCK_SIZE + 128]
    al.syncthreads()
    if tid < 64:
        smem[tid] = smem[tid] + smem[tid + 64]
        smem[tid + BLOCK_SIZE] = smem[tid + BLOCK_SIZE] + smem[tid + BLOCK_SIZE + 64]
        smem[tid + 2 * BLOCK_SIZE] = smem[tid + 2 * BLOCK_SIZE] + smem[tid + 2 * BLOCK_SIZE + 64]
        smem[tid + 3 * BLOCK_SIZE] = smem[tid + 3 * BLOCK_SIZE] + smem[tid + 3 * BLOCK_SIZE + 64]
    al.syncthreads()
    if tid < 32:
        smem[tid] = smem[tid] + smem[tid + 32]
        smem[tid + BLOCK_SIZE] = smem[tid + BLOCK_SIZE] + smem[tid + BLOCK_SIZE + 32]
        smem[tid + 2 * BLOCK_SIZE] = smem[tid + 2 * BLOCK_SIZE] + smem[tid + 2 * BLOCK_SIZE + 32]
        smem[tid + 3 * BLOCK_SIZE] = smem[tid + 3 * BLOCK_SIZE] + smem[tid + 3 * BLOCK_SIZE + 32]
    al.syncthreads()
    if tid < 16:
        smem[tid] = smem[tid] + smem[tid + 16]
        smem[tid + BLOCK_SIZE] = smem[tid + BLOCK_SIZE] + smem[tid + BLOCK_SIZE + 16]
        smem[tid + 2 * BLOCK_SIZE] = smem[tid + 2 * BLOCK_SIZE] + smem[tid + 2 * BLOCK_SIZE + 16]
        smem[tid + 3 * BLOCK_SIZE] = smem[tid + 3 * BLOCK_SIZE] + smem[tid + 3 * BLOCK_SIZE + 16]
    al.syncthreads()
    if tid < 8:
        smem[tid] = smem[tid] + smem[tid + 8]
        smem[tid + BLOCK_SIZE] = smem[tid + BLOCK_SIZE] + smem[tid + BLOCK_SIZE + 8]
        smem[tid + 2 * BLOCK_SIZE] = smem[tid + 2 * BLOCK_SIZE] + smem[tid + 2 * BLOCK_SIZE + 8]
        smem[tid + 3 * BLOCK_SIZE] = smem[tid + 3 * BLOCK_SIZE] + smem[tid + 3 * BLOCK_SIZE + 8]
    al.syncthreads()
    if tid < 4:
        smem[tid] = smem[tid] + smem[tid + 4]
        smem[tid + BLOCK_SIZE] = smem[tid + BLOCK_SIZE] + smem[tid + BLOCK_SIZE + 4]
        smem[tid + 2 * BLOCK_SIZE] = smem[tid + 2 * BLOCK_SIZE] + smem[tid + 2 * BLOCK_SIZE + 4]
        smem[tid + 3 * BLOCK_SIZE] = smem[tid + 3 * BLOCK_SIZE] + smem[tid + 3 * BLOCK_SIZE + 4]
    al.syncthreads()
    if tid < 2:
        smem[tid] = smem[tid] + smem[tid + 2]
        smem[tid + BLOCK_SIZE] = smem[tid + BLOCK_SIZE] + smem[tid + BLOCK_SIZE + 2]
        smem[tid + 2 * BLOCK_SIZE] = smem[tid + 2 * BLOCK_SIZE] + smem[tid + 2 * BLOCK_SIZE + 2]
        smem[tid + 3 * BLOCK_SIZE] = smem[tid + 3 * BLOCK_SIZE] + smem[tid + 3 * BLOCK_SIZE + 2]
    al.syncthreads()
    if tid < 1:
        smem[tid] = smem[tid] + smem[tid + 1]
        smem[tid + BLOCK_SIZE] = smem[tid + BLOCK_SIZE] + smem[tid + BLOCK_SIZE + 1]
        smem[tid + 2 * BLOCK_SIZE] = smem[tid + 2 * BLOCK_SIZE] + smem[tid + 2 * BLOCK_SIZE + 1]
        smem[tid + 3 * BLOCK_SIZE] = smem[tid + 3 * BLOCK_SIZE] + smem[tid + 3 * BLOCK_SIZE + 1]

    if tid == 0:
        im_layout = al.make_layout((B, IC, 4), (IC * 4, 4, 1))
        im = al.make_tensor(intermed_ptr, al.f32, im_layout)
        im[b, ic, 0] = smem[0]
        im[b, ic, 1] = smem[BLOCK_SIZE]
        im[b, ic, 2] = smem[2 * BLOCK_SIZE]
        im[b, ic, 3] = smem[3 * BLOCK_SIZE]


@avelang.jit
def final_accumulate_kernel(
    intermed_ptr: al.Pointer(al.f32),
    w_all_ptr: al.Pointer(al.f32),
    w_row0_ptr: al.Pointer(al.f32),
    w_col0_ptr: al.Pointer(al.f32),
    w_corner_ptr: al.Pointer(al.f32),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    IC: al.i32,
    OC: al.i32,
    scale: al.f32,
    bias_scale: al.f32,
):
    """
    Combine the four input partials with precomputed weight sums to produce
    the final (B, OC, 1, 1) output.

    result[b, oc] = bias[oc] * bias_scale
                  + scale * sum_ic(
                        spatial_sum * w_all[ic,oc]
                      - row0_sum    * w_row0[ic,oc]
                      - col0_sum    * w_col0[ic,oc]
                      + corner      * w_corner[ic,oc]
                    )
    """
    tid = al.thread_id(0)
    bid = al.block_id(0)

    idx = bid * BLOCK_SIZE + tid
    total = B * OC

    if idx >= total:
        return

    b = idx // OC
    oc = idx % OC

    im_layout = al.make_layout((B, IC, 4), (IC * 4, 4, 1))
    im = al.make_tensor(intermed_ptr, al.f32, im_layout)

    wa_layout = al.make_layout((IC, OC), (OC, 1))
    w_all = al.make_tensor(w_all_ptr, al.f32, wa_layout)
    w_row0 = al.make_tensor(w_row0_ptr, al.f32, wa_layout)
    w_col0 = al.make_tensor(w_col0_ptr, al.f32, wa_layout)
    w_corner = al.make_tensor(w_corner_ptr, al.f32, wa_layout)

    b_layout = al.make_layout((OC,), (1,))
    bias_t = al.make_tensor(bias_ptr, al.bf16, b_layout)

    out_layout = al.make_layout((B, OC, 1, 1), (OC, 1, 1, 1))
    output_t = al.make_tensor(output_ptr, al.bf16, out_layout)

    acc_total = al.convert(0.0, al.f32)

    for ic in al.range(IC):
        s_sum = im[b, ic, 0]
        s_row0 = im[b, ic, 1]
        s_col0 = im[b, ic, 2]
        s_corner = im[b, ic, 3]

        w_a = w_all[ic, oc]
        w_r0 = w_row0[ic, oc]
        w_c0 = w_col0[ic, oc]
        w_c = w_corner[ic, oc]

        term = al.convert(
            s_sum * w_a - s_row0 * w_r0 - s_col0 * w_c0 + s_corner * w_c,
            al.f32,
        )
        acc_total = al.convert(acc_total + term, al.f32)

    bias_val = al.convert(bias_t[oc], al.f32)
    result = bias_val * bias_scale + acc_total * scale
    output_t[b, oc, 0, 0] = al.convert(result, al.bf16)


def avelang_fused_convtranspose_pool(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    multiplier: float,
) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."
    assert x.ndim == 4, f"Expected 4D input, got {x.ndim}D"

    B, IC, H, W = x.shape
    OC = weight.shape[1]

    OH = (H - 1) * stride - 2 * padding + KH + output_padding
    OW = (W - 1) * stride - 2 * padding + KW + output_padding

    # Precompute weight sums on host (they are tiny: IC * OC = 64 * 128 = 8192 each)
    w_f32 = weight.to(dtype=torch.float32)
    w_all_sum = w_f32.sum(dim=(2, 3))       # (IC, OC) — sum over kh, kw
    w_row0_sum = w_f32[:, :, 0, :].sum(dim=2)  # (IC, OC) — sum over kw at kh=0
    w_col0_sum = w_f32[:, :, :, 0].sum(dim=2)  # (IC, OC) — sum over kh at kw=0
    w_corner = w_f32[:, :, 0, 0]               # (IC, OC) — just the corner

    w_all_gpu = w_all_sum.contiguous().to(device=x.device)
    w_row0_gpu = w_row0_sum.contiguous().to(device=x.device)
    w_col0_gpu = w_col0_sum.contiguous().to(device=x.device)
    w_corner_gpu = w_corner.contiguous().to(device=x.device)

    intermed = torch.empty((B, IC, 4), dtype=torch.float32, device=x.device)
    output = torch.empty((B, OC, 1, 1), dtype=torch.bfloat16, device=x.device)

    scale = float(multiplier) / float(OH * OW)
    bias_scale = float(multiplier)

    # Phase 1: reduce input over spatial dims
    num_blocks_reduce = B * IC
    input_reduce_kernel[lambda: ((num_blocks_reduce, 1, 1), (BLOCK_SIZE, 1, 1))](
        x, intermed,
        B, IC, H, W,
    )

    # Phase 2: accumulate across IC with precomputed weight sums
    num_blocks_final = (B * OC + BLOCK_SIZE - 1) // BLOCK_SIZE
    final_accumulate_kernel[lambda: ((num_blocks_final, 1, 1), (BLOCK_SIZE, 1, 1))](
        intermed, w_all_gpu, w_row0_gpu, w_col0_gpu, w_corner_gpu,
        bias, output,
        B, IC, OC,
        scale, bias_scale,
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.multiplier = multiplier

    def forward(self, x):
        orig_dtype = x.dtype
        x_bf16 = x.to(dtype=torch.bfloat16, device=x.device).contiguous()

        weight = self.conv_transpose.weight.data
        bias = self.conv_transpose.bias.data

        w_bf16 = weight.to(dtype=torch.bfloat16, device=x.device).contiguous()
        b_bf16 = bias.to(dtype=torch.bfloat16, device=x.device).contiguous()

        result = avelang_fused_convtranspose_pool(
            x_bf16, w_bf16, b_bf16,
            self.stride, self.padding, self.output_padding,
            self.multiplier,
        )

        return result.to(orig_dtype)
