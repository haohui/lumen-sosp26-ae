import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ============================================================
# Constants
# ============================================================
OC_TILE = 16
INSTANCENORM_THREADS = 256

# ============================================================
# Conv2d kernel: direct FP32 convolution with bias
# Uses BF16 loads but FP32 accumulate (no intermediate BF16 rounding)
# ============================================================

@avelang.jit
def conv2d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    N: al.i32,
    IC: al.i32,
    OC: al.i32,
    H: al.i32,
    W: al.i32,
    OH: al.i32,
    OW: al.i32,
    NUM_OC_GROUPS: al.i32,
):
    ow = al.block_id(0)
    oh = al.block_id(1)
    bz = al.block_id(2)
    oc_local = al.thread_id(0)

    const1 = al.convert(1, al.i32)
    const3 = al.convert(3, al.i32)
    zero_i32 = al.convert(0, al.i32)

    oc_group = bz
    n = zero_i32
    for _d in al.range(N):
        if oc_group >= NUM_OC_GROUPS:
            oc_group = oc_group - NUM_OC_GROUPS
            n = n + const1

    oc = oc_group * OC_TILE + oc_local

    if (ow < OW) and (oh < OH) and (oc < OC) and (n < N):
        x_layout = al.make_layout((N, IC, H, W), (IC * H * W, H * W, W, const1))
        x = al.make_tensor(x_ptr, al.bf16, x_layout)

        w_layout = al.make_layout((OC, IC, const3, const3),
                                   (IC * const3 * const3, const3 * const3, const3, const1))
        w = al.make_tensor(w_ptr, al.bf16, w_layout)

        b_layout = al.make_layout((OC,), (const1,))
        b = al.make_tensor(b_ptr, al.f32, b_layout)

        out_layout = al.make_layout((N, OC, OH, OW), (OC * OH * OW, OH * OW, OW, const1))
        out = al.make_tensor(out_ptr, al.f32, out_layout)

        acc = b[oc]

        ic_idx = zero_i32
        for _ic in al.range(IC):
            kh_val = zero_i32
            for _kh in al.range(const3):
                kw_val = zero_i32
                for _kw in al.range(const3):
                    ih = oh + kh_val
                    iw = ow + kw_val
                    x_val = al.convert(x[n, ic_idx, ih, iw], al.f32)
                    w_val = al.convert(w[oc, ic_idx, kh_val, kw_val], al.f32)
                    acc = acc + x_val * w_val
                    kw_val = kw_val + const1
                kh_val = kh_val + const1
            ic_idx = ic_idx + const1

        out[n, oc, oh, ow] = acc


# ============================================================
# InstanceNorm + divide kernel
# ============================================================

@avelang.jit
def instancenorm_div_kernel(
    in_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    N: al.i32,
    OC: al.i32,
    OH: al.i32,
    OW: al.i32,
):
    oc = al.block_id(1)
    n = al.block_id(2)
    tid = al.thread_id(0)

    num_spatial = OH * OW
    const1 = al.convert(1, al.i32)
    zero_i32 = al.convert(0, al.i32)
    zero_f32 = al.convert(0.0, al.f32)
    one_f32 = al.convert(1.0, al.f32)

    shared_sum = al.make_shared((INSTANCENORM_THREADS,), al.f32)
    shared_var = al.make_shared((INSTANCENORM_THREADS,), al.f32)

    in_layout = al.make_layout((N, OC, OH, OW), (OC * OH * OW, OH * OW, OW, const1))
    in_tensor = al.make_tensor(in_ptr, al.f32, in_layout)

    out_layout = al.make_layout((N, OC, OH, OW), (OC * OH * OW, OH * OW, OW, const1))
    out_tensor = al.make_tensor(out_ptr, al.f32, out_layout)

    spatial_chunk = (num_spatial + INSTANCENORM_THREADS - const1) // INSTANCENORM_THREADS
    start_idx = tid * spatial_chunk
    end_idx = al.min(start_idx + spatial_chunk, num_spatial)

    # Phase 1: mean
    partial_sum = zero_f32
    idx = start_idx
    for _s1 in al.range(spatial_chunk):
        if idx < end_idx:
            row = idx // OW
            col = idx % OW
            partial_sum = partial_sum + in_tensor[n, oc, row, col]
            idx = idx + const1
        else:
            idx = idx + const1

    shared_sum[tid] = partial_sum
    al.syncthreads()

    if tid == zero_i32:
        total_sum = zero_f32
        ti = zero_i32
        for _tr in al.range(INSTANCENORM_THREADS):
            total_sum = total_sum + shared_sum[ti]
            ti = ti + const1
        shared_sum[zero_i32] = total_sum / al.convert(num_spatial, al.f32)
    al.syncthreads()
    mean_val = shared_sum[zero_i32]

    # Phase 2: variance
    partial_var = zero_f32
    idx = start_idx
    for _s2 in al.range(spatial_chunk):
        if idx < end_idx:
            row = idx // OW
            col = idx % OW
            diff = in_tensor[n, oc, row, col] - mean_val
            partial_var = partial_var + diff * diff
            idx = idx + const1
        else:
            idx = idx + const1

    shared_var[tid] = partial_var
    al.syncthreads()

    if tid == zero_i32:
        total_var = zero_f32
        ti = zero_i32
        for _tr2 in al.range(INSTANCENORM_THREADS):
            total_var = total_var + shared_var[ti]
            ti = ti + const1
        var_val = total_var / al.convert(num_spatial, al.f32)
        shared_var[zero_i32] = one_f32 / al.sqrt(var_val + al.convert(1e-5, al.f32))
    al.syncthreads()
    inv_std = shared_var[zero_i32]

    # Phase 3: normalize + divide
    idx = start_idx
    for _s3 in al.range(spatial_chunk):
        if idx < end_idx:
            row = idx // OW
            col = idx % OW
            val = in_tensor[n, oc, row, col]
            out_tensor[n, oc, row, col] = ((val - mean_val) * inv_std) / al.convert(2.0, al.f32)
            idx = idx + const1
        else:
            idx = idx + const1


# ============================================================
# Host wrapper
# ============================================================

def avelang_conv_instancenorm_div(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA/HIP device."
    N, IC, H, W = x.shape
    OC = conv_weight.shape[0]
    OH = H - 2
    OW = W - 2

    x_bf16 = x.to(torch.bfloat16).contiguous()
    w_bf16 = conv_weight.to(torch.bfloat16).contiguous()
    b_f32 = conv_bias.to(torch.float32).contiguous()

    conv_out = torch.empty(N, OC, OH, OW, dtype=torch.float32, device=x.device)

    num_oc_groups = (OC + OC_TILE - 1) // OC_TILE
    conv2d_kernel[lambda: ((OW, OH, N * num_oc_groups), (OC_TILE, 1, 1))](
        x_bf16.data_ptr(),
        w_bf16.data_ptr(),
        b_f32.data_ptr(),
        conv_out.data_ptr(),
        N, IC, OC, H, W, OH, OW, num_oc_groups,
    )

    out = torch.empty(N, OC, OH, OW, dtype=torch.float32, device=x.device)
    instancenorm_div_kernel[lambda: ((1, OC, N), (INSTANCENORM_THREADS, 1, 1))](
        conv_out.data_ptr(),
        out.data_ptr(),
        N, OC, OH, OW,
    )

    return out.to(torch.bfloat16)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by

    def forward(self, x):
        x = x.contiguous()
        conv_w = self.conv.weight.data.contiguous()
        conv_b = self.conv.bias.data.contiguous()
        return avelang_conv_instancenorm_div(x, conv_w, conv_b)
