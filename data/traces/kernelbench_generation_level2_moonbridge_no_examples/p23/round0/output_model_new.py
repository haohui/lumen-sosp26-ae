import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ============================================================
# Conv3D kernel: tiled with shared memory, BF16 I/O, FP32 acc
# ============================================================

@avelang.jit
def conv3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    K: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    b = al.block_id(0)
    oc_pair = al.block_id(1)
    tile_idx = al.block_id(2)
    tid = al.thread_id(0)
    nt = al.block_dim(0)

    # Two output channels per block
    oc0 = oc_pair * 2
    oc1 = oc0 + 1

    # Thread mapping to output position within tile (TD=4, TH=8, TW=8)
    td = tid // 64
    rem_hw = tid % 64
    th = rem_hw // 8
    tw = rem_hw % 8

    # Compute tile position in output space
    num_tiles_h = (H_out + 7) // 8
    num_tiles_w = (W_out + 7) // 8
    num_tiles_hw = num_tiles_h * num_tiles_w

    tile_d_idx = tile_idx // num_tiles_hw
    rem = tile_idx % num_tiles_hw
    tile_h_idx = rem // num_tiles_w
    tile_w_idx = rem % num_tiles_w

    d_start = tile_d_idx * 4
    h_start = tile_h_idx * 8
    w_start = tile_w_idx * 8

    # Clamp tile size at boundaries
    d_tile = 4
    if d_start + 4 > D_out:
        d_tile = D_out - d_start
    h_tile = 8
    if h_start + 8 > H_out:
        h_tile = H_out - h_start
    w_tile = 8
    if w_start + 8 > W_out:
        w_tile = W_out - w_start

    # Global output position for this thread
    d_out = d_start + td
    h_out = h_start + th
    w_out = w_start + tw

    # Strides
    input_chan_stride = D * H * W
    input_spatial_stride = H * W
    weight_ic_stride = K * K * K
    weight_kd_stride = K * K
    weight_kh_stride = K
    output_spatial_stride = H_out * W_out
    output_chan_stride = D_out * H_out * W_out

    full_input_sz = B * C_in * input_chan_stride
    full_output_sz = B * C_out * output_chan_stride
    weight_total = C_in * K * K * K

    in_lay = al.make_layout((full_input_sz,), (1,))
    input_t = al.make_tensor(input_ptr, al.bf16, in_lay)
    wk_lay = al.make_layout((weight_total,), (1,))
    weight_t = al.make_tensor(weight_ptr, al.bf16, wk_lay)
    bias_lay = al.make_layout((C_out,), (1,))
    bias_t = al.make_tensor(bias_ptr, al.bf16, bias_lay)
    out_lay = al.make_layout((full_output_sz,), (1,))
    output_t = al.make_tensor(output_ptr, al.bf16, out_lay)

    bias0_f32 = al.convert(bias_t[oc0], al.f32)
    bias1_f32 = al.convert(bias_t[oc1], al.f32)

    input_batch_base = b * C_in * input_chan_stride
    weight_oc0_base = oc0 * weight_total
    weight_oc1_base = oc1 * weight_total
    out_batch_base = b * C_out * output_chan_stride
    out_oc0_base = out_batch_base + oc0 * output_chan_stride
    out_oc1_base = out_batch_base + oc1 * output_chan_stride

    # Shared memory for input tile: 6*10*10 = 600 BF16 elements
    in_sh = al.make_shared((600,), al.bf16)

    acc0 = bias0_f32
    acc1 = bias1_f32

    for ic in al.range(C_in):
        input_ic_base = input_batch_base + ic * input_chan_stride
        w_ic_stride = ic * weight_ic_stride

        # Cooperative load: 600 elements, 256 threads -> ~3 loads each
        for ld in al.range(3):
            ld_idx = tid * 3 + ld
            if ld_idx < 600:
                ld_d = ld_idx // 100
                ld_rem = ld_idx % 100
                ld_h = ld_rem // 10
                ld_w = ld_rem % 10

                g_d = d_start + ld_d
                g_h = h_start + ld_h
                g_w = w_start + ld_w

                valid = 1
                if g_d >= D:
                    valid = 0
                if g_h >= H:
                    valid = 0
                if g_w >= W:
                    valid = 0
                if valid != 0:
                    g_idx = input_ic_base + g_d * input_spatial_stride + g_h * W + g_w
                    in_sh[ld_idx] = input_t[g_idx]
                if valid == 0:
                    in_sh[ld_idx] = al.convert(0.0, al.bf16)

        al.syncthreads()

        # Compute contributions for both output channels
        valid_out = 1
        if td >= d_tile:
            valid_out = 0
        if th >= h_tile:
            valid_out = 0
        if tw >= w_tile:
            valid_out = 0
        if valid_out != 0:
            for kd in al.range(K):
                in_d = td + kd
                in_d_off = in_d * 100
                for kh in al.range(K):
                    in_h = th + kh
                    in_h_off = in_h * 10
                    for kw in al.range(K):
                        in_w = tw + kw
                        in_idx = in_d_off + in_h_off + in_w

                        w_kernel_base = kd * weight_kd_stride + kh * weight_kh_stride + kw
                        w0_idx = weight_oc0_base + w_ic_stride + w_kernel_base
                        w1_idx = weight_oc1_base + w_ic_stride + w_kernel_base

                        inp_val = al.convert(in_sh[in_idx], al.f32)
                        w0_val = al.convert(weight_t[w0_idx], al.f32)
                        w1_val = al.convert(weight_t[w1_idx], al.f32)
                        acc0 = acc0 + inp_val * w0_val
                        acc1 = acc1 + inp_val * w1_val

        al.syncthreads()

    # Write outputs
    valid_out = 1
    if td >= d_tile:
        valid_out = 0
    if th >= h_tile:
        valid_out = 0
    if tw >= w_tile:
        valid_out = 0
    if valid_out != 0:
        out_spatial_idx = d_out * output_spatial_stride + h_out * W_out + w_out
        out0_idx = out_oc0_base + out_spatial_idx
        out1_idx = out_oc1_base + out_spatial_idx
        output_t[out0_idx] = al.convert(acc0, al.bf16)
        output_t[out1_idx] = al.convert(acc1, al.bf16)


# ============================================================
# GroupNorm kernel: training-mode, per-(batch,group) stats
# ============================================================

@avelang.jit
def group_norm_kernel(
    input_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_out: al.i32,
    G: al.i32,
    C_per_group: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
):
    b = al.block_id(0)
    g = al.block_id(1)
    tid = al.thread_id(0)
    nt = al.block_dim(0)

    spatial_size = D * H * W
    total_per_group = C_per_group * spatial_size
    total_elts = B * C_out * spatial_size
    eps = al.convert(1e-5, al.f32)

    sum_sh = al.make_shared((256,), al.f32)
    sq_sh = al.make_shared((256,), al.f32)

    in_lay = al.make_layout((total_elts,), (1,))
    input_t = al.make_tensor(input_ptr, al.bf16, in_lay)

    gamma_lay = al.make_layout((C_out,), (1,))
    gamma_t = al.make_tensor(gamma_ptr, al.bf16, gamma_lay)

    beta_lay = al.make_layout((C_out,), (1,))
    beta_t = al.make_tensor(beta_ptr, al.bf16, beta_lay)

    out_lay = al.make_layout((total_elts,), (1,))
    output_t = al.make_tensor(output_ptr, al.bf16, out_lay)

    group_base = b * C_out * spatial_size + g * C_per_group * spatial_size

    # --- Phase 1: compute partial sums ---
    chunk = (total_per_group + nt - 1) // nt
    local_sum = al.convert(0.0, al.f32)
    local_sq = al.convert(0.0, al.f32)

    for i in al.range(chunk):
        idx = tid * chunk + i
        if idx < total_per_group:
            val = al.convert(input_t[group_base + idx], al.f32)
            local_sum = local_sum + val
            local_sq = local_sq + val * val

    sum_sh[tid] = local_sum
    sq_sh[tid] = local_sq
    al.syncthreads()

    # --- Phase 2: tree reduction ---
    stride = 1
    for _ in al.range(8):
        if tid % (2 * stride) == 0:
            sum_sh[tid] = sum_sh[tid] + sum_sh[tid + stride]
            sq_sh[tid] = sq_sh[tid] + sq_sh[tid + stride]
        stride = stride * 2
        al.syncthreads()

    # --- Phase 3: normalize ---
    total_f = al.convert(total_per_group, al.f32)
    mean_val = sum_sh[0] / total_f
    var_val = sq_sh[0] / total_f - mean_val * mean_val
    inv_std = al.convert(1.0, al.f32) / al.sqrt(var_val + eps)

    for i in al.range(chunk):
        idx = tid * chunk + i
        if idx < total_per_group:
            val_f32 = al.convert(input_t[group_base + idx], al.f32)
            local_c = idx // spatial_size
            global_c = g * C_per_group + local_c

            gamma_val = al.convert(gamma_t[global_c], al.f32)
            beta_val = al.convert(beta_t[global_c], al.f32)

            normalized = (val_f32 - mean_val) * inv_std
            result = gamma_val * normalized + beta_val

            output_t[group_base + idx] = al.convert(result, al.bf16)


# ============================================================
# Mean reduction kernel: reduce all spatial+channel dims per batch
# ============================================================

@avelang.jit
def mean_reduce_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_out: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
):
    b = al.block_id(0)
    tid = al.thread_id(0)
    nt = al.block_dim(0)

    total_per_batch = C_out * D * H * W
    total_input = B * total_per_batch

    in_lay = al.make_layout((total_input,), (1,))
    input_t = al.make_tensor(input_ptr, al.bf16, in_lay)

    out_lay = al.make_layout((B,), (1,))
    output_t = al.make_tensor(output_ptr, al.bf16, out_lay)

    sum_sh = al.make_shared((256,), al.f32)

    chunk = (total_per_batch + nt - 1) // nt
    local_sum = al.convert(0.0, al.f32)

    batch_base = b * total_per_batch
    for i in al.range(chunk):
        idx = tid * chunk + i
        if idx < total_per_batch:
            val = al.convert(input_t[batch_base + idx], al.f32)
            local_sum = local_sum + val

    sum_sh[tid] = local_sum
    al.syncthreads()

    # Tree reduction
    stride = 1
    for _ in al.range(8):
        if tid % (2 * stride) == 0:
            sum_sh[tid] = sum_sh[tid] + sum_sh[tid + stride]
        stride = stride * 2
        al.syncthreads()

    if tid == 0:
        total_f = al.convert(total_per_batch, al.f32)
        result = sum_sh[0] / total_f
        output_t[b] = al.convert(result, al.bf16)


# ============================================================
# ModelNew: host-side wrapper
# ============================================================

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        assert x.is_cuda, "Input must be on CUDA/HIP device"
        B = x.shape[0]
        C_in = x.shape[1]
        D_in = x.shape[2]
        H_in = x.shape[3]
        W_in = x.shape[4]

        K = self.kernel_size
        D_out = D_in - K + 1
        H_out = H_in - K + 1
        W_out = W_in - K + 1
        C_out = self.out_channels
        G = self.num_groups
        C_per_group = C_out // G

        # Convert input and parameters to BF16, contiguous
        x_bf16 = x.to(torch.bfloat16).contiguous()
        w_bf16 = self.conv.weight.data.to(torch.bfloat16).contiguous()
        b_bf16 = self.conv.bias.data.to(torch.bfloat16).contiguous()
        gn_w_bf16 = self.group_norm.weight.data.to(torch.bfloat16).contiguous()
        gn_b_bf16 = self.group_norm.bias.data.to(torch.bfloat16).contiguous()

        # Allocate intermediate tensors
        conv_out = torch.empty(
            B, C_out, D_out, H_out, W_out,
            dtype=torch.bfloat16, device=x.device
        ).contiguous()

        gn_out = torch.empty(
            B, C_out, D_out, H_out, W_out,
            dtype=torch.bfloat16, device=x.device
        ).contiguous()

        output = torch.empty(B, dtype=torch.bfloat16, device=x.device)

        # Compute number of spatial tiles
        num_tiles_d = (D_out + 3) // 4
        num_tiles_h = (H_out + 7) // 8
        num_tiles_w = (W_out + 7) // 8
        num_tiles = num_tiles_d * num_tiles_h * num_tiles_w

        # Launch conv3d kernel (tiled, 2 output channels per block)
        num_oc_pairs = (C_out + 1) // 2
        conv3d_kernel[lambda: ((B, num_oc_pairs, num_tiles), (256, 1, 1))](
            x_bf16.data_ptr(), w_bf16.data_ptr(), b_bf16.data_ptr(),
            conv_out.data_ptr(),
            B, C_in, C_out, D_in, H_in, W_in, K, D_out, H_out, W_out,
        )

        # Launch group_norm kernel
        group_norm_kernel[lambda: ((B, G, 1), (256, 1, 1))](
            conv_out.data_ptr(), gn_w_bf16.data_ptr(), gn_b_bf16.data_ptr(),
            gn_out.data_ptr(),
            B, C_out, G, C_per_group, D_out, H_out, W_out,
        )

        # Launch mean reduction kernel
        mean_reduce_kernel[lambda: ((B, 1, 1), (256, 1, 1))](
            gn_out.data_ptr(), output.data_ptr(),
            B, C_out, D_out, H_out, W_out,
        )

        return output


# ============================================================
# Module-level constants and helpers for harness compat
# ============================================================

batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]
