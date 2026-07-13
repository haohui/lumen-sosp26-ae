import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


# =============================================================================
# Kernel 1: Conv3D + HardSwish
# =============================================================================
@avelang.jit
def conv3d_hardswish_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    D_in: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    K: al.i32,
    D_out: al.i32,
    H_out: al.i32,
    W_out: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    tid_i32 = al.convert(tid, al.i32)

    b_idx = bid // C_out
    oc = bid - b_idx * C_out

    if b_idx < B:
        C_in_total = C_in * D_in * H_in * W_in
        C_out_total = C_out * D_out * H_out * W_out
        K3 = K * K * K

        x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((B * C_in_total,), (1,)))
        w_flat = al.make_tensor(w_ptr, al.bf16, al.make_layout((C_out * C_in * K3,), (1,)))
        b_flat = al.make_tensor(b_ptr, al.bf16, al.make_layout((C_out,), (1,)))
        out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((B * C_out_total,), (1,)))

        bias_val = al.convert(b_flat[oc], al.f32)
        spatial_size = D_out * H_out * W_out
        HW = H_out * W_out

        for pos in al.range(tid_i32, spatial_size, BLOCK_SIZE):
            d = pos // HW
            rem = pos - d * HW
            h = rem // W_out
            w = rem - h * W_out

            out_flat_idx = b_idx * C_out_total + oc * spatial_size + pos

            result = bias_val
            for ic in al.range(C_in):
                ic_off = ic * (D_in * H_in * W_in)
                for kd in al.range(K):
                    d_in = d + kd
                    d_off = d_in * (H_in * W_in)
                    for kh in al.range(K):
                        h_in = h + kh
                        h_off = h_in * W_in
                        for kw in al.range(K):
                            w_in = w + kw
                            in_flat_idx = b_idx * C_in_total + ic_off + d_off + h_off + w_in
                            in_val = al.convert(x_flat[in_flat_idx], al.f32)

                            w_flat_idx = oc * (C_in * K3) + ic * K3 + kd * (K * K) + kh * K + kw
                            w_val = al.convert(w_flat[w_flat_idx], al.f32)

                            result = result + in_val * w_val

            # HardSwish: x * ReLU6(x + 3) / 6
            three = al.convert(3.0, al.f32)
            six = al.convert(6.0, al.f32)
            zero = al.convert(0.0, al.f32)

            tmp = result + three
            if tmp < zero:
                tmp = zero
            if tmp > six:
                tmp = six
            result = result * tmp / six

            out_flat[out_flat_idx] = al.convert(result, al.bf16)


# =============================================================================
# Kernel 2: GroupNorm (single-pass: reduce + apply)
# =============================================================================
@avelang.jit
def groupnorm_kernel(
    x_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    num_groups: al.i32,
    C_per_group: al.i32,
    eps: al.f32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    tid_i32 = al.convert(tid, al.i32)

    b_idx = bid // num_groups
    group = bid - b_idx * num_groups

    if b_idx < B:
        N = C_per_group * D * H * W
        HW = H * W

        C_total = C * D * H * W
        x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((B * C_total,), (1,)))

        gn_w = al.make_tensor(weight_ptr, al.bf16, al.make_layout((C,), (1,)))
        gn_b = al.make_tensor(bias_ptr, al.bf16, al.make_layout((C,), (1,)))

        out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((B * C_total,), (1,)))

        smem_sum = al.make_shared((BLOCK_SIZE,), al.f32)
        smem_sq = al.make_shared((BLOCK_SIZE,), al.f32)

        # Phase 1: compute partial sums for mean and variance
        local_sum = al.convert(0.0, al.f32)
        local_sq = al.convert(0.0, al.f32)

        for i in al.range(tid_i32, N, BLOCK_SIZE):
            cg = i // (D * HW)
            spatial = i - cg * (D * HW)
            d = spatial // HW
            rem2 = spatial - d * HW
            h = rem2 // W
            w = rem2 - h * W
            c = group * C_per_group + cg

            flat_idx = b_idx * C_total + c * (D * HW) + d * HW + h * W + w
            val = al.convert(x_flat[flat_idx], al.f32)
            local_sum = local_sum + val
            local_sq = local_sq + val * val

        smem_sum[tid] = local_sum
        smem_sq[tid] = local_sq
        al.syncthreads()

        # Reduction tree
        if tid < 128:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 128]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 128]
        al.syncthreads()
        if tid < 64:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 64]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 64]
        al.syncthreads()
        if tid < 32:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 32]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 32]
        al.syncthreads()
        if tid < 16:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 16]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 16]
        al.syncthreads()
        if tid < 8:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 8]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 8]
        al.syncthreads()
        if tid < 4:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 4]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 4]
        al.syncthreads()
        if tid < 2:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 2]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 2]
        al.syncthreads()
        if tid < 1:
            smem_sum[tid] = smem_sum[tid] + smem_sum[tid + 1]
            smem_sq[tid] = smem_sq[tid] + smem_sq[tid + 1]

        # Compute mean and rstd, store in shared memory
        if tid == 0:
            N_f32 = al.convert(N, al.f32)
            mean = smem_sum[0] / N_f32
            var = smem_sq[0] / N_f32 - mean * mean
            rstd = al.convert(1.0, al.f32) / al.sqrt(var + eps)
            smem_sum[0] = mean
            smem_sq[0] = rstd

        al.syncthreads()

        mean = smem_sum[0]
        rstd = smem_sq[0]

        # Phase 2: apply normalization and affine transform
        for i in al.range(tid_i32, N, BLOCK_SIZE):
            cg = i // (D * HW)
            spatial = i - cg * (D * HW)
            d = spatial // HW
            rem2 = spatial - d * HW
            h = rem2 // W
            w = rem2 - h * W
            c = group * C_per_group + cg

            flat_idx = b_idx * C_total + c * (D * HW) + d * HW + h * W + w
            val = al.convert(x_flat[flat_idx], al.f32)
            w_val = al.convert(gn_w[c], al.f32)
            b_val = al.convert(gn_b[c], al.f32)

            normalized = (val - mean) * rstd
            result = normalized * w_val + b_val

            out_flat[flat_idx] = al.convert(result, al.bf16)


# =============================================================================
# Kernel 3: Spatial mean reduction (collapse D, H, W dims)
# =============================================================================
@avelang.jit
def spatial_mean_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.f32),
    B: al.i32,
    C: al.i32,
    D: al.i32,
    H: al.i32,
    W: al.i32,
    spatial_size: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    tid_i32 = al.convert(tid, al.i32)

    b_idx = bid // C
    c = bid - b_idx * C

    if b_idx < B:
        HW = H * W

        C_total = C * D * H * W
        x_flat = al.make_tensor(x_ptr, al.bf16, al.make_layout((B * C_total,), (1,)))

        out_flat = al.make_tensor(out_ptr, al.f32, al.make_layout((B * C,), (1,)))

        smem = al.make_shared((BLOCK_SIZE,), al.f32)

        local_sum = al.convert(0.0, al.f32)

        for i in al.range(tid_i32, spatial_size, BLOCK_SIZE):
            d = i // HW
            rem = i - d * HW
            h = rem // W
            w = rem - h * W

            flat_idx = b_idx * C_total + c * (D * HW) + d * HW + h * W + w
            val = al.convert(x_flat[flat_idx], al.f32)
            local_sum = local_sum + val

        smem[tid] = local_sum
        al.syncthreads()

        # Reduction tree
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
            N_f32 = al.convert(spatial_size, al.f32)
            out_flat_idx = b_idx * C + c
            out_flat[out_flat_idx] = smem[0] / N_f32


# =============================================================================
# Host wrapper
# =============================================================================
def avelang_pipeline(
    x: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor,
    gn_w: torch.Tensor,
    gn_b: torch.Tensor,
) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA/HIP device."

    B = x.shape[0]
    C_in = x.shape[1]
    D_in = x.shape[2]
    H_in = x.shape[3]
    W_in = x.shape[4]

    C_out = conv_w.shape[0]
    K = conv_w.shape[2]

    D_out = D_in - K + 1
    H_out = H_in - K + 1
    W_out = W_in - K + 1

    num_groups = 4
    C_per_group = C_out // num_groups

    x_bf16 = x.contiguous().to(torch.bfloat16)
    conv_w_bf16 = conv_w.contiguous().to(torch.bfloat16)
    conv_b_bf16 = conv_b.contiguous().to(torch.bfloat16)
    gn_w_bf16 = gn_w.contiguous().to(torch.bfloat16)
    gn_b_bf16 = gn_b.contiguous().to(torch.bfloat16)

    conv_out = torch.empty(
        (B, C_out, D_out, H_out, W_out),
        dtype=torch.bfloat16,
        device=x.device,
    )

    gn_out = torch.empty(
        (B, C_out, D_out, H_out, W_out),
        dtype=torch.bfloat16,
        device=x.device,
    )

    final_out = torch.empty((B, C_out), dtype=torch.float32, device=x.device)

    conv_grid = (B * C_out, 1, 1)
    conv3d_hardswish_kernel[lambda: (conv_grid, (BLOCK_SIZE, 1, 1))](
        x_bf16,
        conv_w_bf16,
        conv_b_bf16,
        conv_out,
        B,
        C_in,
        C_out,
        D_in,
        H_in,
        W_in,
        K,
        D_out,
        H_out,
        W_out,
    )

    gn_grid = (B * num_groups, 1, 1)
    eps = 1e-5
    groupnorm_kernel[lambda: (gn_grid, (BLOCK_SIZE, 1, 1))](
        conv_out,
        gn_w_bf16,
        gn_b_bf16,
        gn_out,
        B,
        C_out,
        D_out,
        H_out,
        W_out,
        num_groups,
        C_per_group,
        eps,
    )

    spatial_size = D_out * H_out * W_out
    mean_grid = (B * C_out, 1, 1)
    spatial_mean_kernel[lambda: (mean_grid, (BLOCK_SIZE, 1, 1))](
        gn_out,
        final_out,
        B,
        C_out,
        D_out,
        H_out,
        W_out,
        spatial_size,
    )

    return final_out


# =============================================================================
# ModelNew
# =============================================================================
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        conv_w = self.conv.weight.data
        conv_b = self.conv.bias.data
        gn_w = self.group_norm.weight.data
        gn_b = self.group_norm.bias.data

        result = avelang_pipeline(x, conv_w, conv_b, gn_w, gn_b)
        return result.to(x.dtype)


# =============================================================================
# Test config (mirrors input_model.py)
# =============================================================================
batch_size = 1024
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 4


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
