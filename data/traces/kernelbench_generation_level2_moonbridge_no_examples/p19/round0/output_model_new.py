import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ---------------------------------------------------------------------------
# Kernel 1: Transposed 2D convolution (grouped, BF16 compute, FP32 accumulate)
# ---------------------------------------------------------------------------

@avelang.jit
def conv_transpose2d_kernel(
    in_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_in: al.i32,
    C_out: al.i32,
    H_in: al.i32,
    W_in: al.i32,
    H_out: al.i32,
    W_out: al.i32,
    in_n_stride: al.i32,
    in_c_stride: al.i32,
    in_h_stride: al.i32,
    in_w_stride: al.i32,
    w_cin_stride: al.i32,
    w_cout_stride: al.i32,
    w_h_stride: al.i32,
    w_w_stride: al.i32,
    out_n_stride: al.i32,
    out_c_stride: al.i32,
    out_h_stride: al.i32,
    out_w_stride: al.i32,
    TILE_H: al.constexpr,
    TILE_W: al.constexpr,
):
    # Build tensor views from raw pointers + runtime shapes
    in_layout = al.make_layout(
        (N, C_in, H_in, W_in),
        (in_n_stride, in_c_stride, in_h_stride, in_w_stride),
    )
    in_t = al.make_tensor(in_ptr, al.bf16, in_layout)

    w_layout = al.make_layout(
        (C_in, C_out, 3, 3),
        (w_cin_stride, w_cout_stride, w_h_stride, w_w_stride),
    )
    w_t = al.make_tensor(w_ptr, al.bf16, w_layout)

    b_layout = al.make_layout((C_out,), (1,))
    b_t = al.make_tensor(b_ptr, al.f32, b_layout)

    out_layout = al.make_layout(
        (N, C_out, H_out, W_out),
        (out_n_stride, out_c_stride, out_h_stride, out_w_stride),
    )
    out_t = al.make_tensor(out_ptr, al.bf16, out_layout)

    # Grid: (N * C_out, ceil(H_out/TILE_H), ceil(W_out/TILE_W)) — one block per (sample, out_channel, spatial tile)
    block_idx = al.block_id(0)
    h_tile = al.block_id(1)
    w_tile = al.block_id(2)

    n = block_idx // C_out
    oc = block_idx % C_out

    tid_h = al.thread_id(0)
    tid_w = al.thread_id(1)

    h = h_tile * TILE_H + tid_h
    w = w_tile * TILE_W + tid_w

    if h < H_out and w < W_out:
        # Accumulate in FP32
        acc = b_t[oc]

        for ic in al.range(C_in):
            for ky in al.range(3):
                in_h = h - ky
                if in_h >= 0 and in_h < H_in:
                    for kx in al.range(3):
                        in_w = w - kx
                        if in_w >= 0 and in_w < W_in:
                            in_val = al.convert(in_t[n, ic, in_h, in_w], al.f32)
                            w_val = al.convert(w_t[ic, oc, ky, kx], al.f32)
                            acc = acc + in_val * w_val

        out_t[n, oc, h, w] = al.convert(acc, al.bf16)


# ---------------------------------------------------------------------------
# Kernel 2: GELU activation (element-wise, BF16 in/out, FP32 compute)
# ---------------------------------------------------------------------------

@avelang.jit
def gelu_bf16_kernel(
    in_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    numel: al.i32,
    TILE: al.constexpr,
):
    idx = al.block_id(0) * TILE + al.thread_id(0)
    if idx < numel:
        layout = al.make_layout((numel,), (1,))
        in_t = al.make_tensor(in_ptr, al.bf16, layout)
        out_t = al.make_tensor(out_ptr, al.bf16, layout)

        x = al.convert(in_t[idx], al.f32)
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        half = al.convert(0.5, al.f32)
        one = al.convert(1.0, al.f32)
        rsqrt2 = al.convert(0.70710678, al.f32)  # 1/sqrt(2)

        gelu_val = half * x * (one + al.erf(x * rsqrt2))
        out_t[idx] = al.convert(gelu_val, al.bf16)


# ---------------------------------------------------------------------------
# Kernel 3: GroupNorm (training mode, BF16 in/out, FP32 compute)
# ---------------------------------------------------------------------------

@avelang.jit
def group_norm_kernel(
    in_ptr: al.Pointer(al.bf16),
    gamma_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    N: al.i32,
    C: al.i32,
    H: al.i32,
    W: al.i32,
    num_groups: al.i32,
    eps: al.constexpr,
    in_n_stride: al.i32,
    in_c_stride: al.i32,
    in_h_stride: al.i32,
    in_w_stride: al.i32,
    out_n_stride: al.i32,
    out_c_stride: al.i32,
    out_h_stride: al.i32,
    out_w_stride: al.i32,
    BLOCK_SIZE: al.constexpr,
    CHANNELS_PER_GROUP: al.constexpr,
):
    # One block per (sample, group)
    block_idx = al.block_id(0)
    n = block_idx // num_groups
    g = block_idx % num_groups
    tid = al.thread_id(0)

    # Tensor views
    in_layout = al.make_layout(
        (N, C, H, W), (in_n_stride, in_c_stride, in_h_stride, in_w_stride)
    )
    in_t = al.make_tensor(in_ptr, al.bf16, in_layout)

    gamma_layout = al.make_layout((C,), (1,))
    gamma_t = al.make_tensor(gamma_ptr, al.f32, gamma_layout)

    beta_layout = al.make_layout((C,), (1,))
    beta_t = al.make_tensor(beta_ptr, al.f32, beta_layout)

    out_layout = al.make_layout(
        (N, C, H, W), (out_n_stride, out_c_stride, out_h_stride, out_w_stride)
    )
    out_t = al.make_tensor(out_ptr, al.f32, out_layout)

    spatial_size = H * W
    c_start = g * CHANNELS_PER_GROUP

    # Shared memory: [thread, channel, 2]  —  (sum, sum_sq) per channel
    smem = al.make_shared((BLOCK_SIZE, CHANNELS_PER_GROUP, 2), al.f32)

    # Local accumulators (registers): one (sum, sum_sq) pair per channel
    local_acc = al.make_local((CHANNELS_PER_GROUP, 2), al.f32)
    for c in al.range(CHANNELS_PER_GROUP):
        local_acc[c, 0] = al.convert(0.0, al.f32)
        local_acc[c, 1] = al.convert(0.0, al.f32)

    # ---- Phase 1: per-thread accumulation over spatial positions ----
    for idx in al.range(tid, spatial_size, BLOCK_SIZE):
        h = idx // W
        w = idx % W
        for c_local in al.range(CHANNELS_PER_GROUP):
            c = c_start + c_local
            val = al.convert(in_t[n, c, h, w], al.f32)
            local_acc[c_local, 0] = local_acc[c_local, 0] + val
            local_acc[c_local, 1] = local_acc[c_local, 1] + val * val

    # Write partial results to shared memory
    for c_local in al.range(CHANNELS_PER_GROUP):
        smem[tid, c_local, 0] = local_acc[c_local, 0]
        smem[tid, c_local, 1] = local_acc[c_local, 1]
    al.syncthreads()

    # ---- Reduction: first CHANNELS_PER_GROUP threads combine across block ----
    if tid < CHANNELS_PER_GROUP:
        c_local = tid
        sum_val = al.convert(0.0, al.f32)
        sq_val = al.convert(0.0, al.f32)
        for t in al.range(BLOCK_SIZE):
            sum_val = sum_val + smem[t, c_local, 0]
            sq_val = sq_val + smem[t, c_local, 1]

        # Store per-channel partial sums for group-wide reduction
        smem[0, c_local, 0] = sum_val
        smem[0, c_local, 1] = sq_val
    al.syncthreads()

    # Thread 0 combines per-channel sums into group-wide mean and variance
    if tid == 0:
        group_sum = al.convert(0.0, al.f32)
        group_sq = al.convert(0.0, al.f32)
        for c_local in al.range(CHANNELS_PER_GROUP):
            group_sum = group_sum + smem[0, c_local, 0]
            group_sq = group_sq + smem[0, c_local, 1]

        count = al.convert(CHANNELS_PER_GROUP, al.f32) * al.convert(spatial_size, al.f32)
        group_mean = group_sum / count
        group_var = group_sq / count - group_mean * group_mean
        eps_f32 = al.convert(eps, al.f32)
        inv_std = al.convert(1.0, al.f32) / al.sqrt(group_var + eps_f32)

        # Store group mean and inv_std for phase 2 (reuse smem[0,0,:])
        smem[0, 0, 0] = group_mean
        smem[0, 0, 1] = inv_std
    al.syncthreads()

    # ---- Phase 2: normalize ----
    for idx in al.range(tid, spatial_size, BLOCK_SIZE):
        h = idx // W
        w = idx % W
        for c_local in al.range(CHANNELS_PER_GROUP):
            c = c_start + c_local
            val = al.convert(in_t[n, c, h, w], al.f32)
            group_mean = smem[0, 0, 0]
            inv_std = smem[0, 0, 1]
            norm_val = (val - group_mean) * inv_std
            result = norm_val * gamma_t[c] + beta_t[c]
            out_t[n, c, h, w] = result


# ---------------------------------------------------------------------------
# ModelNew: host wrapper that launches the three AveLang kernels
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        input_dtype = x.dtype
        x = x.contiguous()
        N, C_in, H_in, W_in = x.shape

        # --- ConvTranspose (PyTorch eager, matches reference exactly) ---
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            conv_out_autocast = self.conv_transpose(x)
        conv_out = conv_out_autocast.contiguous().to(torch.bfloat16)
        C_out = conv_out.shape[1]
        H_out = conv_out.shape[2]
        W_out = conv_out.shape[3]

        out_n_s = C_out * H_out * W_out
        out_c_s = H_out * W_out
        out_h_s = W_out
        out_w_s = 1

        # --- GELU via AveLang kernel ---
        numel = N * C_out * H_out * W_out
        TILE_GELU = 256
        grid_gelu = (numel + TILE_GELU - 1) // TILE_GELU
        gelu_out = torch.empty(N, C_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)
        gelu_bf16_kernel[lambda: ((grid_gelu, 1, 1), (TILE_GELU, 1, 1))](
            conv_out, gelu_out, numel, TILE_GELU
        )

        # --- Launch GroupNorm ---
        gamma = self.group_norm.weight.contiguous().to(torch.float32)
        beta = self.group_norm.bias.contiguous().to(torch.float32)
        eps = float(self.group_norm.eps)

        num_groups = self.group_norm.num_groups
        BLOCK_SIZE = 256
        CHANNELS_PER_GROUP = C_out // num_groups
        grid_gn = N * num_groups

        gn_out_f32 = torch.empty(N, C_out, H_out, W_out, dtype=torch.float32, device=x.device)
        group_norm_kernel[lambda: ((grid_gn, 1, 1), (BLOCK_SIZE, 1, 1))](
            gelu_out,
            gamma,
            beta,
            gn_out_f32,
            N,
            C_out,
            H_out,
            W_out,
            num_groups,
            eps,
            out_n_s,
            out_c_s,
            out_h_s,
            out_w_s,
            out_n_s,
            out_c_s,
            out_h_s,
            out_w_s,
            BLOCK_SIZE,
            CHANNELS_PER_GROUP,
        )

        # Match reference dtype
        if input_dtype == torch.float32:
            return gn_out_f32
        return gn_out_f32.to(input_dtype)
