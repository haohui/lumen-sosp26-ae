import torch
import avelang
import avelang.language as al

# ── Problem constants ────────────────────────────────────────────────
BATCH = 256
IC = 3
IH = 224
IW = 224
OC = 96
KH = 11
KW = 11
STRIDE = 4
PAD = 2
OH = (IH + 2 * PAD - KH) // STRIDE + 1  # 55
OW = (IW + 2 * PAD - KW) // STRIDE + 1  # 55
M = BATCH * OH * OW  # 774400
K_ORIG = IC * KH * KW  # 363
K_PAD = ((K_ORIG + 15) // 16) * 16  # 368
N_ORIG = OC  # 96
N_PAD = ((N_ORIG + 127) // 128) * 128  # 128

# ── GEMM tile parameters ─────────────────────────────────────────────
TILE_M = 128
TILE_N = 128
TILE_K = 16
THREADS_M = 16
THREADS_N = 16
THREADS = THREADS_M * THREADS_N  # 256
BF16_BYTES = 2

# ── im2col coalesced write parameters ────────────────────────────────
IM2COL_ROWS_PER_BLOCK = 8
IM2COL_THREADS_PER_ROW = 32


# ═══════════════════════════════════════════════════════════════════════
# Kernel 1: im2col with coalesced writes
# ═══════════════════════════════════════════════════════════════════════
@avelang.jit
def im2col_bf16_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    batch: al.i32,
    ic_dim: al.i32,
    ih_dim: al.i32,
    iw_dim: al.i32,
    oh_dim: al.i32,
    ow_dim: al.i32,
    kh_dim: al.i32,
    kw_dim: al.i32,
    stride_h: al.i32,
    stride_w: al.i32,
    pad_h: al.i32,
    pad_w: al.i32,
    k_orig: al.i32,
    k_pad: al.i32,
    total_rows: al.i32,
):
    tid = al.thread_id(0)
    block_base_row = al.block_id(0) * 8
    local_row = tid // 32
    local_col = tid % 32

    row = block_base_row + local_row
    if row >= total_rows:
        return

    oh_ow_total = oh_dim * ow_dim
    n_idx = row // oh_ow_total
    residual = row - n_idx * oh_ow_total
    oh_idx = residual // ow_dim
    ow_idx = residual - oh_idx * ow_dim

    input = al.make_tensor(
        input_ptr, al.bf16,
        al.make_layout((batch, ic_dim, ih_dim, iw_dim),
                       (ic_dim * ih_dim * iw_dim, ih_dim * iw_dim, iw_dim, 1)),
    )
    output = al.make_tensor(
        output_ptr, al.bf16,
        al.make_layout((total_rows, k_pad), (k_pad, 1)),
    )

    kh_kw_total = kh_dim * kw_dim
    zero = al.convert(0.0, al.bf16)

    for col in al.range(local_col, k_pad, 32):
        if col >= k_orig:
            output[row, col] = zero
        else:
            ic_idx = col // kh_kw_total
            residual2 = col - ic_idx * kh_kw_total
            kh_idx = residual2 // kw_dim
            kw_idx = residual2 - kh_idx * kw_dim

            ih_idx = oh_idx * stride_h + kh_idx - pad_h
            iw_idx = ow_idx * stride_w + kw_idx - pad_w

            in_bounds = 1
            if ih_idx < 0:
                in_bounds = 0
            if ih_idx >= ih_dim:
                in_bounds = 0
            if iw_idx < 0:
                in_bounds = 0
            if iw_idx >= iw_dim:
                in_bounds = 0

            if in_bounds != 0:
                output[row, col] = input[n_idx, ic_idx, ih_idx, iw_idx]
            else:
                output[row, col] = zero


# ═══════════════════════════════════════════════════════════════════════
# Kernel 2: Shared-memory tiled GEMM — 128x128 tile, A-reuse
# ═══════════════════════════════════════════════════════════════════════
@avelang.jit
def gemm_tiled_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m_dim: al.u32,
    n_dim: al.u32,
    k_dim: al.u32,
    n_orig: al.u32,
):
    tid = al.thread_id(0)
    local_m = tid % 16
    local_n = tid // 16
    block_m = al.block_id(1)
    block_n = al.block_id(0)

    a_global = al.make_tensor(a_ptr, al.bf16, al.make_layout((m_dim, k_dim), (k_dim, 1)))
    b_global = al.make_tensor(b_ptr, al.bf16, al.make_layout((n_dim, k_dim), (k_dim, 1)))
    g_bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n_dim,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m_dim, n_dim), (n_dim, 1)))

    a_rsrc = al.amdgpu.make_rsrc(a_global, m_dim * k_dim * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_global, n_dim * k_dim * BF16_BYTES)

    shm_a = al.make_shared((128, 16), al.bf16)
    shm_b = al.make_shared((128, 16), al.bf16)

    zero_u32 = al.convert(0, al.u32)
    zero_f32 = al.convert(0.0, al.f32)

    acc = al.make_local((8, 8), al.f32)
    for i in al.range(8):
        for j in al.range(8):
            acc[i, j] = 0

    a_vals = al.make_local((8,), al.f32)

    k_tiles = k_dim // 16
    for kt in al.range(k_tiles):
        k_base = kt * 16

        shm_a_u32 = al.view(shm_a, al.Tensor((256, 4), al.u32))
        for idx in al.range(tid, 256, 256):
            row = idx // 2
            col_vec = idx % 2
            off = ((block_m * 128 + row) * k_dim + k_base + col_vec * 8) * BF16_BYTES
            shm_a_u32[idx] = al.amdgpu.raw_buffer_load_x4(a_rsrc, zero_u32, off, 0)

        shm_b_u32 = al.view(shm_b, al.Tensor((256, 4), al.u32))
        for idx in al.range(tid, 256, 256):
            row = idx // 2
            col_vec = idx % 2
            off = ((block_n * 128 + row) * k_dim + k_base + col_vec * 8) * BF16_BYTES
            shm_b_u32[idx] = al.amdgpu.raw_buffer_load_x4(b_rsrc, zero_u32, off, 0)

        al.syncthreads()

        for kk in al.range(16):
            for mm in al.range(8):
                a_vals[mm] = al.convert(shm_a[mm * 16 + local_m, kk], al.f32)
            for nn in al.range(8):
                b_val = al.convert(shm_b[nn * 16 + local_n, kk], al.f32)
                for mm in al.range(8):
                    acc[mm, nn] = acc[mm, nn] + a_vals[mm] * b_val

        al.syncthreads()

    for mm in al.range(8):
        row = block_m * 128 + mm * 16 + local_m
        for nn in al.range(8):
            col = block_n * 128 + nn * 16 + local_n
            if row < m_dim and col < n_orig:
                result = acc[mm, nn]
                bias_val = al.convert(g_bias[col], al.f32)
                result = result + bias_val
                g_out[row, col] = al.convert(result, al.bf16)


# ═══════════════════════════════════════════════════════════════════════
# Host helpers
# ═══════════════════════════════════════════════════════════════════════
def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    batch, ic, ih, iw = x_bf16.shape
    oc, w_ic, kh, kw = weight_bf16.shape
    if ic != w_ic:
        raise ValueError(f"Channel mismatch: input IC={ic}, weight IC={w_ic}")

    oh = (ih + 2 * PAD - kh) // STRIDE + 1
    ow = (iw + 2 * PAD - kw) // STRIDE + 1
    m_rows = batch * oh * ow

    k_orig_val = ic * kh * kw
    k_pad_val = ((k_orig_val + 15) // 16) * 16
    n_pad_val = ((oc + 127) // 128) * 128

    # Step 1: im2col with coalesced writes
    im2col_buf = torch.empty(
        (m_rows, k_pad_val), device=x_bf16.device, dtype=torch.bfloat16
    )
    im2col_grid = ((m_rows + 8 - 1) // 8, 1, 1)
    im2col_bf16_kernel[lambda: (im2col_grid, (256, 1, 1))](
        x_bf16, im2col_buf,
        batch, ic, ih, iw,
        oh, ow, kh, kw,
        STRIDE, STRIDE, PAD, PAD,
        k_orig_val, k_pad_val, m_rows,
    )

    # Step 2: Pad weight and bias
    weight_2d = weight_bf16.reshape(oc, k_orig_val)
    weight_padded = torch.zeros(
        (n_pad_val, k_pad_val), device=weight_bf16.device, dtype=torch.bfloat16
    )
    weight_padded[:oc, :k_orig_val] = weight_2d

    bias_padded = torch.zeros(
        (n_pad_val,), device=bias_bf16.device, dtype=torch.bfloat16
    )
    bias_padded[:oc] = bias_bf16

    # Step 3: GEMM
    out_2d = torch.empty(
        (m_rows, n_pad_val), device=x_bf16.device, dtype=torch.bfloat16
    )
    gemm_grid = (
        (n_pad_val + 128 - 1) // 128,
        (m_rows + 128 - 1) // 128,
        1,
    )
    gemm_tiled_kernel[lambda: (gemm_grid, (THREADS, 1, 1))](
        im2col_buf, weight_padded, bias_padded, out_2d,
        m_rows, n_pad_val, k_pad_val, oc,
    )

    # Step 4: Reshape to NCHW
    out = out_2d[:, :oc].reshape(batch, oh, ow, oc).permute(0, 3, 1, 2).contiguous()
    return out


class ModelNew(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=IC, out_channels=OC,
            kernel_size=KH, stride=STRIDE, padding=PAD,
        )

    def forward(self, x):
        return avelang_conv2d(x, self.conv1.weight, self.conv1.bias)


def get_inputs():
    return [torch.rand(BATCH, IC, IH, IW)]


def get_init_inputs():
    return [1000]
