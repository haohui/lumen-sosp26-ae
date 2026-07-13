import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Constants ────────────────────────────────────────────────────────
IM2COL_BLOCK_SIZE = 256
TILE_M = 32
TILE_N = 8
TILE_K = 16
GEMM_THREADS_M = TILE_M
GEMM_THREADS_N = TILE_N
GEMM_THREADS = GEMM_THREADS_M * GEMM_THREADS_N   # 256
ELEMS_PER_THREAD_A = (TILE_M * TILE_K) // GEMM_THREADS  # 2
POOL_BLOCK_SIZE = 256

# module-level constant for the GEMM kernel
_subtract_value = 0.5


# ══════════════════════════════════════════════════════════════════════
#  Kernel 1: im2col
# ══════════════════════════════════════════════════════════════════════

@avelang.jit
def im2col_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    in_c: al.i32,
    in_h: al.i32,
    in_w: al.i32,
    k_h: al.i32,
    k_w: al.i32,
    out_h: al.i32,
    out_w: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    global_id = bid * IM2COL_BLOCK_SIZE + tid

    total_m = batch_size * out_h * out_w
    k_patches = in_c * k_h * k_w

    layout_in = al.make_layout(
        (batch_size, in_c, in_h, in_w),
        (in_c * in_h * in_w, in_h * in_w, in_w, 1),
    )
    x = al.make_tensor(input_ptr, al.bf16, layout_in)

    layout_out = al.make_layout(
        (total_m, k_patches),
        (k_patches, 1),
    )
    out = al.make_tensor(output_ptr, al.bf16, layout_out)
    kh_kw = k_h * k_w

    if global_id < total_m:
        m_val = global_id

        n = m_val // (out_h * out_w)
        m_rem = m_val - n * out_h * out_w
        oh = m_rem // out_w
        ow = m_rem - oh * out_w

        for k in al.range(k_patches):
            c = k // kh_kw
            k_rem = k - c * kh_kw
            kh = k_rem // k_w
            kw = k_rem - kh * k_w

            out[m_val, k] = x[n, c, oh + kh, ow + kw]


# ══════════════════════════════════════════════════════════════════════
#  Kernel 2: tiled GEMM + bias + subtract + HardSwish
# ══════════════════════════════════════════════════════════════════════

@avelang.jit
def conv_gemm_hardswish_kernel(
    im2col_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.i32,
    n: al.i32,
    k: al.i32,
):
    tid_m = al.thread_id(0)
    tid_n = al.thread_id(1)
    block_m = al.block_id(0)
    block_n = al.block_id(1)

    row = block_m * TILE_M + tid_m
    col = block_n * TILE_N + tid_n

    if row < m and col < n:
        layout_a = al.make_layout((m, k), (k, 1))
        layout_b = al.make_layout((n, k), (k, 1))
        a = al.make_tensor(im2col_ptr, al.bf16, layout_a)
        b = al.make_tensor(weight_ptr, al.bf16, layout_b)

        shm_a = al.make_shared((TILE_M, TILE_K), al.bf16)
        shm_b = al.make_shared((TILE_K, TILE_N), al.bf16)

        linear_tid = tid_n * TILE_M + tid_m

        acc = al.convert(0.0, al.f32)

        k_tiles = k // TILE_K
        total_a = TILE_M * TILE_K
        total_b = TILE_K * TILE_N
        row_base = block_m * TILE_M
        col_base = block_n * TILE_N

        for kt in al.range(k_tiles):
            k_base = kt * TILE_K

            # cooperative load A tile
            for i in al.range(ELEMS_PER_THREAD_A):
                a_idx = linear_tid + i * GEMM_THREADS
                if a_idx < total_a:
                    a_m = a_idx // TILE_K
                    a_k = a_idx % TILE_K
                    shm_a[a_m, a_k] = a[row_base + a_m, k_base + a_k]

            # cooperative load B tile
            if linear_tid < total_b:
                b_k = linear_tid // TILE_N
                b_n = linear_tid % TILE_N
                shm_b[b_k, b_n] = b[col_base + b_n, k_base + b_k]

            al.syncthreads()

            for kk in al.range(TILE_K):
                a_val = al.convert(shm_a[tid_m, kk], al.f32)
                b_val = al.convert(shm_b[kk, tid_n], al.f32)
                acc = acc + a_val * b_val

            al.syncthreads()

        # epilogue: bias + subtract + HardSwish
        layout_bias = al.make_layout((n,), (1,))
        g_bias = al.make_tensor(bias_ptr, al.bf16, layout_bias)
        bias_val = al.convert(g_bias[col], al.f32)

        result = acc + bias_val
        sub = al.convert(_subtract_value, al.f32)
        result = result - sub

        three = al.convert(3.0, al.f32)
        six = al.convert(6.0, al.f32)
        one_over_six = al.convert(1.0 / 6.0, al.f32)
        zero = al.convert(0.0, al.f32)

        hs_in = result + three
        if hs_in < zero:
            hs_in = zero
        if hs_in > six:
            hs_in = six
        result = result * hs_in * one_over_six

        layout_out = al.make_layout((m, n), (n, 1))
        g_out = al.make_tensor(out_ptr, al.bf16, layout_out)
        g_out[row, col] = al.convert(result, al.bf16)


# ══════════════════════════════════════════════════════════════════════
#  Kernel 3: MaxPool 2x2 + Mish activation
# ══════════════════════════════════════════════════════════════════════

@avelang.jit
def maxpool_mish_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    num_ch: al.i32,
    in_h: al.i32,
    in_w: al.i32,
    out_h: al.i32,
    out_w: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    global_id = bid * POOL_BLOCK_SIZE + tid
    total_out = batch_size * num_ch * out_h * out_w

    if global_id < total_out:
        idx = global_id
        ow_out = idx % out_w
        idx = idx // out_w
        oh_out = idx % out_h
        idx = idx // out_h
        oc = idx % num_ch
        n = idx // num_ch

        layout_in = al.make_layout(
            (batch_size, in_h, in_w, num_ch),
            (in_h * in_w * num_ch, in_w * num_ch, num_ch, 1),
        )
        x = al.make_tensor(input_ptr, al.bf16, layout_in)

        ih0 = oh_out * 2
        iw0 = ow_out * 2

        max_f32 = al.convert(-1e30, al.f32)
        for dh in al.range(2):
            for dw in al.range(2):
                val = al.convert(x[n, ih0 + dh, iw0 + dw, oc], al.f32)
                if val > max_f32:
                    max_f32 = val

        one = al.convert(1.0, al.f32)
        result = max_f32 * al.tanh(al.log(one + al.exp(max_f32)))

        layout_out = al.make_layout(
            (batch_size, num_ch, out_h, out_w),
            (num_ch * out_h * out_w, out_h * out_w, out_w, 1),
        )
        out = al.make_tensor(output_ptr, al.bf16, layout_out)
        out[n, oc, oh_out, ow_out] = al.convert(result, al.bf16)


# ══════════════════════════════════════════════════════════════════════
#  Host wrappers
# ══════════════════════════════════════════════════════════════════════

def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().to(dtype=torch.bfloat16)


def avelang_conv_pipeline(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    subtract_value: float,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)
    bias_bf16 = _prepare_bf16_cuda_contiguous(bias)

    batch_val, in_c, in_h, in_w = x_bf16.shape
    out_c, w_c, k_h, k_w = weight_bf16.shape

    if w_c != in_c:
        raise ValueError(f"Weight/input C mismatch")

    oh = in_h - k_h + 1
    ow = in_w - k_w + 1
    m_patches = batch_val * oh * ow
    k_patches = in_c * k_h * k_w

    oh_out = oh // 2
    ow_out = ow // 2

    # Phase 1: im2col
    im2col_buf = torch.empty((m_patches, k_patches), dtype=torch.bfloat16, device=x_bf16.device)
    grid_im2col = ((m_patches + IM2COL_BLOCK_SIZE - 1) // IM2COL_BLOCK_SIZE, 1, 1)
    im2col_kernel[lambda: (grid_im2col, (IM2COL_BLOCK_SIZE, 1, 1))](
        x_bf16, im2col_buf, batch_val, in_c, in_h, in_w, k_h, k_w, oh, ow,
    )

    # Phase 2: fused GEMM + bias + subtract + HardSwish
    weight_flat = weight_bf16.reshape(out_c, k_patches).contiguous()

    conv_out = torch.empty((m_patches, out_c), dtype=torch.bfloat16, device=x_bf16.device)
    grid_m = (m_patches + TILE_M - 1) // TILE_M
    grid_n = (out_c + TILE_N - 1) // TILE_N
    grid_gemm = (grid_m, grid_n, 1)
    conv_gemm_hardswish_kernel[lambda: (grid_gemm, (GEMM_THREADS_M, GEMM_THREADS_N, 1))](
        im2col_buf, weight_flat, bias_bf16, conv_out,
        m_patches, out_c, k_patches,
    )

    # Phase 3: maxpool 2x2 + Mish
    out = torch.empty(
        (batch_val, out_c, oh_out, ow_out),
        dtype=torch.bfloat16, device=x_bf16.device,
    )
    total_out = batch_val * out_c * oh_out * ow_out
    grid_pool = ((total_out + POOL_BLOCK_SIZE - 1) // POOL_BLOCK_SIZE, 1, 1)
    maxpool_mish_kernel[lambda: (grid_pool, (POOL_BLOCK_SIZE, 1, 1))](
        conv_out, out, batch_val, out_c, oh, ow, oh_out, ow_out,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        return avelang_conv_pipeline(
            x, self.conv.weight, self.conv.bias, self.subtract_value,
        )


batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]
