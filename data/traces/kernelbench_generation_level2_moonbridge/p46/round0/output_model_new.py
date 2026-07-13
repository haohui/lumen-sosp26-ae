import torch
import torch.nn as nn
import avelang
import avelang.language as al

# =============================================================================
# Kernel parameters
# =============================================================================
BLOCK_SIZE = 256

# GEMM tiling
TILE_M = 64
TILE_N = 128
TILE_K = 16
THREADS_M = 16
THREADS_N = 16
ROWS_PER_THREAD = TILE_M // THREADS_M
COLS_PER_THREAD = TILE_N // THREADS_N
A_TILE_ELEMS = TILE_M * TILE_K
B_TILE_ELEMS = TILE_N * TILE_K
LOADS_A = (A_TILE_ELEMS + BLOCK_SIZE - 1) // BLOCK_SIZE
LOADS_B = (B_TILE_ELEMS + BLOCK_SIZE - 1) // BLOCK_SIZE

# Problem constants
H_IN = 128
W_IN = 128
KH = 3
KW = 3
OH = H_IN - KH + 1
OW = W_IN - KH + 1
C_IN = 64
C_OUT = 128
KHW = KH * KW

SUB1 = 0.5
SUB2 = 0.2


# =============================================================================
# Kernel 1: Fused im2col + GEMM, K-outermost, B loads hoisted
# =============================================================================
@avelang.jit
def fused_conv_gemm_kernel(
    input_ptr: al.Pointer(al.bf16),
    weight_ptr: al.Pointer(al.bf16),
    bias_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    N: al.i32,
    m: al.i32,
    n: al.i32,
    k: al.i32,
):
    tid = al.thread_id(0)
    block_m = al.block_id(0)
    tm = tid // THREADS_N
    tn = tid % THREADS_N

    in_layout = al.make_layout((N, C_IN, H_IN, W_IN), (C_IN * H_IN * W_IN, H_IN * W_IN, W_IN, 1))
    in_tensor = al.make_tensor(input_ptr, al.bf16, in_layout)

    w_layout = al.make_layout((n, k), (k, 1))
    w_tensor = al.make_tensor(weight_ptr, al.bf16, w_layout)

    g_bias = al.make_tensor(bias_ptr, al.bf16, al.make_layout((n,), (1,)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((m, n), (n, 1)))

    shm_a = al.make_shared((TILE_M, TILE_K), al.bf16)
    shm_b = al.make_shared((TILE_N, TILE_K), al.bf16)

    acc = al.make_local((ROWS_PER_THREAD * COLS_PER_THREAD,), al.f32)
    for i in al.range(ROWS_PER_THREAD * COLS_PER_THREAD):
        acc[i] = 0.0

    sub1 = al.convert(SUB1, al.f32)
    sub2 = al.convert(SUB2, al.f32)
    cols_per_th = al.convert(COLS_PER_THREAD, al.i32)
    nine = al.convert(KHW, al.i32)
    three = al.convert(KW, al.i32)
    oh_val = al.convert(OH, al.i32)
    ow_val = al.convert(OW, al.i32)
    oh_ow = al.convert(OH * OW, al.i32)

    m_base = block_m * TILE_M
    k_tiles = k // TILE_K

    for kt in al.range(k_tiles):
        k_base = kt * TILE_K

        # Cooperative load A tile: im2col on the fly from 4D input
        for i in al.range(LOADS_A):
            lid = tid + i * BLOCK_SIZE
            if lid < A_TILE_ELEMS:
                ar = lid // TILE_K
                ak = lid - ar * TILE_K

                m_idx = m_base + ar
                n_batch = m_idx // oh_ow
                m_rest = m_idx - n_batch * oh_ow
                oh = m_rest // ow_val
                ow = m_rest - oh * ow_val

                k_idx = k_base + ak
                c = k_idx // nine
                k_rest = k_idx - c * nine
                kh = k_rest // three
                kw = k_rest - kh * three

                shm_a[ar, ak] = in_tensor[n_batch, c, oh + kh, ow + kw]

        # Cooperative load B tile from weight matrix
        for i in al.range(LOADS_B):
            lid = tid + i * BLOCK_SIZE
            if lid < B_TILE_ELEMS:
                br = lid // TILE_K
                bk = lid - br * TILE_K
                shm_b[br, bk] = w_tensor[br, k_base + bk]

        al.syncthreads()

        # K-outermost inner loop: A values reused across columns
        for kk in al.range(TILE_K):
            for ri in al.range(ROWS_PER_THREAD):
                row = tm + ri * THREADS_M
                a_val = al.convert(shm_a[row, kk], al.f32)
                for ci in al.range(COLS_PER_THREAD):
                    col = tn + ci * THREADS_N
                    b_val = al.convert(shm_b[col, kk], al.f32)
                    acc_idx = ri * cols_per_th + ci
                    acc[acc_idx] = acc[acc_idx] + a_val * b_val

        al.syncthreads()

    # Writeback with epilogue
    for ri in al.range(ROWS_PER_THREAD):
        row = m_base + tm + ri * THREADS_M
        if row < m:
            for ci in al.range(COLS_PER_THREAD):
                col = tn + ci * THREADS_N
                if col < n:
                    acc_idx = ri * cols_per_th + ci
                    bias_val = al.convert(g_bias[col], al.f32)
                    result = acc[acc_idx] + bias_val
                    result = result - sub1
                    result = al.tanh(result)
                    result = result - sub2
                    g_out[row, col] = al.convert(result, al.bf16)


# =============================================================================
# Kernel 2: Average pooling 2x2
# =============================================================================
POOL_CHUNK = 8

@avelang.jit
def avgpool2d_2x2_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    N: al.i32,
    C_out: al.i32,
    OH_in: al.i32,
    OW_in: al.i32,
    OH_out: al.i32,
    OW_out: al.i32,
    input_M: al.i32,
    total_output: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)
    base = bid * BLOCK_SIZE * POOL_CHUNK + tid * POOL_CHUNK

    half = al.convert(0.25, al.f32)
    two = al.convert(2, al.i32)

    stride_n = OH_in * OW_in
    stride_m = C_out
    in_flat = input_M * C_out

    in_layout = al.make_layout((in_flat,), (1,))
    in_tensor = al.make_tensor(input_ptr, al.bf16, in_layout)

    out_layout = al.make_layout((N, C_out, OH_out, OW_out), (C_out * OH_out * OW_out, OH_out * OW_out, OW_out, 1))
    out_tensor = al.make_tensor(output_ptr, al.bf16, out_layout)

    for i in al.range(POOL_CHUNK):
        idx = base + i
        if idx < total_output:
            pw = idx % OW_out
            rest1 = idx // OW_out
            ph = rest1 % OH_out
            rest2 = rest1 // OH_out
            c = rest2 % C_out
            n = rest2 // C_out

            ih0 = ph * two
            ih1 = ih0 + 1
            iw0 = pw * two
            iw1 = iw0 + 1

            base_n = n * stride_n
            v00 = al.convert(in_tensor[(base_n + ih0 * OW_in + iw0) * stride_m + c], al.f32)
            v01 = al.convert(in_tensor[(base_n + ih0 * OW_in + iw1) * stride_m + c], al.f32)
            v10 = al.convert(in_tensor[(base_n + ih1 * OW_in + iw0) * stride_m + c], al.f32)
            v11 = al.convert(in_tensor[(base_n + ih1 * OW_in + iw1) * stride_m + c], al.f32)

            pool_sum = v00 + v01 + v10 + v11
            pool_avg = pool_sum * half
            out_tensor[n, c, ph, pw] = al.convert(pool_avg, al.bf16)


# =============================================================================
# Host helpers
# =============================================================================
def _prepare_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


# =============================================================================
# ModelNew
# =============================================================================
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N_val, C, H, W = x.shape
        C_out = self.out_channels
        KH_val = self.kernel_size
        KW_val = self.kernel_size
        OH_val = H - KH_val + 1
        OW_val = W - KW_val + 1

        M_im2col = N_val * OH_val * OW_val
        K_im2col = C * KH_val * KW_val

        x_bf16 = _prepare_bf16_contiguous(x)
        weight_2d = self.weight.data.reshape(C_out, K_im2col)
        w_bf16 = _prepare_bf16_contiguous(weight_2d)
        b_bf16 = _prepare_bf16_contiguous(self.bias.data)

        # Step 1: fused im2col + GEMM
        gemm_out = torch.empty((M_im2col, C_out), device=x_bf16.device, dtype=torch.bfloat16)
        grid_m = (M_im2col + TILE_M - 1) // TILE_M
        fused_conv_gemm_kernel[lambda: ((grid_m, 1, 1), (THREADS_M * THREADS_N, 1, 1))](
            x_bf16, w_bf16, b_bf16, gemm_out,
            N_val, M_im2col, C_out, K_im2col,
        )

        # Step 2: avgpool 2x2
        pool_size = self.kernel_size_pool
        OH_pool = OH_val // pool_size
        OW_pool = OW_val // pool_size
        total_pool_output = N_val * C_out * OH_pool * OW_pool

        pool_out = torch.empty((N_val, C_out, OH_pool, OW_pool), device=x_bf16.device, dtype=torch.bfloat16)
        elems_per_block_pool = BLOCK_SIZE * POOL_CHUNK
        num_blocks_pool = (total_pool_output + elems_per_block_pool - 1) // elems_per_block_pool
        avgpool2d_2x2_kernel[lambda: ((num_blocks_pool, 1, 1), (BLOCK_SIZE, 1, 1))](
            gemm_out, pool_out, N_val, C_out, OH_val, OW_val, OH_pool, OW_pool, M_im2col, total_pool_output,
        )

        return pool_out


# =============================================================================
# Preserve original module-level API
# =============================================================================
batch_size = 128
in_channels = 64
out_channels = 128
height = 128
width = 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
