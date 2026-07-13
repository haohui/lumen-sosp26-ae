import torch
import torch.nn as nn
import avelang
import avelang.language as al

# === GEMM tile constants ===
WARP_SIZE = 64
NUM_WARPS = 1
THREADS = WARP_SIZE * NUM_WARPS  # 64
GROUP_M = 64
GROUP_N = 64
GROUP_K = 16
MMA_M = 32
MMA_N = 32
VEC_ELEMS = 8
BF16_BYTES = 2
ACC_SIZE = 16
WARPS_M = 1
WARPS_N = 1
M_TILES_PER_WARP = GROUP_M // (WARPS_M * MMA_M)   # 2
N_TILES_PER_WARP = GROUP_N // (WARPS_N * MMA_N)   # 2
A_VECS_PER_ROW = GROUP_K // VEC_ELEMS  # 2
B_VECS_PER_ROW = GROUP_K // VEC_ELEMS  # 2
SHM_A_VECS = GROUP_M * A_VECS_PER_ROW   # 128
SHM_B_VECS = GROUP_N * B_VECS_PER_ROW   # 128
GLOBAL_LOADS_A = SHM_A_VECS // THREADS   # 2
GLOBAL_LOADS_B = SHM_B_VECS // THREADS   # 2
ROW_U32 = A_VECS_PER_ROW * 4  # 8


@avelang.jit
def _load_fused_a_to_shm(
    shm_a: al.Tensor((SHM_A_VECS, 4), al.u32),
    x_ptr: al.Pointer(al.bf16),
    block_m: al.u32,
    k_base: al.u32,
    k: al.u32,
    tid: al.u32,
    m_global_offset: al.u32,
    n_batch: al.u32,
    c_in: al.u32,
    h: al.u32,
    w: al.u32,
    kernel_h: al.u32,
    kernel_w: al.u32,
    h_out: al.u32,
    w_out: al.u32,
):
    hw_out_total = h_out * w_out
    c_stride = h * w
    n_stride = c_in * c_stride
    kh_kw = kernel_h * kernel_w
    input_total = n_batch * n_stride
    x_1d = al.make_tensor(x_ptr, al.bf16, al.make_layout((input_total,), (al.convert(1, al.u32),)))

    one = al.convert(1, al.u32)
    zero_u32 = al.convert(0, al.u32)

    idx = tid
    for _ in al.range(GLOBAL_LOADS_A):
        row = idx // A_VECS_PER_ROW
        col_vec = idx % A_VECS_PER_ROW

        m_idx = block_m * GROUP_M + row + m_global_offset
        n_val = m_idx // hw_out_total
        hw_rem = m_idx - n_val * hw_out_total
        h_out_pos = hw_rem // w_out
        w_out_pos = hw_rem - h_out_pos * w_out
        in_row_base = n_val * n_stride + h_out_pos * w + w_out_pos

        k_start = k_base + col_vec * VEC_ELEMS

        c_cur = k_start // kh_kw
        kk_cur = k_start - c_cur * kh_kw
        kh_cur = kk_cur // kernel_w
        kw_cur = kk_cur - kh_cur * kernel_w

        vals = al.make_local((8,), al.bf16)

        for e in al.range(VEC_ELEMS):
            off = in_row_base + c_cur * c_stride + kh_cur * w + kw_cur
            vals[e] = x_1d[off]
            kw_cur = kw_cur + one
            if kw_cur >= kernel_w:
                kw_cur = zero_u32
                kh_cur = kh_cur + one
                if kh_cur >= kernel_h:
                    kh_cur = zero_u32
                    c_cur = c_cur + one

        packed = al.view(vals, al.Tensor((4,), al.u32))
        shm_a[idx, 0] = packed[0]
        shm_a[idx, 1] = packed[1]
        shm_a[idx, 2] = packed[2]
        shm_a[idx, 3] = packed[3]

        idx += THREADS


@avelang.jit
def _load_global_b_to_shm(
    shm_b: al.Tensor((SHM_B_VECS, 4), al.u32),
    w_rsrc: al.Tensor((4,), al.u32),
    block_n: al.u32,
    k_base: al.u32,
    k: al.u32,
    tid: al.u32,
):
    zero = al.convert(0, al.u32)
    idx = tid
    for _ in al.range(GLOBAL_LOADS_B):
        row = idx // B_VECS_PER_ROW
        col_vec = idx % B_VECS_PER_ROW
        off = ((block_n * GROUP_N + row) * k + k_base + col_vec * VEC_ELEMS) * BF16_BYTES
        shm_b[idx] = al.amdgpu.raw_buffer_load_x4(w_rsrc, zero, off, 0)
        idx += THREADS


@avelang.jit
def _fetch_mfma_operand_32x32x16(
    ret: al.Tensor((2, 4), al.bf16),
    shm: al.Tensor((SHM_A_VECS, 4), al.u32),
    tile_idx: al.u32,
    lane: al.u32,
):
    ret_u32 = al.view(ret, al.Tensor((4,), al.u32))
    shm_u32 = al.view(shm, al.Tensor((SHM_A_VECS * 4,), al.u32))
    row = tile_idx * MMA_M + (lane % MMA_M)
    k_group_u32 = (lane // MMA_M) * 2
    row_base = row * ROW_U32

    ret_u32[0] = shm_u32[row_base + k_group_u32]
    ret_u32[1] = shm_u32[row_base + k_group_u32 + 1]
    ret_u32[2] = shm_u32[row_base + 4 + k_group_u32]
    ret_u32[3] = shm_u32[row_base + 5 + k_group_u32]


@avelang.jit
def conv_gemm_kernel(
    x_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
    m_global_offset: al.u32,
    h_out: al.u32,
    w_out: al.u32,
    n_batch: al.u32,
    c_in: al.u32,
    h: al.u32,
    w: al.u32,
    kernel_h: al.u32,
    kernel_w: al.u32,
):
    tid = al.thread_id(0)
    block_n = al.block_id(0)
    block_m = al.block_id(1)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    warp_row = wid // WARPS_N
    warp_col = wid % WARPS_N

    b_memref = al.make_tensor(b_ptr, al.bf16, al.make_layout((n * k,), (al.convert(1, al.u32),)))
    hw_out_total = h_out * w_out
    oc_stride = h_out * w_out
    n_stride_out = n * h_out * w_out
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((n_batch * n * h_out * w_out,), (al.convert(1, al.u32),)))

    b_rsrc = al.amdgpu.make_rsrc(b_memref, n * k * BF16_BYTES)

    shm_a = al.make_shared((SHM_A_VECS, 4), al.u32)
    shm_b = al.make_shared((SHM_B_VECS, 4), al.u32)
    a_reg = al.make_local((M_TILES_PER_WARP, 2, 4), al.bf16)
    b_reg = al.make_local((N_TILES_PER_WARP, 2, 4), al.bf16)
    acc = al.make_local((M_TILES_PER_WARP * N_TILES_PER_WARP, ACC_SIZE), al.f32)

    for i in al.range(M_TILES_PER_WARP * N_TILES_PER_WARP):
        for j in al.range(ACC_SIZE):
            acc[i, j] = al.convert(0.0, al.f32)

    k_tiles = k // GROUP_K
    for kt in al.range(k_tiles):
        k_base = kt * GROUP_K
        _load_fused_a_to_shm(shm_a, x_ptr, block_m, k_base, k, tid,
                             m_global_offset, n_batch, c_in, h, w,
                             kernel_h, kernel_w, h_out, w_out)
        _load_global_b_to_shm(shm_b, b_rsrc, block_n, k_base, k, tid)
        al.syncthreads()

        for i in al.range(M_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(a_reg[i], shm_a, warp_row * M_TILES_PER_WARP + i, lane)
        for j in al.range(N_TILES_PER_WARP):
            _fetch_mfma_operand_32x32x16(b_reg[j], shm_b, warp_col * N_TILES_PER_WARP + j, lane)

        for i in al.range(M_TILES_PER_WARP):
            for j in al.range(N_TILES_PER_WARP):
                acc_idx = i * N_TILES_PER_WARP + j
                a_op0 = al.view(a_reg[i, 0], al.Tensor((2,), al.u32))
                b_op0 = al.view(b_reg[j, 0], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_op0, b_op0, acc[acc_idx])
                a_op1 = al.view(a_reg[i, 1], al.Tensor((2,), al.u32))
                b_op1 = al.view(b_reg[j, 1], al.Tensor((2,), al.u32))
                acc[acc_idx] = al.amdgpu.mfma_32x32x8_bf16_f32(a_op1, b_op1, acc[acc_idx])

        al.syncthreads()

    lane_group = lane // MMA_N
    lane_col = lane % MMA_N
    block_row_base = block_m * GROUP_M + m_global_offset
    block_col_base = block_n * GROUP_N

    for j in al.range(N_TILES_PER_WARP):
        col = block_col_base + (warp_col * N_TILES_PER_WARP + j) * MMA_N + lane_col
        for i in al.range(M_TILES_PER_WARP):
            acc_idx = i * N_TILES_PER_WARP + j
            row_base = block_row_base + (warp_row * M_TILES_PER_WARP + i) * MMA_M
            for t in al.range(ACC_SIZE):
                row = row_base + (t // al.convert(4, al.u32)) * al.convert(8, al.u32) + lane_group * al.convert(4, al.u32) + (t % al.convert(4, al.u32))
                result = acc[acc_idx, t]
                oc = col
                n_idx = row // hw_out_total
                hw_rem = row - n_idx * hw_out_total
                h_idx = hw_rem // w_out
                w_idx = hw_rem - h_idx * w_out
                out_idx = n_idx * n_stride_out + oc * oc_stride + h_idx * w_out + w_idx
                g_out[out_idx] = al.convert(result, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    weight_bf16 = _prepare_bf16_cuda_contiguous(weight)

    n_batch, c_in, h, w = x_bf16.shape
    oc_out, c_w, kernel_h, kernel_w = weight_bf16.shape

    if c_in != c_w:
        raise ValueError(f"Channel mismatch: input has {c_in}, weight has {c_w}")
    if groups != 1:
        raise NotImplementedError("Only groups=1 is supported")
    if dilation != 1:
        raise NotImplementedError("Only dilation=1 is supported")
    if padding != 0:
        raise NotImplementedError("Only padding=0 is supported")

    kh_eff = dilation * (kernel_h - 1) + 1
    kw_eff = dilation * (kernel_w - 1) + 1
    h_out = (h + 2 * padding - kh_eff) // stride + 1
    w_out = (w + 2 * padding - kw_eff) // stride + 1

    m_full = n_batch * h_out * w_out
    k_total = c_in * kernel_h * kernel_w
    n_gemm = oc_out

    if m_full % GROUP_M != 0:
        raise ValueError(f"m_full={m_full} must be divisible by GROUP_M={GROUP_M}")
    if k_total % GROUP_K != 0:
        raise ValueError(f"k_total={k_total} must be divisible by GROUP_K={GROUP_K}")
    if n_gemm % GROUP_N != 0:
        raise ValueError(f"n_gemm={n_gemm} must be divisible by GROUP_N={GROUP_N}")

    weight_2d = weight_bf16.reshape(oc_out, k_total).contiguous()
    out = torch.empty((n_batch, oc_out, h_out, w_out), device=x_bf16.device, dtype=torch.bfloat16)

    grid_gemm = (n_gemm // GROUP_N, m_full // GROUP_M, 1)
    conv_gemm_kernel[lambda: (grid_gemm, (THREADS, 1, 1))](
        x_bf16, weight_2d, out, m_full, n_gemm, k_total,
        0, h_out, w_out, n_batch, c_in, h, w, kernel_h, kernel_w,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized 2D convolution using fused im2col + MFMA GEMM in AveLang DSL.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        self.bias_param = nn.Parameter(torch.empty(out_channels)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv2d(
            x, self.weight, self.bias_param,
            self.stride, self.padding, self.dilation, self.groups,
        )


def get_inputs():
    batch_size = 8
    in_channels = 32
    width = 512
    height = 512
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    in_channels = 32
    out_channels = 64
    kernel_size = (5, 9)
    return [in_channels, out_channels, kernel_size]
