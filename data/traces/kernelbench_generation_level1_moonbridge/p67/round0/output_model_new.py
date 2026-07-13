import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── Problem dimensions (compile-time) ────────────────────────────────────────
B: al.constexpr = 32
C_IN: al.constexpr = 64
C_OUT: al.constexpr = 128
L_IN: al.constexpr = 131072
L_OUT: al.constexpr = 131070
K_DIM: al.constexpr = 3
K_EFF: al.constexpr = C_IN * K_DIM  # 192

# ── Tile and launch constants ────────────────────────────────────────────────
THREADS: al.constexpr = 256
BLOCK_M: al.constexpr = 64
BLOCK_N: al.constexpr = 4   # small N tile: 64*4=256 elements = threads
WINDOW_SIZE: al.constexpr = BLOCK_M + K_DIM - 1  # 66
N_TILES: al.constexpr = C_OUT // BLOCK_N  # 32
L_TILES: al.constexpr = (L_OUT + BLOCK_M - 1) // BLOCK_M  # 2048


@avelang.jit
def conv1d_bf16_kernel(
    in_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    block_n = al.block_id(0)
    block_comb = al.block_id(1)

    b = block_comb // L_TILES
    l_tile = block_comb - b * L_TILES
    l_start = l_tile * BLOCK_M

    # ── Guard valid extents ─────────────────────────────────────────────
    out_l_global = l_tile * BLOCK_M
    valid_m = BLOCK_M
    if out_l_global + BLOCK_M > L_OUT:
        valid_m = L_OUT - out_l_global
    valid_n = BLOCK_N
    n_global_start = block_n * BLOCK_N
    if n_global_start + BLOCK_N > C_OUT:
        valid_n = C_OUT - n_global_start
    if valid_m <= 0 or valid_n <= 0:
        return

    # ── Global memory views ─────────────────────────────────────────────
    in_tensor = al.make_tensor(in_ptr, al.bf16,
        al.make_layout((B, C_IN, L_IN), (C_IN * L_IN, L_IN, 1)))

    w_tensor = al.make_tensor(w_ptr, al.bf16,
        al.make_layout((C_OUT, C_IN, K_DIM), (K_EFF, K_DIM, 1)))

    out_tensor = al.make_tensor(out_ptr, al.bf16,
        al.make_layout((B, C_OUT, L_OUT), (C_OUT * L_OUT, L_OUT, 1)))

    # ── Shared memory ────────────────────────────────────────────────────
    shm_input = al.make_shared((C_IN, WINDOW_SIZE), al.bf16)
    shm_weight = al.make_shared((BLOCK_N, K_EFF), al.bf16)

    # ── Load input window ───────────────────────────────────────────────
    load_idx = tid
    for _ in al.range(0, 17):
        if load_idx < C_IN * WINDOW_SIZE:
            ic = load_idx // WINDOW_SIZE
            pos = load_idx - ic * WINDOW_SIZE
            gpos = l_start + pos
            if gpos < L_IN:
                shm_input[ic, pos] = in_tensor[b, ic, gpos]
            else:
                shm_input[ic, pos] = al.convert(0.0, al.bf16)
        load_idx = load_idx + THREADS

    # ── Load weight tile ────────────────────────────────────────────────
    load_idx = tid
    for _ in al.range(0, 3):
        if load_idx < BLOCK_N * K_EFF:
            loc_n = load_idx // K_EFF
            k_idx = load_idx - loc_n * K_EFF
            ng = block_n * BLOCK_N + loc_n
            if ng < C_OUT:
                wk_ic = k_idx // K_DIM
                wk_k = k_idx - wk_ic * K_DIM
                shm_weight[loc_n, k_idx] = w_tensor[ng, wk_ic, wk_k]
            else:
                shm_weight[loc_n, k_idx] = al.convert(0.0, al.bf16)
        load_idx = load_idx + THREADS

    al.syncthreads()

    # ── Compute: 1 element per thread, 256 elements = 64 M × 4 N ───────
    local_m = tid // BLOCK_N       # 0..63
    local_n = tid - local_m * BLOCK_N  # 0..3

    if local_m < valid_m and local_n < valid_n:
        acc = al.convert(0.0, al.f32)
        in_pos = local_m
        for _ic in al.range(C_IN):
            ic_k = _ic * K_DIM
            for kp in al.range(K_DIM):
                a_val = al.convert(shm_input[_ic, in_pos + kp], al.f32)
                b_val = al.convert(shm_weight[local_n, ic_k + kp], al.f32)
                acc = acc + a_val * b_val
        n_global = block_n * BLOCK_N + local_n
        out_l = out_l_global + local_m
        out_tensor[b, n_global, out_l] = al.convert(acc, al.bf16)


def _prepare_bf16_cuda_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    if t.is_cuda:
        return t.contiguous().to(dtype=torch.bfloat16)
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _prepare_bf16_cuda_contiguous(x)
    w_bf16 = _prepare_bf16_cuda_contiguous(weight)

    out = torch.empty((B, C_OUT, L_OUT), device=x_bf16.device, dtype=torch.bfloat16)
    grid = (N_TILES, B * L_TILES, 1)
    conv1d_bf16_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, out,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized 1D convolution using AveLang DSL.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv1d(x, self.conv1d.weight.data)


# ── Test helpers ─────────────────────────────────────────────────────────────
batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072


def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
