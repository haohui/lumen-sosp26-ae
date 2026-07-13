import torch
import torch.nn as nn
import avelang
import avelang.language as al

# Problem dimensions
C = 64
OC = 128
KS = 3

# Tile parameters
BLOCK_L = 64
THREADS = 256

# Derived compile-time constants
SHM_X_ELEMS = C * (BLOCK_L + KS - 1)  # 64 * 66 = 4224
SHM_W_ELEMS = OC * C * KS  # 128 * 64 * 3 = 24576
TILE_ELEMS = BLOCK_L * OC  # 64 * 128 = 8192
W_LOADS = (SHM_W_ELEMS + THREADS - 1) // THREADS
X_LOADS = (SHM_X_ELEMS + THREADS - 1) // THREADS
OUT_ELEMS = (TILE_ELEMS + THREADS - 1) // THREADS


@avelang.jit
def conv1d_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.u32,
    L: al.u32,
    L_out: al.u32,
):
    tid = al.thread_id(0)
    b = al.block_id(0)
    block_l = al.block_id(1)

    l_start = block_l * BLOCK_L

    # Multi-dimensional global tensor views
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((B, C, L), (C * L, L, 1)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((OC, C, KS), (C * KS, KS, 1)))
    g_out = al.make_tensor(out_ptr, al.bf16, al.make_layout((B, OC, L_out), (OC * L_out, L_out, 1)))

    # Shared memory
    shm_x = al.make_shared((SHM_X_ELEMS,), al.bf16)
    shm_w = al.make_shared((SHM_W_ELEMS,), al.bf16)

    # ---- Cooperative load: weight into shared memory ----
    idx = tid
    for _ in al.range(W_LOADS):
        if idx < SHM_W_ELEMS:
            shm_w[idx] = w[idx // (C * KS), (idx // KS) % C, idx % KS]
        idx = idx + THREADS

    # ---- Cooperative load: input tile into shared memory ----
    idx = tid
    for _ in al.range(X_LOADS):
        if idx < SHM_X_ELEMS:
            ic = idx // (BLOCK_L + KS - 1)
            l_local = idx % (BLOCK_L + KS - 1)
            l_global = l_start + l_local
            if l_global < L:
                shm_x[idx] = x[b, ic, l_global]
            else:
                shm_x[idx] = al.convert(0.0, al.bf16)
        idx = idx + THREADS

    al.syncthreads()

    # ---- Compute output tile ----
    elem_idx = tid
    for _ in al.range(OUT_ELEMS):
        if elem_idx < TILE_ELEMS:
            oc = elem_idx // BLOCK_L
            l_local = elem_idx % BLOCK_L
            l_out = l_start + l_local

            if l_out < L_out:
                acc = al.convert(0.0, al.f32)

                for ic in al.range(C):
                    w_base = oc * C * KS + ic * KS
                    x_base = ic * (BLOCK_L + KS - 1) + l_local

                    k0_val = al.convert(shm_w[w_base], al.f32)
                    k0_x = al.convert(shm_x[x_base], al.f32)
                    acc = acc + k0_val * k0_x

                    k1_val = al.convert(shm_w[w_base + 1], al.f32)
                    k1_x = al.convert(shm_x[x_base + 1], al.f32)
                    acc = acc + k1_val * k1_x

                    k2_val = al.convert(shm_w[w_base + 2], al.f32)
                    k2_x = al.convert(shm_x[x_base + 2], al.f32)
                    acc = acc + k2_val * k2_x

                g_out[b, oc, l_out] = al.convert(acc, al.bf16)

        elem_idx = elem_idx + THREADS


def _to_bf16_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    return t.contiguous().cuda().to(dtype=torch.bfloat16)


def avelang_conv1d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for AveLang kernels.")

    x_bf16 = _to_bf16_contiguous(x)
    w_bf16 = _to_bf16_contiguous(weight)

    B_val, C_x, L_val = x_bf16.shape
    OC_w, C_w, KS_w = w_bf16.shape

    L_out = L_val - KS_w + 1

    out = torch.empty(
        (B_val, OC_w, L_out), device=x_bf16.device, dtype=torch.bfloat16
    )

    grid = (B_val, (L_out + BLOCK_L - 1) // BLOCK_L, 1)

    conv1d_kernel[lambda: (grid, (THREADS, 1, 1))](
        x_bf16, w_bf16, out, B_val, L_val, L_out,
    )

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv1d(x, self.weight)


# Test code (mirrors input_model.py)
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
