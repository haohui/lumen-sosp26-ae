import torch
import torch.nn as nn
import avelang
import avelang.language as al

# ── compile-time constants ──────────────────────────────────────────────
BLOCK_SIZE = 256
OC = 64
IC = 16
KH = 3
KW = 3
IC_KH_KW = IC * KH * KW           # 144
KH_KW = KH * KW                   #   9
NUM_WEIGHTS = OC * IC_KH_KW       # 9216
NUM_WEIGHTS_ALIGNED = ((NUM_WEIGHTS + 7) // 8) * 8
LOADS_PER_THREAD = (NUM_WEIGHTS_ALIGNED + BLOCK_SIZE - 1) // BLOCK_SIZE


# ═══════════════════════════════════════════════════════════════════════
#  AveLang kernel: fused Conv2d → channel-min → tanh → tanh
# ═══════════════════════════════════════════════════════════════════════
@avelang.jit
def conv2d_min_tanh_kernel(
    x_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    H: al.i32,
    W: al.i32,
    OH: al.i32,
    OW: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    gid = bid * BLOCK_SIZE + tid
    total = B * OH * OW

    if gid < total:
        # ── decode (b, h, w) ──────────────────────────────────────────
        oh_ow = OH * OW
        b = gid // oh_ow
        rem = gid - b * oh_ow
        h = rem // OW
        w = rem - h * OW

        # ── tensor views ──────────────────────────────────────────────
        x = al.make_tensor(
            x_ptr, al.bf16,
            al.make_layout((B, IC, H, W), (IC * H * W, H * W, W, 1)),
        )
        w_flat = al.make_tensor(
            w_ptr, al.bf16,
            al.make_layout((NUM_WEIGHTS,), (1,)),
        )
        b_t = al.make_tensor(
            b_ptr, al.bf16,
            al.make_layout((OC,), (1,)),
        )
        out = al.make_tensor(
            out_ptr, al.bf16,
            al.make_layout((B, 1, OH, OW), (OH * OW, OH * OW, OW, 1)),
        )

        # ── stage weights into shared memory ─────────────────────────
        shm_w = al.make_shared((NUM_WEIGHTS_ALIGNED,), al.bf16)
        for i in al.range(LOADS_PER_THREAD):
            idx = tid + i * BLOCK_SIZE
            if idx < NUM_WEIGHTS:
                shm_w[idx] = w_flat[idx]
        al.syncthreads()

        # ── load input patch into local registers once ────────────────
        x_patch = al.make_local((IC, KH, KW), al.bf16)
        for ci in al.range(IC):
            for kh in al.range(KH):
                for kw in al.range(KW):
                    x_patch[ci, kh, kw] = x[b, ci, h + kh, w + kw]

        # ── seed channel-min with output channel 0 ────────────────────
        zero = al.convert(0, al.i32)
        acc = al.convert(b_t[zero], al.f32)
        w_base = zero * IC_KH_KW
        for ci in al.range(IC):
            ci_off = ci * KH_KW
            for kh in al.range(KH):
                kh_off = kh * KW
                for kw in al.range(KW):
                    xv = al.convert(x_patch[ci, kh, kw], al.f32)
                    wv = al.convert(shm_w[w_base + ci_off + kh_off + kw], al.f32)
                    acc = acc + xv * wv
        best = acc

        # ── remaining output channels ─────────────────────────────────
        for co in al.range(1, OC):
            acc = al.convert(b_t[co], al.f32)
            w_base = co * IC_KH_KW
            for ci in al.range(IC):
                ci_off = ci * KH_KW
                for kh in al.range(KH):
                    kh_off = kh * KW
                    for kw in al.range(KW):
                        xv = al.convert(x_patch[ci, kh, kw], al.f32)
                        wv = al.convert(shm_w[w_base + ci_off + kh_off + kw], al.f32)
                        acc = acc + xv * wv
            if acc < best:
                best = acc

        # ── epilogue: tanh ∘ tanh ─────────────────────────────────────
        result = al.tanh(best)
        result = al.tanh(result)
        out[b, 0, h, w] = al.convert(result, al.bf16)


# ═══════════════════════════════════════════════════════════════════════
#  host wrapper
# ═══════════════════════════════════════════════════════════════════════
def avelang_conv_min_tanh(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Fused Conv2d + channel-min + tanh + tanh via AveLang."""
    if not x.is_cuda:
        raise RuntimeError("Input must be on CUDA/HIP device.")

    B, in_c, H, W = x.shape
    out_c, wt_ic, kh, kw = weight.shape

    x_bf16 = x.contiguous().to(torch.bfloat16)
    w_bf16 = weight.contiguous().to(torch.bfloat16)
    b_bf16 = bias.contiguous().to(torch.bfloat16)

    OH = H - kh + 1
    OW_val = W - kw + 1

    out_bf16 = torch.empty(
        (B, 1, OH, OW_val), dtype=torch.bfloat16, device=x.device,
    )

    total = B * OH * OW_val
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)

    conv2d_min_tanh_kernel[lambda: (grid, (BLOCK_SIZE, 1, 1))](
        x_bf16, w_bf16, b_bf16, out_bf16, B, H, W, OH, OW_val,
    )

    return out_bf16


# ═══════════════════════════════════════════════════════════════════════
#  ModelNew  —  same public contract as the reference Model
# ═══════════════════════════════════════════════════════════════════════
class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_conv_min_tanh(x, self.conv.weight, self.conv.bias)
