import torch
import torch.nn as nn
import avelang
import avelang.language as al


@avelang.jit
def rms_norm_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.bf16),
    B: al.i32,
    F: al.i32,
    D1: al.i32,
    D2: al.i32,
    EPS: al.f32,
    BLOCK_D2: al.constexpr,
):
    tid = al.thread_id(0)
    bid_b = al.block_id(0)
    bid_d1 = al.block_id(1)
    bid_d2 = al.block_id(2)

    d2 = bid_d2 * BLOCK_D2 + tid

    f_stride = D1 * D2
    b_stride = F * f_stride

    layout = al.make_layout(
        (B, F, D1, D2),
        (b_stride, f_stride, D2, al.convert(1, al.i32)),
    )
    x = al.make_tensor(x_ptr, al.bf16, layout)
    out = al.make_tensor(out_ptr, al.bf16, layout)

    if d2 < D2:
        # Phase 1: accumulate sum of squares over all features
        sq_sum = al.convert(0.0, al.f32)
        for f in al.range(F):
            val_bf16 = x[bid_b, f, bid_d1, d2]
            val_f32 = al.convert(val_bf16, al.f32)
            sq_sum = sq_sum + val_f32 * val_f32

        # Compute RMS
        mean_sq = sq_sum / al.convert(F, al.f32)
        rms = al.sqrt(mean_sq + EPS)

        # Phase 2: normalize each feature and write output
        for f in al.range(F):
            val_bf16 = x[bid_b, f, bid_d1, d2]
            val_f32 = al.convert(val_bf16, al.f32)
            norm_f32 = val_f32 / rms
            out[bid_b, f, bid_d1, d2] = al.convert(norm_f32, al.bf16)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, D1, D2 = x.shape
        assert F == self.num_features, (
            f"Feature dimension mismatch: {F} != {self.num_features}"
        )

        out = torch.empty_like(x)

        BLOCK_D2 = 256
        grid_d2 = (D2 + BLOCK_D2 - 1) // BLOCK_D2

        rms_norm_kernel[lambda: ((B, D1, grid_d2), (BLOCK_D2, 1, 1))](
            x, out, B, F, D1, D2, self.eps, BLOCK_D2
        )
        return out
