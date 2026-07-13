import torch
import torch.nn as nn
import avelang
import avelang.language as al

TILE_K = 256


@avelang.jit
def argmin_kernel(
    x_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.i64),
    B: al.i32,
    R: al.i32,
    K: al.i32,
    stride_b: al.i32,
    stride_r: al.i32,
    total_elements: al.i32,
    out_elements: al.i32,
    num_k_tiles: al.i32,
):
    x_layout = al.make_layout((total_elements,), (1,))
    x = al.make_tensor(x_ptr, al.bf16, x_layout)

    out_layout = al.make_layout((out_elements,), (1,))
    out = al.make_tensor(out_ptr, al.i64, out_layout)

    bid = al.block_id(0)
    b = bid // num_k_tiles
    k_tile = bid % num_k_tiles

    tid = al.thread_id(0)
    k = k_tile * TILE_K + tid

    if k < K:
        ptr = b * stride_b + k
        local_min = al.convert(3.402823e+38, al.f32)
        local_idx = al.convert(-1, al.i64)

        for r in al.range(0, R):
            val = al.convert(x[ptr], al.f32)
            if val < local_min:
                local_min = val
                local_idx = al.convert(r, al.i64)
            ptr = ptr + stride_r

        out[b * K + k] = local_idx


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input must be on CUDA/HIP."
        assert x.dim() == 3, f"Expected 3D input, got shape {x.shape}"

        x = x.contiguous()
        x_bf16 = x.to(torch.bfloat16)

        B, R, K = x_bf16.shape

        stride_b = R * K
        stride_r = K
        total_elements = B * R * K
        out_elements = B * K

        out = torch.empty((B, K), dtype=torch.int64, device=x.device)

        num_k_tiles = (K + TILE_K - 1) // TILE_K
        grid = (B * num_k_tiles, 1, 1)
        block = (TILE_K, 1, 1)

        argmin_kernel[lambda: (grid, block)](
            x_bf16, out,
            B, R, K,
            stride_b, stride_r,
            total_elements, out_elements,
            num_k_tiles,
        )

        return out
