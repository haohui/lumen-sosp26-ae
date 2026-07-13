import torch
import torch.nn as nn
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def maxpool3d_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    B: al.i32,
    C: al.i32,
    D1: al.i32,
    D2: al.i32,
    D3: al.i32,
    OD1: al.i32,
    OD2: al.i32,
    OD3: al.i32,
    K: al.i32,
    S: al.i32,
    P: al.i32,
    DL: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    gid = bid * BLOCK_SIZE + tid
    total = B * C * OD1 * OD2 * OD3

    if gid < total:
        # Unflatten linear index into 5D output coordinates (b, c, od1, od2, od3)
        tmp = gid
        od3 = tmp % OD3
        tmp = tmp // OD3
        od2 = tmp % OD2
        tmp = tmp // OD2
        od1 = tmp % OD1
        tmp = tmp // OD1
        c = tmp % C
        b = tmp // C

        # Compute input base coordinates from output position
        id1_base = od1 * S - P
        id2_base = od2 * S - P
        id3_base = od3 * S - P

        layout_in = al.make_layout(
            (B, C, D1, D2, D3),
            (C * D1 * D2 * D3, D1 * D2 * D3, D2 * D3, D3, 1),
        )
        inp = al.make_tensor(input_ptr, al.bf16, layout_in)

        # Initialize with a very negative value as -inf for max
        neg_inf_f32 = al.convert(-1.0e9, al.f32)
        cur_max = al.convert(neg_inf_f32, al.bf16)

        for kd1 in al.range(K):
            id1 = id1_base + kd1 * DL
            if id1 >= 0 and id1 < D1:
                for kd2 in al.range(K):
                    id2 = id2_base + kd2 * DL
                    if id2 >= 0 and id2 < D2:
                        for kd3 in al.range(K):
                            id3 = id3_base + kd3 * DL
                            if id3 >= 0 and id3 < D3:
                                val = inp[b, c, id1, id2, id3]
                                if val > cur_max:
                                    cur_max = val

        layout_out = al.make_layout(
            (B, C, OD1, OD2, OD3),
            (C * OD1 * OD2 * OD3, OD1 * OD2 * OD3, OD2 * OD3, OD3, 1),
        )
        out = al.make_tensor(output_ptr, al.bf16, layout_out)
        out[b, c, od1, od2, od3] = cur_max


def _compute_output_size(input_size: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
    """PyTorch-compatible MaxPool3d output size (floor mode)."""
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def avelang_maxpool3d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"
    assert x.ndim == 5, "Input must be 5D (N, C, D, H, W)"

    B, C, D1, D2, D3 = x.shape
    K = kernel_size
    S = stride
    P = padding
    DL = dilation

    OD1 = _compute_output_size(D1, K, S, P, DL)
    OD2 = _compute_output_size(D2, K, S, P, DL)
    OD3 = _compute_output_size(D3, K, S, P, DL)

    x_contiguous = x.contiguous()
    output = torch.empty((B, C, OD1, OD2, OD3), dtype=torch.bfloat16, device=x.device)

    total_output = B * C * OD1 * OD2 * OD3
    num_blocks = (total_output + BLOCK_SIZE - 1) // BLOCK_SIZE

    maxpool3d_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contiguous, output, B, C, D1, D2, D3, OD1, OD2, OD3, K, S, P, DL
    )

    return output


class ModelNew(nn.Module):
    """
    Optimized Max Pooling 3D using AveLang DSL kernel.
    """
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super(ModelNew, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avelang_maxpool3d(x, self.kernel_size, self.stride, self.padding, self.dilation)
