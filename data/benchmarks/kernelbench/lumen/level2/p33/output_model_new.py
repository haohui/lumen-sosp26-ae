import torch
import torch.nn as nn
import torch.nn.functional as F
import substrate
import substrate.language as S


BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
POST_BLOCK = 256


@substrate.jit
def scale_bf16_kernel(
    x: S.Pointer(S.bf16),
    out: S.Pointer(S.bf16),
    scale: S.Pointer(S.bf16),
    rows: S.u32,
    cols: S.u32,
):
    layout_x = S.make_layout((rows, cols), (cols, 1))
    layout_out = S.make_layout((rows, cols), (cols, 1))
    layout_scale = S.make_layout((cols,), (1,))

    g_x = S.make_tensor(x, S.bf16, layout_x)
    g_out = S.make_tensor(out, S.bf16, layout_out)
    g_scale = S.make_tensor(scale, S.bf16, layout_scale)

    row = S.block_id(1)
    col = S.block_id(0) * S.block_dim(0) + S.thread_id(0)
    if row < rows and col < cols:
        g_out[row, col] = g_x[row, col] * g_scale[col]


def _launch_scale(rows: int, cols: int, x: torch.Tensor, out: torch.Tensor, scale: torch.Tensor):
    grid_x = (cols + POST_BLOCK - 1) // POST_BLOCK
    scale_bf16_kernel[lambda: ((grid_x, rows, 1), (POST_BLOCK, 1, 1))](
        x, out, scale, rows, cols
    )


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

        self._scale_workspace = None
        self._scale_workspace_key = None

    def _get_scale_workspace(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        workspace_key = (batch_size, self.gemm.out_features, device, dtype)
        if self._scale_workspace is None or self._scale_workspace_key != workspace_key:
            self._scale_workspace = torch.empty(
                (batch_size, self.gemm.out_features),
                device=device,
                dtype=dtype,
            )
            self._scale_workspace_key = workspace_key
        return self._scale_workspace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device required for Substrate kernels.")

        target_device = self.gemm.weight.device
        target_dtype = self.gemm.weight.dtype
        src_device = x.device
        src_dtype = x.dtype

        if x.device != target_device or x.dtype != target_dtype:
            x = x.to(device=target_device, dtype=target_dtype)
        if not x.is_contiguous():
            x = x.contiguous()

        gemm_out = F.linear(x, self.gemm.weight, self.gemm.bias)
        scaled = self._get_scale_workspace(gemm_out.shape[0], target_device, gemm_out.dtype)
        _launch_scale(gemm_out.shape[0], gemm_out.shape[1], gemm_out, scaled, self.scale)
        out = self.bn(scaled)

        if out.device == src_device and out.dtype == src_dtype:
            return out
        return out.to(device=src_device, dtype=src_dtype)


batch_size = BATCH_SIZE
in_features = IN_FEATURES
out_features = OUT_FEATURES
scale_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, scale_shape]
