import torch
import torch.nn as nn
import torch.nn.functional as F


BATCH_SIZE = 1024
INPUT_SIZE = 8192
HIDDEN_SIZE = 8192
SCALE_FACTOR = 2.0
CLAMP_MIN = -10.0
CLAMP_MAX = 10.0
K_CHUNK = 1024
K_UNROLL = 2


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self._cached_weight_fp32 = None
        self._cached_weight_ptr = None
        self._cached_weight_device = None
        self._cached_bias_fp32 = None
        self._cached_bias_ptr = None
        self._cached_bias_device = None

    def _get_weight_fp32(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.matmul.weight
        weight_ptr = weight.untyped_storage().data_ptr()
        if (
            self._cached_weight_fp32 is None
            or self._cached_weight_ptr != weight_ptr
            or self._cached_weight_device != x.device
        ):
            self._cached_weight_fp32 = weight.detach().to(device=x.device, dtype=torch.float32).contiguous()
            self._cached_weight_ptr = weight_ptr
            self._cached_weight_device = x.device
        return self._cached_weight_fp32

    def _get_bias_fp32(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.matmul.bias
        bias_ptr = bias.untyped_storage().data_ptr()
        if (
            self._cached_bias_fp32 is None
            or self._cached_bias_ptr != bias_ptr
            or self._cached_bias_device != x.device
        ):
            self._cached_bias_fp32 = bias.detach().to(device=x.device, dtype=torch.float32).contiguous()
            self._cached_bias_ptr = bias_ptr
            self._cached_bias_device = x.device
        return self._cached_bias_fp32

    def _chunked_linear_pipelined(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.to(torch.float32)
        weight_fp32 = self._get_weight_fp32(x)
        bias_fp32 = self._get_bias_fp32(x)

        batch = x_fp32.shape[0]
        hidden = weight_fp32.shape[0]
        k_total = weight_fp32.shape[1]
        out = torch.zeros((batch, hidden), device=x.device, dtype=torch.float32)

        prefetch = []
        warmup_end = min(k_total, K_CHUNK * K_UNROLL)
        for k0 in range(0, warmup_end, K_CHUNK):
            k1 = min(k0 + K_CHUNK, k_total)
            prefetch.append((x_fp32[:, k0:k1], weight_fp32[:, k0:k1]))

        loop_k = 0
        while prefetch:
            current_x, current_w = prefetch.pop(0)
            next_k = loop_k + K_CHUNK * K_UNROLL
            if next_k < k_total:
                next_k1 = min(next_k + K_CHUNK, k_total)
                prefetch.append((x_fp32[:, next_k:next_k1], weight_fp32[:, next_k:next_k1]))

            out.add_(torch.einsum("bk,nk->bn", current_x, current_w))
            loop_k += K_CHUNK

        out.add_(bias_fp32)
        return out

    def forward(self, x):
        out_dtype = x.dtype
        x = self._chunked_linear_pipelined(x).to(out_dtype)
        x = x * (self.scale_factor * 2.0)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        return x * F.mish(x)
