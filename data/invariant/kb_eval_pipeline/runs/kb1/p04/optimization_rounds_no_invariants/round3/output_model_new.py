import torch
import torch.nn as nn


M = 2048
K = 1048576
N = 1


def _fixed_shape_matvec(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # The benchmark case is GEMV with a single output column, so use mv directly.
    return torch.mv(A, B.reshape(K)).reshape(M, 1)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._compiled_fast_path = None
        self._compile_attempted = False

        if hasattr(torch, "compile"):
            try:
                self._compiled_fast_path = torch.compile(
                    _fixed_shape_matvec,
                    fullgraph=True,
                    dynamic=False,
                )
            except Exception:
                self._compiled_fast_path = None
            self._compile_attempted = True

    def _can_use_fast_path(self, A: torch.Tensor, B: torch.Tensor) -> bool:
        return (
            tuple(A.shape) == (M, K)
            and tuple(B.shape) == (K, N)
            and A.dtype == torch.bfloat16
            and B.dtype == torch.bfloat16
            and A.is_cuda
            and B.is_cuda
        )

    def forward(self, A, B):
        if self._can_use_fast_path(A, B):
            if self._compiled_fast_path is not None:
                return self._compiled_fast_path(A, B)
            return _fixed_shape_matvec(A, B)
        return torch.matmul(A, B)
