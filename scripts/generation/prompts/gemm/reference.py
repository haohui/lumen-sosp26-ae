import torch
import torch.nn as nn


class Model(nn.Module):
    """Single GEMM reference: C = A @ B.T."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Perform matrix multiplication with B stored as [N, K].

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A, B.T)


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2


def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(N, K)
    return [A, B]


def get_init_inputs():
    return []
