import torch
import substrate
import substrate.language as S

# Block size for 1D tiling
BLOCK_SIZE: S.constexpr = 256

N_VAL: S.constexpr = 4096
M_VAL: S.constexpr = 4096


@substrate.jit
def diag_matmul_kernel(
    A: S.Tensor((N_VAL,), S.bf16),
    B: S.Tensor((N_VAL, M_VAL), S.bf16),
    C: S.Tensor((N_VAL, M_VAL), S.bf16),
):
    tid = S.thread_id(0)
    block_idx = S.block_id(0)

    # Each thread processes one element
    idx = block_idx * BLOCK_SIZE + tid
    row = idx // M_VAL
    col = idx % M_VAL

    if idx < N_VAL * M_VAL:
        a_val = A[row]
        C[row, col] = a_val * B[row, col]


def substrate_diag_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA/HIP device."

    A_bf16 = A.to(dtype=torch.bfloat16, device="cuda").contiguous()
    B_bf16 = B.to(dtype=torch.bfloat16, device="cuda").contiguous()

    C = torch.empty_like(B_bf16)

    total_elements = N_VAL * M_VAL
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    diag_matmul_kernel[lambda: ((grid_size, 1, 1), (BLOCK_SIZE, 1, 1))](
        A_bf16, B_bf16, C
    )

    return C


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return substrate_diag_matmul(A, B)
