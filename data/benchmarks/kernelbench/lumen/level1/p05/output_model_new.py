import torch
import substrate
import substrate.language as S


THREADS = 256
TILE_SIZE = 4


@substrate.jit
def matrix_scalar_mul_kernel(
    A_ptr: S.Pointer(S.bf16),
    C_ptr: S.Pointer(S.bf16),
    scalar_ptr: S.Pointer(S.bf16),
    n: S.u32,
):
    data_layout = S.make_layout((n,), (1,))
    A = S.make_tensor(A_ptr, S.bf16, data_layout)
    C = S.make_tensor(C_ptr, S.bf16, data_layout)
    scalar_tensor = S.make_tensor(scalar_ptr, S.bf16, data_layout)

    scalar_f32 = S.convert(scalar_tensor[0], S.f32)

    tid = S.thread_id(0)
    block_idx = S.block_id(0)
    base_idx = (block_idx * THREADS + tid) * TILE_SIZE

    for t in S.range(TILE_SIZE):
        idx = base_idx + t
        if idx < n:
            a_val = S.convert(A[idx], S.f32)
            C[idx] = S.convert(a_val * scalar_f32, S.bf16)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_scalar_storage", torch.zeros(1, dtype=torch.bfloat16), persistent=False)

    def _set_scalar(self, s, device: torch.device):
        if isinstance(s, torch.Tensor):
            s_dev = s.to(device=device, dtype=self._scalar_storage.dtype)
            if s_dev.numel() != 1:
                raise ValueError(f"Expected scalar tensor for s, got shape {tuple(s.shape)}")
            self._scalar_storage.copy_(s_dev.reshape(1))
        else:
            self._scalar_storage.fill_(s)

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        assert A.is_cuda, "Tensor must be on CUDA/HIP device."

        A_bf16 = A if (A.dtype == torch.bfloat16 and A.is_contiguous()) else A.to(dtype=torch.bfloat16, device=A.device).contiguous()
        C = torch.empty_like(A_bf16)
        total_elements = A_bf16.numel()
        if total_elements == 0:
            return C

        self._set_scalar(s, A_bf16.device)

        elements_per_block = THREADS * TILE_SIZE
        grid_size = (total_elements + elements_per_block - 1) // elements_per_block
        matrix_scalar_mul_kernel[lambda: ((grid_size, 1, 1), (THREADS, 1, 1))](
            A_bf16.data_ptr(),
            C.data_ptr(),
            self._scalar_storage.data_ptr(),
            total_elements,
        )
        return C
