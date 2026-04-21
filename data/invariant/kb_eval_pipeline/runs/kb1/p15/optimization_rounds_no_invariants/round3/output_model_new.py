import torch
import torch.nn as nn


M = 4096

TILE_M = 128
TILE_N = 128
TILE_K = 64


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._workspace_key = None
        self._stage_a = None
        self._stage_b = None
        self._ready_events = None
        self._released_events = None
        self._prefetch_stream = None
        self._diag_mask = None

    def _ensure_workspace(self, device: torch.device, dtype: torch.dtype):
        key = (device.type, device.index, dtype)
        if self._workspace_key == key:
            return

        self._workspace_key = key
        self._stage_a = [
            torch.empty((TILE_M, TILE_K), device=device, dtype=dtype),
            torch.empty((TILE_M, TILE_K), device=device, dtype=dtype),
        ]
        self._stage_b = [
            torch.empty((TILE_K, TILE_N), device=device, dtype=dtype),
            torch.empty((TILE_K, TILE_N), device=device, dtype=dtype),
        ]
        self._ready_events = [torch.cuda.Event() for _ in range(2)]
        self._released_events = [torch.cuda.Event() for _ in range(2)]
        self._prefetch_stream = torch.cuda.Stream(device=device)
        self._diag_mask = torch.tril(
            torch.ones((TILE_M, TILE_N), device=device, dtype=torch.bool)
        )
        current = torch.cuda.current_stream(device=device)
        for event in self._released_events:
            event.record(current)

    def _stage_tile(self, A, B, row0, row1, col0, col1, k0, buf_idx):
        a_buf = self._stage_a[buf_idx]
        b_buf = self._stage_b[buf_idx]
        m_extent = row1 - row0
        n_extent = col1 - col0
        k_extent = min(max(row1 - k0, 0), TILE_K)

        with torch.cuda.stream(self._prefetch_stream):
            self._prefetch_stream.wait_event(self._released_events[buf_idx])
            a_buf.zero_()
            b_buf.zero_()
            a_buf[:m_extent, :k_extent].copy_(
                A[row0:row1, k0 : k0 + k_extent], non_blocking=True
            )
            b_buf[:k_extent, :n_extent].copy_(
                B[k0 : k0 + k_extent, col0:col1], non_blocking=True
            )
            self._ready_events[buf_idx].record(self._prefetch_stream)

    def _compute_tile(self, out, A, B, row0, col0):
        row1 = min(row0 + TILE_M, A.shape[0])
        col1 = min(col0 + TILE_N, B.shape[1])
        if col0 >= row1:
            return

        k_start = col0
        if k_start >= row1:
            return

        m_extent = row1 - row0
        n_extent = col1 - col0
        num_k_tiles = (row1 - k_start + TILE_K - 1) // TILE_K
        if num_k_tiles <= 0:
            return

        c_tile = torch.zeros((m_extent, n_extent), device=A.device, dtype=torch.float32)
        current_stream = torch.cuda.current_stream(device=A.device)
        num_k_tiles_rounded = (num_k_tiles + 1) & ~1

        self._stage_tile(A, B, row0, row1, col0, col1, k_start, 0)
        self._stage_tile(A, B, row0, row1, col0, col1, k_start + TILE_K, 1)

        for k_tile in range(0, num_k_tiles_rounded, 2):
            buf0 = k_tile & 1
            buf1 = (k_tile + 1) & 1

            current_stream.wait_event(self._ready_events[buf0])
            current_stream.wait_event(self._ready_events[buf1])

            c_tile += torch.einsum(
                "ik,kj->ij",
                self._stage_a[buf0][:m_extent].to(torch.float32),
                self._stage_b[buf0][:, :n_extent].to(torch.float32),
            )
            c_tile += torch.einsum(
                "ik,kj->ij",
                self._stage_a[buf1][:m_extent].to(torch.float32),
                self._stage_b[buf1][:, :n_extent].to(torch.float32),
            )

            self._released_events[buf0].record(current_stream)
            self._released_events[buf1].record(current_stream)

            self._stage_tile(
                A,
                B,
                row0,
                row1,
                col0,
                col1,
                k_start + (k_tile + 2) * TILE_K,
                buf0,
            )
            self._stage_tile(
                A,
                B,
                row0,
                row1,
                col0,
                col1,
                k_start + (k_tile + 3) * TILE_K,
                buf1,
            )

        out[row0:row1, col0:col1] = c_tile.to(out.dtype)
        if row0 == col0:
            out[row0:row1, col0:col1].masked_fill_(
                ~self._diag_mask[:m_extent, :n_extent], 0
            )

    def forward(self, A, B):
        A = A.contiguous()
        B = B.contiguous()

        if not A.is_cuda or not B.is_cuda:
            C = torch.einsum("ik,kj->ij", A, B)
            return torch.tril(C)

        self._ensure_workspace(A.device, A.dtype)

        out = torch.zeros((A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)
        for row0 in range(0, A.shape[0], TILE_M):
            max_col = min(row0 + TILE_M, B.shape[1])
            for col0 in range(0, max_col, TILE_N):
                self._compute_tile(out, A, B, row0, col0)
        return out
