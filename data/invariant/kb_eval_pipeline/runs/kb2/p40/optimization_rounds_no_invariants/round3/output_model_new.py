import torch
import torch.nn as nn
import substrate
import substrate.language as S


BATCH_SIZE = 16384
IN_FEATURES = 4096
OUT_FEATURES = 4096
SCALING_FACTOR = 0.5

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVE_SIZE = 64
WARPS_PER_BLOCK = 4
THREADS_PER_BLOCK = WAVE_SIZE * WARPS_PER_BLOCK
NUM_K_TILES = IN_FEATURES // BLOCK_K
SCALE = 1.0 + SCALING_FACTOR
X_NUM_BYTES = BATCH_SIZE * IN_FEATURES * 2
W_NUM_BYTES = OUT_FEATURES * IN_FEATURES * 2
A_STAGE_WORDS = BLOCK_M * 2
B_STAGE_WORDS = BLOCK_N * 2


def _launch():
    return ((OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))


@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    W: S.Tensor((OUT_FEATURES, IN_FEATURES), S.bf16),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    tid = S.thread_id(0)
    lane = tid % WAVE_SIZE
    warp_id = tid // WAVE_SIZE
    warp_m = warp_id // 2
    warp_n = warp_id % 2
    lane_quad = lane // 32
    lane_32 = lane % 32

    block_row = S.block_id(1) * BLOCK_M
    block_col = S.block_id(0) * BLOCK_N

    zero = S.convert(0, S.i32)
    x_rsrc = S.amdgpu.make_rsrc(X, S.convert(X_NUM_BYTES, S.i32))
    w_rsrc = S.amdgpu.make_rsrc(W, S.convert(W_NUM_BYTES, S.i32))

    shared_a = S.make_shared((A_STAGE_WORDS * 2, 4), S.u32)
    shared_b = S.make_shared((B_STAGE_WORDS * 2, 4), S.u32)

    c_lane = S.full((16,), 0.0, S.f32)

    is_a_loader = tid < A_STAGE_WORDS
    a_stage_base0 = 0
    b_stage_base0 = 0
    a_stage_base1 = A_STAGE_WORDS
    b_stage_base1 = B_STAGE_WORDS

    if is_a_loader:
        row = tid // 2
        k_vec = tid % 2
        x_elem_offset = ((block_row + row) * IN_FEATURES + k_vec * 8) * 2
        shared_a[a_stage_base0 + tid] = S.amdgpu.raw_buffer_load_x4(
            x_rsrc, zero, S.convert(x_elem_offset, S.i32), 0
        )
    else:
        b_tid = tid - A_STAGE_WORDS
        col = b_tid // 2
        k_vec = b_tid % 2
        w_elem_offset = ((block_col + col) * IN_FEATURES + k_vec * 8) * 2
        shared_b[b_stage_base0 + b_tid] = S.amdgpu.raw_buffer_load_x4(
            w_rsrc, zero, S.convert(w_elem_offset, S.i32), 0
        )

    S.syncthreads()

    for k_pair in S.range(NUM_K_TILES // 2):
        k_tile0 = k_pair * 2
        k_tile1 = k_tile0 + 1
        k_tile2 = k_tile0 + 2

        a_frag0 = S.view(
            shared_a[a_stage_base0 + (warp_m * 32 + lane_32) * 2 + lane_quad],
            S.Tensor((2, 4, 1), S.bf16),
        )
        b_frag0 = S.view(
            shared_b[b_stage_base0 + (warp_n * 32 + lane_32) * 2 + lane_quad],
            S.Tensor((2, 4, 1), S.bf16),
        )

        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], c_lane)

        if is_a_loader:
            row = tid // 2
            k_vec = tid % 2
            x_elem_offset = ((block_row + row) * IN_FEATURES + k_tile1 * BLOCK_K + k_vec * 8) * 2
            shared_a[a_stage_base1 + tid] = S.amdgpu.raw_buffer_load_x4(
                x_rsrc, zero, S.convert(x_elem_offset, S.i32), 0
            )
        else:
            b_tid = tid - A_STAGE_WORDS
            col = b_tid // 2
            k_vec = b_tid % 2
            w_elem_offset = ((block_col + col) * IN_FEATURES + k_tile1 * BLOCK_K + k_vec * 8) * 2
            shared_b[b_stage_base1 + b_tid] = S.amdgpu.raw_buffer_load_x4(
                w_rsrc, zero, S.convert(w_elem_offset, S.i32), 0
            )

        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], c_lane)
        S.syncthreads()

        a_frag1 = S.view(
            shared_a[a_stage_base1 + (warp_m * 32 + lane_32) * 2 + lane_quad],
            S.Tensor((2, 4, 1), S.bf16),
        )
        b_frag1 = S.view(
            shared_b[b_stage_base1 + (warp_n * 32 + lane_32) * 2 + lane_quad],
            S.Tensor((2, 4, 1), S.bf16),
        )

        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], c_lane)

        if is_a_loader:
            row = tid // 2
            k_vec = tid % 2
            x_elem_offset = ((block_row + row) * IN_FEATURES + k_tile2 * BLOCK_K + k_vec * 8) * 2
            shared_a[a_stage_base0 + tid] = S.amdgpu.raw_buffer_load_x4(
                x_rsrc, zero, S.convert(x_elem_offset, S.i32), 0
            )
        else:
            b_tid = tid - A_STAGE_WORDS
            col = b_tid // 2
            k_vec = b_tid % 2
            w_elem_offset = ((block_col + col) * IN_FEATURES + k_tile2 * BLOCK_K + k_vec * 8) * 2
            shared_b[b_stage_base0 + b_tid] = S.amdgpu.raw_buffer_load_x4(
                w_rsrc, zero, S.convert(w_elem_offset, S.i32), 0
            )

        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], c_lane)
        S.syncthreads()

    for slot in S.range(16):
        row_in_warp = (slot % 4) + 8 * (slot // 4) + 4 * lane_quad
        col_in_warp = lane_32
        out_row = block_row + warp_m * 32 + row_in_warp
        out_col = block_col + warp_n * 32 + col_in_warp
        acc = c_lane[slot] + S.convert(BIAS0[out_col], S.f32)
        Y[out_row, out_col] = S.convert(acc * S.convert(SCALE, S.f32), S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES) or x.dtype != torch.bfloat16 or self.scaling_factor != SCALING_FACTOR:
            raise RuntimeError("This fused kernel only supports the benchmark input shape and dtype.")
        if not x.is_cuda:
            raise RuntimeError("This fused kernel requires a CUDA/ROCm device input.")
        if self.matmul.weight.device != x.device or self.matmul.bias.device != x.device:
            raise RuntimeError("Model parameters must already be on the same device as the input.")
        if self.matmul.weight.dtype != torch.bfloat16 or self.matmul.bias.dtype != torch.bfloat16:
            raise RuntimeError("Model parameters must be bfloat16.")

        y = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)
        fused_kernel[_launch](x.contiguous(), self.matmul.weight, self.matmul.bias, y)
        return y
