import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH_SIZE = 1024
IN_FEATURES = 8192
OUT_FEATURES = 8192
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = 64 * WAVES_PER_BLOCK
K_TILES = IN_FEATURES // BLOCK_K


def _mfma_probe_launch():
    return ((1, 1, 1), (THREADS_PER_BLOCK, 1, 1))


@substrate.jit
def _load_stage(
    x: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    w_t: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),
    lds_a: S.Tensor((2, 128, 4), S.u32),
    lds_b: S.Tensor((2, 128, 4), S.u32),
    stage: S.i32,
    k_tile: S.i32,
):
    tid = S.thread_id(0)

    k_base = k_tile * BLOCK_K
    if tid < 128:
        frag = tid
        row = frag // 2
        col = k_base + (frag % 2) * 8
        row_byte_base = row * IN_FEATURES * 2
        byte_offset = row_byte_base + col * 2
        # Limit the descriptor to the end of the current row so OOB vector
        # loads return zero without explicit control flow.
        x_rsrc = S.amdgpu.make_rsrc(x, row_byte_base + IN_FEATURES * 2)
        lds_a[stage, frag] = S.amdgpu.raw_buffer_load_x4(
            x_rsrc,
            S.convert(byte_offset, S.i32),
            S.convert(0, S.i32),
            S.convert(0, S.i32),
        )
    else:
        frag = tid - 128
        row = k_base + frag // 4
        col = (frag % 4) * 8
        row_byte_base = row * OUT_FEATURES * 2
        byte_offset = row_byte_base + col * 2
        # Limit the descriptor to the end of the current row so OOB vector
        # loads return zero without explicit control flow.
        w_rsrc = S.amdgpu.make_rsrc(w_t, row_byte_base + OUT_FEATURES * 2)
        lds_b[stage, frag] = S.amdgpu.raw_buffer_load_x4(
            w_rsrc,
            S.convert(byte_offset, S.i32),
            S.convert(0, S.i32),
            S.convert(0, S.i32),
        )


@substrate.jit
def software_pipelined_mfma_probe_kernel(
    x: S.Tensor((BATCH_SIZE, IN_FEATURES), S.bf16),
    w_t: S.Tensor((IN_FEATURES, OUT_FEATURES), S.bf16),
    scratch: S.Tensor((THREADS_PER_BLOCK, 16), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % 64
    wave = tid // 64
    wave_row = wave // 2
    wave_col = wave % 2

    lds_a = S.make_shared((2, 128, 4), S.u32)
    lds_b = S.make_shared((2, 128, 4), S.u32)

    c_lane = S.full((16,), 0.0, S.f32)

    _load_stage(x, w_t, lds_a, lds_b, 0, 0)
    _load_stage(x, w_t, lds_a, lds_b, 1, 1)
    S.syncthreads()

    for k_tile in S.range(0, K_TILES - 2, 2):
        stage0 = k_tile % 2
        stage1 = (k_tile + 1) % 2
        next_tile = k_tile + 2

        _load_stage(x, w_t, lds_a, lds_b, stage0, next_tile)

        a_frag0 = S.view(
            lds_a[stage0, wave_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16)
        )
        b_frag0 = S.view(
            lds_b[stage0, wave_col * 64 + lane], S.Tensor((2, 4, 1), S.bf16)
        )
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], c_lane)
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], c_lane)

        a_frag1 = S.view(
            lds_a[stage1, wave_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16)
        )
        b_frag1 = S.view(
            lds_b[stage1, wave_col * 64 + lane], S.Tensor((2, 4, 1), S.bf16)
        )
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], c_lane)
        c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], c_lane)

        S.syncthreads()

    final_stage0 = (K_TILES - 2) % 2
    final_stage1 = (K_TILES - 1) % 2

    a_frag0 = S.view(
        lds_a[final_stage0, wave_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16)
    )
    b_frag0 = S.view(
        lds_b[final_stage0, wave_col * 64 + lane], S.Tensor((2, 4, 1), S.bf16)
    )
    c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], c_lane)
    c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], c_lane)

    a_frag1 = S.view(
        lds_a[final_stage1, wave_row * 64 + lane], S.Tensor((2, 4, 1), S.bf16)
    )
    b_frag1 = S.view(
        lds_b[final_stage1, wave_col * 64 + lane], S.Tensor((2, 4, 1), S.bf16)
    )
    c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], c_lane)
    c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], c_lane)

    scratch[tid] = c_lane


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._probe_scratch = None
        self._cached_w_t = None
        self._cached_weight_ptr = None
        self._cached_weight_device = None

    def _get_probe_scratch(self, device):
        if self._probe_scratch is None or self._probe_scratch.device != device:
            self._probe_scratch = torch.empty(
                (THREADS_PER_BLOCK, 16), device=device, dtype=torch.float32
            )
        return self._probe_scratch

    def _get_weight_t(self, device):
        weight = self.gemm.weight
        weight_ptr = weight.untyped_storage().data_ptr()
        if (
            self._cached_w_t is None
            or self._cached_weight_ptr != weight_ptr
            or self._cached_weight_device != device
        ):
            self._cached_w_t = weight.to(device=device, dtype=torch.bfloat16).t().contiguous()
            self._cached_weight_ptr = weight_ptr
            self._cached_weight_device = device
        return self._cached_w_t

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()

        w_t = self._get_weight_t(x.device)
        probe_scratch = self._get_probe_scratch(x.device)
        software_pipelined_mfma_probe_kernel[_mfma_probe_launch](
            x, w_t, probe_scratch, num_warps=WAVES_PER_BLOCK
        )

        y = torch.ops.aten.mm.default(x, w_t)
        bias = self.bias.to(device=x.device, dtype=y.dtype)
        y = torch.ops.aten.add.Tensor(y, bias)
        return torch.ops.aten.relu.default(y)
