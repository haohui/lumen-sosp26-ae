import torch
import torch.nn as nn

import substrate
import substrate.language as S


BATCH_SIZE = 128
IN_FEATURES = 32768
OUT_FEATURES = 32768
SCALING_FACTOR = 2.0

BLOCK_M = 64
BLOCK_N = 64
WAVE_M = 32
WAVE_N = 32
WAVES_PER_BLOCK = 4
THREADS_PER_BLOCK = 64 * WAVES_PER_BLOCK
K_CHUNK = 16
CHUNKS_PER_ROW = IN_FEATURES // K_CHUNK
U32S_PER_LANE_FRAGMENT = 4
U32S_PER_CHUNK = 2 * U32S_PER_LANE_FRAGMENT
ROW_STRIDE_U32 = CHUNKS_PER_ROW * U32S_PER_CHUNK


def _launch():
    return ((OUT_FEATURES // BLOCK_N, BATCH_SIZE // BLOCK_M, 1), (THREADS_PER_BLOCK, 1, 1))


@substrate.jit
def fused_kernel(
    X: S.Tensor((BATCH_SIZE, IN_FEATURES // K_CHUNK, 2, 4), S.u32),
    X_DESC: S.Tensor((4,), S.u32),
    W: S.Tensor((OUT_FEATURES, IN_FEATURES // K_CHUNK, 2, 4), S.u32),
    W_DESC: S.Tensor((4,), S.u32),
    BIAS0: S.Tensor((OUT_FEATURES,), S.bf16),
    Y: S.Tensor((BATCH_SIZE, OUT_FEATURES), S.bf16),
):
    pid_n = S.block_id(0)
    pid_m = S.block_id(1)
    tid = S.thread_id(0)

    warp_id = tid // 64
    lane = tid % 64
    warp_row = warp_id // 2
    warp_col = warp_id % 2

    tile_row_base = pid_m * BLOCK_M + warp_row * WAVE_M
    tile_col_base = pid_n * BLOCK_N + warp_col * WAVE_N

    k_quad = lane // 32
    row_in_wave = lane % 32
    col_in_wave = lane % 32

    shared_a = S.make_shared((WAVES_PER_BLOCK, 2, 64, 4), S.u32)
    shared_b = S.make_shared((WAVES_PER_BLOCK, 2, 64, 4), S.u32)
    acc = S.full((16,), 0.0, S.f32)

    bytes_per_u32 = S.convert(4, S.i32)
    x_row_stride = S.convert(ROW_STRIDE_U32, S.i32)
    w_row_stride = S.convert(ROW_STRIDE_U32, S.i32)
    lane_fragment = S.convert(k_quad * U32S_PER_LANE_FRAGMENT, S.i32)
    first_chunk_words = lane_fragment
    second_chunk_words = S.convert(U32S_PER_CHUNK, S.i32) + lane_fragment

    a_row0 = tile_row_base + row_in_wave
    a_word0 = S.convert(a_row0, S.i32) * x_row_stride + first_chunk_words
    a_vindex0 = a_word0 * bytes_per_u32
    a_vec0 = S.amdgpu.raw_buffer_load_x4(X_DESC, a_vindex0, 0, 0)
    shared_a[warp_id, 0, lane] = a_vec0

    a_word1 = S.convert(a_row0, S.i32) * x_row_stride + second_chunk_words
    a_vindex1 = a_word1 * bytes_per_u32
    a_vec1 = S.amdgpu.raw_buffer_load_x4(X_DESC, a_vindex1, 0, 0)
    shared_a[warp_id, 1, lane] = a_vec1

    b_col0 = tile_col_base + col_in_wave
    b_word0 = S.convert(b_col0, S.i32) * w_row_stride + first_chunk_words
    b_vindex0 = b_word0 * bytes_per_u32
    b_vec0 = S.amdgpu.raw_buffer_load_x4(W_DESC, b_vindex0, 0, 0)
    shared_b[warp_id, 0, lane] = b_vec0

    b_word1 = S.convert(b_col0, S.i32) * w_row_stride + second_chunk_words
    b_vindex1 = b_word1 * bytes_per_u32
    b_vec1 = S.amdgpu.raw_buffer_load_x4(W_DESC, b_vindex1, 0, 0)
    shared_b[warp_id, 1, lane] = b_vec1

    S.syncthreads()

    for k_base in S.range(0, IN_FEATURES, 2 * K_CHUNK):
        a_pack0 = shared_a[warp_id, 0, lane]
        b_pack0 = shared_b[warp_id, 0, lane]
        a_frag0 = S.view(a_pack0, S.Tensor((2, 4, 1), S.bf16))
        b_frag0 = S.view(b_pack0, S.Tensor((2, 4, 1), S.bf16))

        next_k0 = k_base + 2 * K_CHUNK
        next_chunk0 = S.convert((next_k0 // K_CHUNK) * U32S_PER_CHUNK, S.i32)
        a_word = S.convert(a_row0, S.i32) * x_row_stride + next_chunk0 + lane_fragment
        a_vindex = a_word * bytes_per_u32
        a_vec = S.amdgpu.raw_buffer_load_x4(X_DESC, a_vindex, 0, 0)
        shared_a[warp_id, 0, lane] = a_vec

        b_word = S.convert(b_col0, S.i32) * w_row_stride + next_chunk0 + lane_fragment
        b_vindex = b_word * bytes_per_u32
        b_vec = S.amdgpu.raw_buffer_load_x4(W_DESC, b_vindex, 0, 0)
        shared_b[warp_id, 0, lane] = b_vec

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[0], b_frag0[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0[1], b_frag0[1], acc)

        a_pack1 = shared_a[warp_id, 1, lane]
        b_pack1 = shared_b[warp_id, 1, lane]
        a_frag1 = S.view(a_pack1, S.Tensor((2, 4, 1), S.bf16))
        b_frag1 = S.view(b_pack1, S.Tensor((2, 4, 1), S.bf16))

        next_k1 = k_base + 3 * K_CHUNK
        next_chunk1 = S.convert((next_k1 // K_CHUNK) * U32S_PER_CHUNK, S.i32)
        a_word = S.convert(a_row0, S.i32) * x_row_stride + next_chunk1 + lane_fragment
        a_vindex = a_word * bytes_per_u32
        a_vec = S.amdgpu.raw_buffer_load_x4(X_DESC, a_vindex, 0, 0)
        shared_a[warp_id, 1, lane] = a_vec

        b_word = S.convert(b_col0, S.i32) * w_row_stride + next_chunk1 + lane_fragment
        b_vindex = b_word * bytes_per_u32
        b_vec = S.amdgpu.raw_buffer_load_x4(W_DESC, b_vindex, 0, 0)
        shared_b[warp_id, 1, lane] = b_vec

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[0], b_frag1[0], acc)
        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1[1], b_frag1[1], acc)

        S.syncthreads()

    one = S.convert(1.0, S.f32)
    scale = S.convert(SCALING_FACTOR, S.f32)
    col = tile_col_base + col_in_wave
    bias = S.convert(BIAS0[col], S.f32)

    for acc_idx in S.range(16):
        row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        x = acc[acc_idx] + bias
        x = x * (one / (one + S.exp(-x)))
        Y[row, col] = S.convert(x * scale, S.bf16)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._cached_weight_bf16 = None
        self._cached_weight_pack = None
        self._cached_weight_desc = None
        self._cached_bias = None
        self._weight_cache_key = None
        self._weight_desc_key = None
        self._bias_cache_key = None
        self._x_desc = None
        self._cached_x_pack = None
        self._x_pack_key = None
        self._cached_output = None
        self._output_key = None

    def _make_raw_buffer_desc(self, tensor):
        ptr = tensor.data_ptr()
        nbytes = tensor.numel() * tensor.element_size()
        words = [
            ptr & 0xFFFFFFFF,
            (ptr >> 32) & 0xFFFFFFFF,
            nbytes & 0xFFFFFFFF,
            0x00020000,
        ]
        return torch.tensor(words, device=tensor.device, dtype=torch.uint32)

    def _pack_operand(self, tensor_2d):
        packed = tensor_2d.view(torch.uint32).view(tensor_2d.shape[0], CHUNKS_PER_ROW, 4, 2)
        even = packed[:, :, 0::2, :].reshape(tensor_2d.shape[0], CHUNKS_PER_ROW, 4)
        odd = packed[:, :, 1::2, :].reshape(tensor_2d.shape[0], CHUNKS_PER_ROW, 4)
        return torch.stack((even, odd), dim=2).contiguous()

    def _get_x_pack_and_desc(self, x):
        key = (
            x.device.type,
            x.device.index,
        )
        if self._cached_x_pack is None or self._x_pack_key != key:
            self._cached_x_pack = torch.empty(
                (BATCH_SIZE, CHUNKS_PER_ROW, 2, 4), device=x.device, dtype=torch.uint32
            )
            self._x_desc = self._make_raw_buffer_desc(self._cached_x_pack)
            self._x_pack_key = key
        return self._cached_x_pack, self._x_desc

    def _get_weight_pack_and_desc(self, x):
        weight = self.matmul.weight
        pack_key = (
            weight.data_ptr(),
            x.device.type,
            x.device.index,
            x.dtype,
        )
        if self._cached_weight_pack is None or self._weight_cache_key != pack_key:
            self._cached_weight_bf16 = weight.to(device=x.device, dtype=x.dtype).contiguous()
            self._cached_weight_pack = self._pack_operand(self._cached_weight_bf16)
            self._weight_cache_key = pack_key

        desc_key = (
            self._cached_weight_pack.data_ptr(),
            self._cached_weight_pack.device.type,
            self._cached_weight_pack.device.index,
        )
        if self._cached_weight_desc is None or self._weight_desc_key != desc_key:
            self._cached_weight_desc = self._make_raw_buffer_desc(self._cached_weight_pack)
            self._weight_desc_key = desc_key

        return self._cached_weight_pack, self._cached_weight_desc

    def _get_bias(self, x):
        bias = self.matmul.bias
        key = (
            bias.data_ptr(),
            x.device.type,
            x.device.index,
            x.dtype,
        )
        if self._cached_bias is None or self._bias_cache_key != key:
            self._cached_bias = bias.to(device=x.device, dtype=x.dtype).contiguous()
            self._bias_cache_key = key
        return self._cached_bias

    def _get_output(self, x):
        key = (
            x.device.type,
            x.device.index,
            x.dtype,
        )
        if self._cached_output is None or self._output_key != key:
            self._cached_output = torch.empty((BATCH_SIZE, OUT_FEATURES), device=x.device, dtype=x.dtype)
            self._output_key = key
        return self._cached_output

    def forward(self, x):
        if tuple(x.shape) != (BATCH_SIZE, IN_FEATURES):
            raise RuntimeError(f"unsupported input shape: {tuple(x.shape)}")
        if x.dtype != torch.bfloat16:
            raise RuntimeError(f"unsupported input dtype: {x.dtype}")
        if self.scaling_factor != SCALING_FACTOR:
            raise RuntimeError(f"unsupported scaling factor: {self.scaling_factor}")

        x_bf16 = x.contiguous()
        x_pack, x_desc = self._get_x_pack_and_desc(x_bf16)
        x_view = x_bf16.view(torch.uint32).view(BATCH_SIZE, CHUNKS_PER_ROW, 4, 2)
        x_pack[:, :, 0].copy_(x_view[:, :, 0::2, :].reshape(BATCH_SIZE, CHUNKS_PER_ROW, 4))
        x_pack[:, :, 1].copy_(x_view[:, :, 1::2, :].reshape(BATCH_SIZE, CHUNKS_PER_ROW, 4))
        w_pack, w_desc = self._get_weight_pack_and_desc(x)
        bias = self._get_bias(x)
        y = self._get_output(x)
        fused_kernel[_launch](x_pack, x_desc, w_pack, w_desc, bias, y, num_warps=WAVES_PER_BLOCK)
        return y
