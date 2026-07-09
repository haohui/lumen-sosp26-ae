## 1: Prompt 1

Implement the Conv2D path with AveLang kernels.

Replace scalar implicit-GEMM accumulation with a 4-wave MFMA Conv2D kernel
using `al.amdgpu.mfma_32x32x8_bf16_f32`. Preserve the public behavior of the
reference `Model.forward`, including input and output layout, dtype behavior,
stride, padding, dilation, groups, and bias semantics when they are present in
the target architecture.

MFMA invariants:
  - Keep the block tile fixed at `128 x 128`.
  - Interpret the 4 warps as a fixed `2 x 2` warp grid.
  - Keep the MFMA work per warp fixed at a `64 x 64` tile built as a `2 x 2`
    array of `32 x 32` MFMA tiles.

Operand-fragment invariants:
  - Use:
    - `lane_col = lane % 32`
    - `lane_k_base = (lane // 32) * 4`
  - Advance K in MFMA-sized chunks:
    - `k = k_tile * 8 + lane_k_base + e`, where `e in [0, 4)`
  - For A, for each warp-row subtile `tm in [0, 2)`:
    - `m = group_m_base + warp_row * 64 + tm * 32 + lane_col`
    - `a_frag[tm, e]` is the 4-element BF16 fragment for that `(m, k:k+4)`
      slice of the implicit Conv2D input matrix.
  - For B, for each warp-col subtile `tn in [0, 2)`:
    - `n = group_n_base + warp_col * 64 + tn * 32 + lane_col`
    - `b_frag[tn, e]` is the 4-element BF16 fragment for that `(k:k+4, n)`
      slice of the implicit Conv2D weight matrix.
  - Do not invent a different swizzle, lane permutation, or fragment packing
    rule.

Accumulator invariants:
  - Keep one MFMA accumulator per `32 x 32` subtile:
    - `acc[tm, tn] = al.amdgpu.mfma_32x32x8_bf16_f32(a_frag[tm], b_frag[tn], acc[tm, tn])`
  - Build the per-warp `64 x 64` tile only by applying the fixed subtile offsets
    `tm * 32` and `tn * 32`.
  - Build the full block tile only by applying the fixed warp-grid offsets
    `warp_row * 64` and `warp_col * 64`.

Writeback invariants:
  - The MFMA accumulator layout is fixed and must be unpacked exactly as:
    - `tile_row_base = group_m_base + warp_row * 64 + tm * 32`
    - `tile_col_base = group_n_base + warp_col * 64 + tn * 32`
    - `col = tile_col_base + (lane % 32)`
    - `row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)`
  - Fixed mapping:
    - `acc[tm, tn, acc_idx] -> output(row, col)`
  - Do not use a contiguous `acc_idx -> row` mapping.

Scope invariants:
  - This is an MFMA-only transformation.
  - Do not add LDS staging, vectorized loads, async copies, or a different tile
    shape as part of this change.
  - Preserve the supported behavior of the existing kernel.
  - Do not look at other commits in the repo.

## 2: Prompt 2

Implement Split-K reduction for the MFMA Conv2D kernel with
`SPLIT_K_SLICES = 2`. Keep the existing `128 x 128` output tile, 4-wave block,
and per-wave `al.amdgpu.mfma_32x32x8_bf16_f32` execution unchanged. Extend the
launch grid in `x` by `SPLIT_K_SLICES`, let each split compute a partial FP32
accumulation for the same `(group_m, group_n)` tile, and reduce the partial sums
into a shared FP32 workspace. Use an AveLang-supported atomic add if one is
available for the target dtype; otherwise use a separate reduction/finalization
kernel. After all split-K partial sums are written, run a second AveLang kernel
that converts the FP32 workspace back to the final output layout expected by the
reference model.

Split-K invariants:
  Block decomposition invariant:
  `linear_block_id = tile_block_id * SPLIT_K_SLICES + split_k_id`, where
  `tile_block_id` selects the same `(group_m, group_n)` tile as the baseline
  kernel and `split_k_id in [0, SPLIT_K_SLICES)` only chooses the K-slice. Do
  not change M/N tile ownership when adding split-K.

  Channel partition invariant:
  `c_per_split = ceil_div(in_channels, SPLIT_K_SLICES)`,
  `c_start = split_k_id * c_per_split`, and
  `c_end = min(in_channels, c_start + c_per_split)`. The split ranges must be
  contiguous, disjoint, channel-aligned, and their union must cover
  `[0, in_channels)`.

  K linearization invariant:
  Inside the split kernel, flatten K as
  `k_idx = c * kernel_area + kh * kernel_w + kw`, so
  `c = k_idx // kernel_area`, `spatial = k_idx % kernel_area`,
  `kh = spatial // kernel_w`, and `kw = spatial % kernel_w`. Do not use
  `c = k_idx % in_channels`; split boundaries must stay channel-aligned.

  Operand layout invariant:
  Read the input and weights directly in the layout used by the reference model,
  typically `input[batch, c, h, w]` and `weight[out_channel, c, kh, kw]` for
  NCHW / OIHW Conv2D. Do not rely on NHWC / OHWI pre-permutation once split-K is
  introduced unless the reference model itself exposes that layout.

  Accumulator/output invariant:
  Per-split MFMA math remains FP32, and no split may write the final reduced
  output directly. All splits must reduce into an FP32 workspace with GEMM-major
  indexing `linear_idx = row * gemm_n + col`, where
  `row = batch * hw_out + hw_idx` and `col = out_channel`.

  Finalization invariant:
  The final output keeps the original reference layout. A separate store kernel
  must remap the FP32 GEMM-major workspace back to the reference output layout
  only after the split reductions are complete.

MFMA and writeback invariants:
  Keep the existing MFMA lane mapping and accumulator-to-output mapping
  unchanged. For every wave tile, `col = tile_col_base + (lane % 32)` and
  `row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)`.
  Split-K changes reduction ownership, not the MFMA fragment layout.

Preserve the 4-way writeback regrouping before reducing partial sums.
`row_local` must map to
`writeback_group = (row_local // 64) * 2 + ((row_local % 32) // 16)` and
`group_row = (row_local % 16) + 16 * ((row_local % 64) // 32)`. The inverse
mapping used by the partial-sum writeback or finalization pass must reconstruct
the same `row_local`. Do not invent a different regrouping if you still stage
through shared memory.
