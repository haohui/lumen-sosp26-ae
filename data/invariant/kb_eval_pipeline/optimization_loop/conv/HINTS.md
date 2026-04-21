## 1: Prompt 1

Optimize the substrate Conv2D kernel in xxx. Replace the scalar implicit-GEMM accumulation in `_igemm_kernel` with a 4-wave MFMA kernel using `S.amdgpu.mfma_32x32x8_bf16_f32`. Keep the existing launch structure and public behavior intact. Rename the optimized entrypoint from `conv2d_asym_naive` to `conv2d_asym`, update `__all__`, and update the benchmark/test callsites to use `conv2d_asym`.

  MFMA invariants:
  - Keep the block tile fixed at `128 x 128`.
  - Interpret the 4 warps as a fixed `2 x 2` warp grid:
  - Keep the MFMA work per warp fixed at a `64 x 64` tile built as a `2 x 2` array of `32 x 32` MFMA tiles.

  Operand-fragment invariants:
  - Use:
    - `lane_col = lane % 32`
    - `lane_k_base = (lane // 32) * 4`
  - Advance K in MFMA-sized chunks:
    - `k = k_tile * 8 + lane_k_base + e`, where `e in [0, 4)`
  - For A, for each warp-row subtile `tm in [0, 2)`:
    - `m = group_m_base + warp_row * 64 + tm * 32 + lane_col`
    - `a_frag[tm, e]` is the 4-element BF16 fragment for that `(m, k:k+4)` slice
  - For B, for each warp-col subtile `tn in [0, 2)`:
    - `n = group_n_base + warp_col * 64 + tn * 32 + lane_col`
    - `b_frag[tn, e]` is the 4-element BF16 fragment for that `(k:k+4, n)` slice
  - Do not invent a different swizzle, lane permutation, or fragment packing rule.

  Accumulator invariants:
  - Keep one MFMA accumulator per `32 x 32` subtile:
    - `acc[tm, tn] = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[tm], b_frag[tn], acc[tm, tn])`
  - Build the per-warp `64 x 64` tile only by applying the fixed subtile offsets `tm * 32` and `tn * 32`.
  - Build the full block tile only by applying the fixed warp-grid offsets `warp_row * 64` and `warp_col * 64`.

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
  - Do not add LDS staging, vectorized loads, async copies, or a different tile shape as part of this change.
  - Preserve the supported behavior of the existing kernel.
  - Do not look at other commits in the repo.

  Update callsites:
  - /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py
  - /workspace/substrate/benchmark/conv2d/bench_conv2d.py

  Validation:
  - Write `path/to/case.txt` so it contains exactly one line: the absolute path of the current round directory.
  - Run `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py`

- Make the optimized path cudagraph-safe: never build descriptor / metadata device tensors inside `forward()`. Prebuild or cache them and reuse them; only rebuild if the underlying storage pointer changes.
- Do not use torch native compute anywhere in `output_model_new.py` to perform convolution, multiplication, or linear algebra, including fallback branches.
- The optimized kernel must actually issue MFMA instructions in the substrate kernel; a solution that does not use MFMA is not acceptable.
- Do not use git to try to find any old files!!!
Strict constraints:
  - Do not browse the web.
  - Do not search online for documentation, examples, repos, or references.
  - Do not use any network access at all.
  - Use only the files already present in the workspace dir.
  - Do not read any other kernel from anywhere in the space!!!


## 2: Prompt 2

optimize the substrate kernel in xxx.

Implement Split-K reduction for the MFMA Conv2D kernel with `SPLIT_K_SLICES = 2`. Keep the existing `128 x 128` output tile, 4-wave block, and per-wave `S.amdgpu.mfma_32x32x8_bf16_f32` execution unchanged. Extend the launch grid in `x` by `SPLIT_K_SLICES`, let each split compute a partial fp32 accumulation for the same `(group_m, group_n)` tile, and reduce the partial sums into a shared fp32 workspace with `S.amdgpu.buffer_atomic_add_f32`. After all split-K partial sums are written, run a second kernel that converts the fp32 workspace back to the final bf16 NCHW output.

Split-K invariants:
  Block decomposition invariant:
  `linear_block_id = tile_block_id * SPLIT_K_SLICES + split_k_id`, where `tile_block_id` selects the same `(group_m, group_n)` tile as the baseline kernel and `split_k_id in [0, SPLIT_K_SLICES)` only chooses the K-slice. Do not change M/N tile ownership when adding split-K.

  Channel partition invariant:
  `c_per_split = ceil_div(in_channels, SPLIT_K_SLICES)`, `c_start = split_k_id * c_per_split`, `c_end = min(in_channels, c_start + c_per_split)`. The split ranges must be contiguous, disjoint, channel-aligned, and their union must cover `[0, in_channels)`.

  K linearization invariant:
  Inside the split kernel, flatten K as `k_idx = c * kernel_area + kh * kernel_w + kw`, so `c = k_idx // kernel_area`, `spatial = k_idx % kernel_area`, `kh = spatial // kernel_w`, and `kw = spatial % kernel_w`. Do not use the old `c = k_idx % in_channels` mapping; split boundaries must stay channel-aligned.

  Operand layout invariant:
  Read the input and weights directly as NCHW / OIHW tensors, i.e. `input[batch, c, h, w]` and `weight[out_channel, c, kh, kw]`. Do not rely on NHWC / OHWI pre-permutation once split-K is introduced.

  Accumulator/output invariant:
  Per-split MFMA math remains fp32, and no split may write bf16 output directly. All splits must reduce into a fp32 workspace with GEMM-major indexing `linear_idx = row * gemm_n + col`, where `row = batch * hw_out + hw_idx` and `col = out_channel`.

  Finalization invariant:
  The final bf16 output keeps the original NCHW layout. A separate store kernel must remap the fp32 GEMM-major workspace back to `out[batch, out_channel, hw_idx]` only after the split reductions are complete.

MFMA and writeback invariants:
  Keep the existing MFMA lane mapping and accumulator-to-output mapping unchanged. For every wave tile, `col = tile_col_base + (lane % 32)` and `row = tile_row_base + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)`. Split-K changes reduction ownership, not the MFMA fragment layout.

Preserve the 4-way writeback regrouping before the atomic add. `row_local` must map to `writeback_group = (row_local // 64) * 2 + ((row_local % 32) // 16)` and `group_row = (row_local % 16) + 16 * ((row_local % 64) // 32)`. The inverse mapping used by the atomic writeback pass must reconstruct the same `row_local`. Do not invent a different regrouping if you still stage through shared memory.

Notes:

1. The current snapshot of the repo under /workspace/substrate has MFMA examples in `/workspace/substrate/test/examples/gemm/amdgpu/test_gemm_mfma.py`.
- Write `path/to/case.txt` so it contains exactly one line: the absolute path of the current round directory.
- Run `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py`
- Make the optimized path cudagraph-safe: never build descriptor / metadata device tensors inside `forward()`. Prebuild or cache them and reuse them; only rebuild if the underlying storage pointer changes.
- Do not use torch native compute anywhere in `output_model_new.py` to perform convolution, multiplication, or linear algebra, including fallback branches.
- The optimized kernel must actually issue MFMA instructions in the substrate kernel; a solution that does not use MFMA is not acceptable.
- Do not use git to try to find any old files!!!
Strict constraints:
  - Do not browse the web.
  - Do not search online for documentation, examples, repos, or references.
  - Do not use any network access at all.
  - Use only the files already present in the workspace dir.
  - Do not read any other kernel from anywhere in the space!!!
